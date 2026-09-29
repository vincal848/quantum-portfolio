"""SUPERSEDED. Kept for reference; this is one of the original coursework scripts.

None of these three files run on a current install. Qiskit 2.x removed the V1
`Sampler` and `BackendSamplerV2` that Quantum_optimizer_RealData_Classes.py imports
from qiskit.primitives, and yfinance no longer exposes an 'Adj Close' column now
that auto_adjust defaults to True.

Problems the rewrite documents and tests:

1. Bit ordering. `[i for i, bit in enumerate(binary_decision) if bit == "1"]` reads
   the bitstring left to right, but Qiskit is little-endian - qubit 0 is the
   RIGHTMOST character. The reported selection was mirrored: asset 3 named when the
   answer was asset 0. Invisible on a symmetric solution, which the toy example had.
2. No reference answer. NumPyMinimumEigensolver returns a selection and nothing
   checks whether it is the optimum, or whether a greedy pass would match it.
3. The Sharpe ratio divides excess return by the VARIANCE rather than the standard
   deviation, which inflates it by roughly 1/sigma.
4. It compares a DAILY mean return against an ANNUAL risk-free rate.
5. The efficient-frontier loop appends `format(x, ".3f")` strings and then plots
   them, so both axes are categorical text rather than numbers.
6. The random covariance matrix is symmetrized uniform noise and is not positive
   semi-definite - two negative eigenvalues at the seed used.
7. NumPyMinimumEigensolver is exact classical diagonalization of the Hamiltonian,
   i.e. an exhaustive 2^n search. It is not a quantum method.

The working version is portfolio.py, solvers.py, metrics.py, data.py and run.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_aer import Aer
from scipy.stats import norm
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler,Estimator, Sampler, StatevectorEstimator, BackendSamplerV2

class QuantumPortfolioOptimizer:
    def __init__(self, stocks, start_date, end_date, lambda_risk=0.5, max_assets=3, initial_capital=1):
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        self.lambda_risk = lambda_risk
        self.max_assets = max_assets
        self.initial_capital = initial_capital
        self.returns = None
        self.cov_matrix = None
        
        self.qp = None

    def fetch_data(self):
        """Fetch stock data and calculate returns and covariance."""
        data = yf.download(self.stocks, start=self.start_date, end=self.end_date)['Adj Close']
        self.returns = data.pct_change().mean().values
        self.cov_matrix = data.pct_change().cov().values
        print("Data fetched successfully.")
    
    def setup_quadratic_program(self):
        """Set up the quadratic program."""
        self.qp = QuadraticProgram()
        num_assets = len(self.stocks)

        
        for i in range(num_assets):
            self.qp.binary_var(f"x{i}")

        
        linear_coeffs = {f"x{i}": -self.returns[i] for i in range(num_assets)}
        quadratic_coeffs = {
            (f"x{i}", f"x{j}"): self.lambda_risk * self.cov_matrix[i][j]
            for i in range(num_assets) for j in range(num_assets)
        }
        self.qp.minimize(linear=linear_coeffs, quadratic=quadratic_coeffs)

        
        self.qp.linear_constraint(
            linear={f"x{i}": 1 for i in range(num_assets)},
            sense="==",
            rhs=self.max_assets,
            name="budget_constraint"
        )
        print("Quadratic program set up successfully.")

    def solve(self, solver="NumPy"):
        """Solve the quadratic program."""
        qubo_converter = QuadraticProgramToQubo()
        qp_qubo = qubo_converter.convert(self.qp)

        operator, offset = qp_qubo.to_ising()

        if solver == "NumPy":
            solver_instance = NumPyMinimumEigensolver()
        elif solver == "QAOA":
            backend = Aer.get_backend("aer_simulator")  
            sampler = BackendSamplerV2(backend)  
            optimizer = COBYLA(maxiter=100)
            solver_instance = QAOA(sampler=sampler, optimizer=optimizer, reps=2)
        else:
            raise ValueError("Invalid solver. Choose 'NumPy' or 'QAOA'.")

        
        result = solver_instance.compute_minimum_eigenvalue(operator)

        
        probabilities = result.eigenstate  
        max_prob_index = max(probabilities, key=probabilities.get)  
        binary_decision = f"{max_prob_index:0{len(self.stocks)}b}"

        
        selected_assets = [i for i, bit in enumerate(binary_decision) if bit == "1"]
        allocation = [
            self.initial_capital / len(selected_assets) if i in selected_assets else 0
            for i in range(len(self.stocks))
        ]
        portfolio_return = sum(self.returns[i] * allocation[i] for i in range(len(self.stocks)))
        portfolio_risk = np.dot(allocation, np.dot(self.cov_matrix, allocation))

        return {
            "selected_assets": [self.stocks[i] for i in selected_assets],
            "allocation": allocation,
            "portfolio_return": portfolio_return,
            "portfolio_risk": portfolio_risk,
        }

    def visualize(self, allocation):
        """Visualize portfolio allocation."""
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(self.stocks)), allocation, color="blue", alpha=0.7)
        plt.xlabel("Stock Ticker")
        plt.ylabel("Capital Allocation")
        plt.title("Portfolio Allocation")
        plt.xticks(range(len(self.stocks)), self.stocks)
        plt.show()

    def sharpe_ratio(self, portfolio_return, portfolio_risk, risk_free_rate=0.02):
        """Calculate the Sharpe ratio."""
        return (portfolio_return - risk_free_rate) / portfolio_risk

# Example Usage
optimizer = QuantumPortfolioOptimizer(
    stocks=["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"],
    start_date="2020-01-01",
    end_date="2023-01-01",
    lambda_risk=0.5,
    max_assets=3
)

optimizer.fetch_data()
optimizer.setup_quadratic_program()
result = optimizer.solve(solver="QAOA")

print("Selected Assets:", result["selected_assets"])
print("Capital Allocation:", result["allocation"])
print("Portfolio Return:", result["portfolio_return"]*100)
print("Portfolio Risk:", result["portfolio_risk"])

optimizer.visualize(result["allocation"])

sharpe = optimizer.sharpe_ratio(result["portfolio_return"], result["portfolio_risk"])
print("Sharpe Ratio:", sharpe)
