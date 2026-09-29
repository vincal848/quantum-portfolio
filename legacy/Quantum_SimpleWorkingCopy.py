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
from qiskit_optimization import QuadraticProgram
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

def quantum_portfolio_optimization(returns, covariance, lambda_risk):
    
    qp = QuadraticProgram()

    num_assets = len(returns)
    for i in range(num_assets):
        qp.binary_var(f"x{i}")

    
    linear_coeffs = {f"x{i}": -returns[i] for i in range(num_assets)}  
    quadratic_coeffs = {
        (f"x{i}", f"x{j}"): lambda_risk * covariance[i][j]
        for i in range(num_assets) for j in range(num_assets)
    }
    qp.minimize(linear=linear_coeffs, quadratic=quadratic_coeffs)


    operator, offset = qp.to_ising()

    
    solver = NumPyMinimumEigensolver()
    result = solver.compute_minimum_eigenvalue(operator)
    
    
    statevector = result.eigenstate.data
    probabilities = np.abs(statevector) ** 2  
    max_prob_index = np.argmax(probabilities)  
    binary_decision = f"{max_prob_index:0{num_assets}b}"  

    selected_assets = [i for i, bit in enumerate(binary_decision) if bit == "1"]

    portfolio_return = sum(returns[i] for i in selected_assets)
    portfolio_risk = sum(
        covariance[i][j] for i in selected_assets for j in selected_assets
    )

    return {
        "selected_assets": selected_assets,
        "portfolio_return": portfolio_return,
        "portfolio_risk": portfolio_risk,
        "minimum_value": result.eigenvalue + offset,
    }

returns = [0.1, 0.2]
covariance = [[0.005, 0.002], [0.002, 0.006]]
lambda_risk = 0.5

result = quantum_portfolio_optimization(returns, covariance, lambda_risk)

print("Selected Assets:", result["selected_assets"])
print("Portfolio Return:", result["portfolio_return"])
print("Portfolio Risk:", result["portfolio_risk"])
print("Minimum Value:", result["minimum_value"])
print("Minimum Value:", result["minimum_value"])
