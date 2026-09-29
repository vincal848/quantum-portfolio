# Background

Where this project came from, the classical/quantum comparison table, and my notes
on QAOA and adiabatic quantum computing. This is my original write-up, kept as
written.

For what the code does with it, and what the measured results are, see the
[README](../README.md).

---

The origin of this project was extension on a project in a senior physics course *PHYS 4315: Introduction to Quantum Computing*, which is crosslisted as a masters course in electrical engineering *ECE 5332: Special Topics in Electrical Engineering*. The majority of the background and coursework was from the text ***Introduction to Classical and Quantum Computing*** by Thomas G. Wong. The text explores classical computing and its quantum counterparts. This leads into the eventual work of Nielsen and Chuang's ***Quantum Computation and Quantum Information***, which we covered sections of along with some of Schrödinger's equation for algorithmic preparation.

This is a brief overview of differences in the classical and quantum, listed from Wong's text.

| Concept                  | Classical               | Quantum                |
|--------------------------|-------------------------|------------------------|
| Fundamental Unit         | Bit                    | Qubit                 |
| Gates                   | Logic Gates            | Unitary Gates          |
| Gates Reversible         | Sometimes              | Always                 |
| Universal Gate Set       | NAND                 | H, T, CNOT           |
| Programming Language     | Verilog                | OpenQASM               |
| Algebra                  | Boolean                | Linear                 |
| Error Correcting Code    | Repetition Code        | Shor Code              |
| Complexity Class         | P                      | BQP                    |
| Strong Church-Turing Thesis | Supports           | Possibly Violates       |

# Quantum Approximate Optimization Algorithm

## Solving a Network

The way I understood this to work was to consider nodes as representing each asset in a set, where the edges could represent items from covariance to high-correlation or other ways assets may be connected. The QAOA is best in this case as we can then consider a MaxCut problem, which I primarily referenced from [here](https://www.mustythoughts.com/quantum-approximate-optimization-algorithm-explained).

The interesting start to QAOA is with a utility or cost function represented by a Hamiltonian matrix. From this we initiate a second Hamiltonian that can operate as the "exploration" figure on the cost function.
In the standard version of these, we have two angles which we are trying to determine to produce the eventual solution. We can call the former an $H_0$ and the latter $H_E$.

For this set up we use Dirac notation starting with an initial state, usually $|0...0 \rangle$ which we define for the initial state $|s \rangle$. Considering the angles, $\beta, \text{ and } \gamma$, we form the state:

$$|\beta_n, \gamma_n \rangle = U(H_E, \beta_n) U(H_0, \gamma_n) ... U(H_E, \beta_1) U(H_0, \gamma_1) | s \rangle $$

where p is the number of steps taken in applying $U(H_E, \beta) U(H_), \gamma)$ onto the state. We act on each period by starting with some initial parameters, measuring the result, and update the angles to approach a solution.

# Adiabatic Quantum Computing (AQC)

There are two ways of Considering the background, there is an unknown Hamiltonian that is composed of the cost curve and a Hamiltonian that is known. The unknown is $H_C$ and the known is $H_S$. From this, we hope to match the known to the unknown by applying the above. So, we can form a new process as $H(\alpha) = (1 - \alpha) H_s + \alpha H_C$, which results in $H_S$ when $\alpha = 1$ and $H_C$ when $\alpha = 0$. The system evolves as we increment $\alpha$ that we define as $\alpha = \frac{n}{N}$. 

## Conversion of Real-World Problems into Hamiltonian

If we can not encode data into readable information, then we can encode the problems using *Ising Models*. These models encode values in a method than summations inputs that desire to be in opposite spins, by a negative weight. This can be modeled as:

$$ H(\sigma) = - \sum{\langle i,j \rangle}{} J_{i,j} \sigma_i \sigma_j$$

From this, *i* represents a particle and *j* represents a particle and the $J_{i,j}$ value represents the strength between particles.

* Note: The approach called *Quantum Annealing* is an imperfect implementation of the AQC process.


# Future Interests: Quantum Coins & Quantum Walks
