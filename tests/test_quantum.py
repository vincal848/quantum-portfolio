"""The statevector engine against dense linear algebra, and the solvers' contracts."""

import numpy as np
import pytest
from scipy.linalg import expm

from qportfolio import data, models, solvers
from qportfolio.problem import bit_table, weight_k_states
from qportfolio.quantum import WarmStartMixer, XMixer, XYMixer, cvar, evolve

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
I2 = np.eye(2)


def op_on(n, ops):
    """Dense operator with ops[q] on qubit q (little-endian: qubit 0 is last factor)."""
    out = np.array([[1.0 + 0j]])
    for q in reversed(range(n)):
        out = np.kron(out, ops.get(q, I2))
    return out


def test_x_mixer_matches_the_matrix_exponential():
    n, beta = 4, 0.37
    rng = np.random.default_rng(0)
    psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    H = sum(op_on(n, {q: X}) for q in range(n))
    assert XMixer(n).apply(psi, beta) == pytest.approx(expm(-1j * beta * H) @ psi)


def test_xy_mixer_matches_dense_gates_and_keeps_hamming_weight():
    n, k, beta = 5, 2, 0.61
    mix = XYMixer(n, k, "ring")
    rng = np.random.default_rng(1)
    sub = rng.normal(size=len(mix.states)) + 1j * rng.normal(size=len(mix.states))
    full = np.zeros(2 ** n, dtype=complex)
    full[mix.states] = sub
    for a, b in [(0, 1), (2, 3), (4, 0), (1, 2), (3, 4)]:      # the same parity order
        G = (op_on(n, {a: X, b: X}) + op_on(n, {a: Y, b: Y})) / 2
        full = expm(-1j * beta * G) @ full
    got = mix.apply(sub, beta)
    assert got == pytest.approx(full[mix.states])
    outside = np.setdiff1d(np.arange(2 ** n), weight_k_states(n, k))
    assert np.abs(full[outside]).max() < 1e-12
    assert np.linalg.norm(got) == pytest.approx(np.linalg.norm(sub))


def test_dicke_and_warm_start_initial_states():
    mix = XYMixer(6, 3)
    assert len(mix.initial()) == 20
    assert np.abs(mix.initial()) ** 2 == pytest.approx(np.full(20, 1 / 20))
    c = np.array([0.3, 0.6, 0.5])
    warm = WarmStartMixer(3, c, epsilon=0.0)
    probs = np.abs(warm.initial()) ** 2
    # Product state: P(x_i = 1) equals the relaxed value.
    marg = bit_table(3).T @ probs
    assert marg == pytest.approx(c)


def test_warm_start_mixer_has_the_initial_state_as_eigenstate():
    """The initial state is an eigenstate of the warm mixer, so the mixer alone only
    changes its global phase - the warm-start analogue of |+> under X."""
    warm = WarmStartMixer(3, [0.3, 0.6, 0.4])
    psi = warm.initial()
    out = warm.apply(psi, 0.8)
    assert abs(np.vdot(psi, out)) == pytest.approx(1.0)


def test_cvar():
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    e = np.array([4.0, 1.0, 3.0, 2.0])
    assert cvar(probs, e, 1.0) == pytest.approx(probs @ e)
    assert cvar(probs, e, 0.2) == pytest.approx(1.0)              # all mass at e = 1
    assert cvar(probs, e, 0.4) == pytest.approx((0.2 * 1 + 0.2 * 2) / 0.4)
    assert cvar(probs, e, 0.4) <= cvar(probs, e, 1.0)


def test_evolve_with_no_layers_is_the_initial_state():
    mix = XMixer(3)
    assert evolve(np.zeros(8), mix, [], []) == pytest.approx(mix.initial())


# --- solvers ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def problem():
    return models.selection(data.synthetic_market(8, seed=5), k=3)


def test_exact_finds_the_minimum(problem):
    best, ref = solvers.exact(problem)
    values = [problem.objective(x) for x in bit_table(8) if x.sum() == 3]
    assert best.objective == pytest.approx(min(values))
    assert ref.worst == pytest.approx(max(values))
    assert ref.ratio(ref.best) == 1.0 and ref.ratio(ref.worst) == 0.0


def test_simulated_annealing_finds_the_optimum_on_a_small_problem(problem):
    _, ref = solvers.exact(problem)
    sa = solvers.simulated_annealing(problem, seed=1)
    assert sa.feasible and sa.objective == pytest.approx(ref.best)


@pytest.mark.parametrize("mixer", ["x", "xy", "warm"])
def test_qaoa_beats_its_starting_point_and_finds_the_optimum(problem, mixer):
    _, ref = solvers.exact(problem)
    r = solvers.qaoa(problem, ref, p=2, mixer=mixer, restarts=2)
    assert r.feasible and r.objective == pytest.approx(ref.best)
    losses = r.info["loss_by_depth"]
    assert all(b <= a + 1e-9 for a, b in zip(losses, losses[1:])), "INTERP is monotone"


def test_xy_qaoa_never_leaves_the_feasible_set(problem):
    _, ref = solvers.exact(problem)
    r = solvers.qaoa(problem, ref, p=2, mixer="xy", restarts=1)
    assert r.info["p_feasible"] == pytest.approx(1.0)
    assert r.info["hilbert_dim"] == 56          # C(8,3), not 2^8


def test_xy_mixer_concentrates_more_than_the_x_mixer(problem):
    """The headline effect, pinned so it cannot silently regress."""
    _, ref = solvers.exact(problem)
    x = solvers.qaoa(problem, ref, p=3, mixer="x", restarts=2)
    xy = solvers.qaoa(problem, ref, p=3, mixer="xy", restarts=2)
    assert xy.info["p_optimal"] > 5 * x.info["p_optimal"]
    assert xy.info["amplification"] > 1


def test_slower_adiabatic_evolution_gets_closer_to_the_ground_state(problem):
    """The two best portfolios here differ by 4% of a standard deviation in
    energy, a small gap, so convergence is slow but steady, as the adiabatic
    theorem says it should be."""
    _, ref = solvers.exact(problem)
    p = [solvers.adiabatic(problem, ref, steps=4 * T, total_time=T, mixer="xy")
         .info["p_optimal"] for T in (2, 10, 100)]
    assert p[0] < p[1] < p[2]
    assert p[2] > 0.5 > 10 * ref.optimal_states.size / ref.n_feasible


def test_xy_mixer_is_refused_without_a_fixed_weight():
    p = models.allocation(data.synthetic_market(3, seed=1))
    _, ref = solvers.exact(p)
    with pytest.raises(ValueError):
        solvers.qaoa(p, ref, mixer="xy")


def test_qaoa_is_reproducible(problem):
    _, ref = solvers.exact(problem)
    a = solvers.qaoa(problem, ref, p=2, seed=3, restarts=2)
    b = solvers.qaoa(problem, ref, p=2, seed=3, restarts=2)
    assert a.info["p_optimal"] == pytest.approx(b.info["p_optimal"])
    assert list(a.x) == list(b.x)
