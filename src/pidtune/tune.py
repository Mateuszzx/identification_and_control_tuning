"""Tuning of the PID defined in :mod:`pid`.

``method`` selects the search. ``powell`` is a local bound-constrained search
started from the low, middle, and high end of each free gain. ``pygad`` is a
genetic search of the same box. ``pso`` is a global-best particle swarm of
that box. All three score a candidate by the integral of squared setpoint
error from a simulation of :meth:`pid.controller.PID.step`.
"""

import logging
from dataclasses import dataclass
from itertools import product

import numpy as np
import pygad
from scipy.optimize import minimize
from tqdm.auto import tqdm

from pid import Gains, Limits
from pidtune.loop import simulate, steps
from plantid.plant import Plant

log = logging.getLogger("pidtune")


def load_plant(path) -> Plant:
    """Read a plant JSON file written by ``identify.py``.

    Parameters
    ----------
    path : str or pathlib.Path
        Plant JSON.

    Returns
    -------
    Plant
        Plant rebuilt by :meth:`plantid.plant.Plant.load`.
    """
    return Plant.load(path)


@dataclass(frozen=True)
class Solution:
    """Gains chosen by the search, and the waveform they were scored on.

    Attributes
    ----------
    gains : pid.controller.Gains
        Selected ``Kp``, ``Ki``, and ``Kd``.
    ise : float
        Integral of squared setpoint error, weighted by ``we``.
    reference : numpy.ndarray
        Setpoint the gains were scored on.
    response : numpy.ndarray
        Simulated plant output.
    output : numpy.ndarray
        Controller command.
    """

    gains: Gains
    ise: float
    reference: np.ndarray
    response: np.ndarray
    output: np.ndarray

    def __str__(self) -> str:
        """Text summary of the gains, ISE, and worst end-of-hold error.

        Returns
        -------
        str
            Two lines: the gains, then ISE and the largest ``|e|`` at the
            end of a hold.
        """
        gains = self.gains
        worst = max(_final_errors(self.reference, self.response), default=0.0)
        return (
            f"Kp = {gains.kp:.4g}   Ki = {gains.ki:.4g}   Kd = {gains.kd:.4g}\n"
            f"ISE = {self.ise:.4g}   |e|_end {worst:.3f}"
        )


def tune(
    plant: Plant,
    *,
    levels,
    hold: float,
    dt: float,
    kp: tuple[float, float],
    ki: tuple[float, float],
    kd: tuple[float, float] = (0.0, 0.0),
    limits: Limits | None = None,
    we: float = 1.0,
    method: str = "pygad",
    generations: int = 40,
    population: int = 24,
    parents: int | None = None,
    selection: str = "sss",
    crossover: str = "single_point",
    mutation: str = "random",
    mutation_by_replacement: bool = True,
    elitism: int = 1,
    inertia: float = 0.7,
    cognitive: float = 1.5,
    social: float = 1.5,
    maxfev: int = 40,
    xtol: float = 1e-3,
    ftol: float = 1e-3,
    seed: int | None = 1,
) -> Solution:
    """Search ``kp``, ``ki``, and ``kd``.

    Each value in ``levels`` is held for ``hold`` seconds. ``powell``,
    ``pygad``, and ``pso`` minimize the same simulated ISE and stay inside
    the caller's gain limits. An interactive terminal shows Powell
    evaluations, PyGAD generations, or particle-swarm iterations. The bar
    stays hidden when stderr is not a terminal. Dead zone and accumulator
    live in :mod:`pid` and are not part of this search.

    Parameters
    ----------
    plant : plantid.plant.Plant
        Plant to control. It is not identified here.
    levels : sequence of float
        Setpoint waveform, one value per hold.
    hold : float
        Seconds each level is held.
    dt : float
        Sample time in seconds, from ``tune.dt`` in the config.
    kp, ki, kd : tuple of float
        ``(low, high)`` search interval. One number, or equal ends, fixes
        that gain. ``kd`` defaults to ``(0, 0)``.
    limits : pid.controller.Limits, optional
        Controller limits from :class:`pid.controller.Limits`. Default is that class.
    we : float, optional
        Weight on the integral of squared error. Default is 1.
    method : {"powell", "pygad", "pso"}, optional
        Search algorithm. Default is ``"pygad"``.
    generations : int, optional
        PyGAD generations, or particle-swarm iterations. Ignored by Powell.
        Default is 40.
    population : int, optional
        PyGAD population, or particle-swarm size. Ignored by Powell.
        Default is 24.
    parents : int or None, optional
        Parents mated each generation. ``None`` uses one third of
        ``population``, at least 2. Ignored by Powell.
    selection : {"sss", "rws", "sus", "rank", "random", "tournament"}, optional
        PyGAD parent selection. Ignored by Powell. Default is ``"sss"``.
    crossover : {"single_point", "two_points", "uniform", "scattered"}, optional
        PyGAD crossover. Ignored by Powell. Default is ``"single_point"``.
    mutation : {"random", "swap", "inversion", "scramble"}, optional
        PyGAD mutation. Ignored by Powell. Default is ``"random"``.
    mutation_by_replacement : bool, optional
        Replace a mutated gene with a new value from its range. Ignored by
        Powell. Default is true.
    elitism : int, optional
        Best solutions copied unchanged into the next generation. Ignored by
        Powell and the particle swarm. Default is 1.
    inertia : float, optional
        Particle-swarm velocity weight. Ignored by Powell and PyGAD.
        Default is 0.7.
    cognitive : float, optional
        Pull toward each particle's own best position. Ignored by Powell and
        PyGAD. Default is 1.5.
    social : float, optional
        Pull toward the swarm's best position. Ignored by Powell and PyGAD.
        Default is 1.5.
    maxfev : int, optional
        Function evaluations allowed after Powell's starting points. Ignored
        by PyGAD and the particle swarm. Default is 40.
    xtol : float, optional
        Powell step tolerance. Ignored by PyGAD and the particle swarm.
        Default is ``1e-3``.
    ftol : float, optional
        Powell cost tolerance. Ignored by PyGAD and the particle swarm.
        Default is ``1e-3``.
    seed : int or None, optional
        PyGAD and particle-swarm seed. Ignored by Powell. Default is 1.

    Returns
    -------
    Solution
        Gains with the smallest simulated ISE, and the waveform they produced.

    Raises
    ------
    ValueError
        ``method`` is not ``"powell"``, ``"pygad"``, or ``"pso"``.
    """
    method = _method(method)
    if limits is None:
        limits = Limits()
    reference = steps(levels, hold, dt)
    spans = {"kp": _span(kp), "ki": _span(ki), "kd": _span(kd)}

    def score(genes: np.ndarray) -> tuple[Gains, float, np.ndarray, np.ndarray]:
        """Simulate one candidate and return its weighted ISE.

        Parameters
        ----------
        genes : numpy.ndarray
            Candidate ``[kp, ki, kd]``. Values outside the spans are clipped.

        Returns
        -------
        gains : pid.controller.Gains
            Clipped gains.
        ise : float
            ``we * sum((reference - y) ** 2) * dt``.
        response : numpy.ndarray
            Simulated plant output.
        command : numpy.ndarray
            Controller output.
        """
        gains = _clip(genes, spans)
        y, u = simulate(plant, reference, gains, dt=dt, limits=limits)
        ise = float(np.sum(np.square(reference - y)) * dt * we)
        return gains, ise, y, u

    if all(hi - lo <= 1e-12 for lo, hi in spans.values()):
        gains, ise, response, output = score(
            np.array([spans["kp"][0], spans["ki"][0], spans["kd"][0]])
        )
        return Solution(gains, ise, reference, response, output)

    if method == "powell":
        return _powell(score, spans, reference, maxfev, xtol, ftol)
    if method == "pso":
        return _pso(
            score,
            spans,
            reference,
            generations,
            population,
            inertia,
            cognitive,
            social,
            seed,
        )
    return _pygad(
        score,
        spans,
        reference,
        generations,
        population,
        parents,
        selection,
        crossover,
        mutation,
        mutation_by_replacement,
        elitism,
        seed,
    )


def _method(value) -> str:
    """Normalize a search-method name.

    Parameters
    ----------
    value : str
        Method name.

    Returns
    -------
    str
        ``"powell"``, ``"pygad"``, or ``"pso"``.

    Raises
    ------
    ValueError
        ``value`` is not one of those names.
    """
    name = str(value).strip().lower()
    if name not in ("powell", "pygad", "pso"):
        raise ValueError(f"tune method must be 'powell', 'pygad', or 'pso', got {value!r}")
    return name


def _powell(score, spans, reference, maxfev: int, xtol: float, ftol: float) -> Solution:
    """Powell from the low, middle, and high end of every free gain.

    The returned point is the visited candidate with the smallest ISE. Powell's
    own final point is one of those visits.

    Parameters
    ----------
    score : callable
        ``score(genes) -> (gains, ise, response, output)``.
    spans : dict of str to tuple of float
        ``kp``, ``ki``, and ``kd`` intervals.
    reference : numpy.ndarray
        Setpoint stored on the returned solution.
    maxfev : int
        Function evaluations allowed after the starting points.
    xtol : float
        Step tolerance. Must be finite and greater than zero.
    ftol : float
        Cost tolerance. Must be finite and greater than zero.

    Returns
    -------
    Solution
        Visited candidate with the smallest ISE.

    Raises
    ------
    ValueError
        ``maxfev`` is below 1, or a tolerance is not a finite number greater
        than zero.
    """
    if isinstance(maxfev, bool) or not isinstance(maxfev, int) or maxfev < 1:
        raise ValueError(f"maxfev must be an integer >= 1, got {maxfev!r}")
    for name, tolerance in (("xtol", xtol), ("ftol", ftol)):
        if isinstance(tolerance, bool) or not np.isfinite(tolerance) or float(tolerance) <= 0.0:
            raise ValueError(f"{name} must be a finite number > 0, got {tolerance!r}")
    names = ("kp", "ki", "kd")
    free = [name for name in names if spans[name][1] - spans[name][0] > 1e-12]
    fixed = {name: spans[name][0] for name in names if name not in free}
    bounds = [spans[name] for name in free]
    visited: list[tuple[Gains, float, np.ndarray, np.ndarray]] = []
    starts = _box_starts(bounds)
    bar = tqdm(total=len(starts) + maxfev, desc="tune", unit="eval", disable=None)
    best_ise = float("inf")

    def evaluate(point: np.ndarray) -> float:
        """ISE of one Powell point. The full result is kept in ``visited``.

        Parameters
        ----------
        point : numpy.ndarray
            Free-gain coordinates, in the order of ``free``.

        Returns
        -------
        float
            Weighted ISE.
        """
        nonlocal best_ise
        values = dict(fixed)
        values.update(zip(free, point))
        result = score(np.array([values[name] for name in names]))
        visited.append(result)
        best_ise = min(best_ise, result[1])
        bar.update(1)
        bar.set_postfix(ise=f"{best_ise:.4g}")
        return result[1]

    try:
        best_start = min(starts, key=evaluate)
        minimize(
            evaluate,
            best_start,
            method="Powell",
            bounds=bounds,
            options={"maxfev": maxfev, "xtol": float(xtol), "ftol": float(ftol)},
        )
    finally:
        bar.close()
    gains, ise, response, output = min(visited, key=lambda item: item[1])
    return Solution(gains, ise, reference, response, output)


def _pso(
    score,
    spans,
    reference,
    iterations: int,
    swarm: int,
    inertia: float,
    cognitive: float,
    social: float,
    seed: int | None,
) -> Solution:
    """Global-best particle swarm of the same box scored by ``score``.

    Each iteration moves every particle toward its own best position and the
    swarm's best position, then clips the result to the gain limits. A fixed
    gain keeps velocity zero.

    Parameters
    ----------
    score : callable
        ``score(genes) -> (gains, ise, response, output)``.
    spans : dict of str to tuple of float
        ``kp``, ``ki``, and ``kd`` intervals.
    reference : numpy.ndarray
        Setpoint stored on the returned solution.
    iterations : int
        Number of swarm updates. Each update scores every particle.
    swarm : int
        Number of particles.
    inertia, cognitive, social : float
        Velocity weight, pull toward a particle's own best, and pull toward
        the swarm best. Each must be finite and at least 0.
    seed : int or None
        Seed for the initial positions and the random pulls.

    Returns
    -------
    Solution
        Best particle visited, rescored by the last accepted simulation.

    Raises
    ------
    ValueError
        ``iterations`` or ``swarm`` is below 1, or a coefficient is not a
        finite number at least 0.
    """
    order = ("kp", "ki", "kd")
    low = np.array([spans[name][0] for name in order], dtype=float)
    high = np.array([spans[name][1] for name in order], dtype=float)
    width = high - low
    free = width > 1e-12
    if iterations < 1 or swarm < 1:
        raise ValueError("particle swarm needs a positive iteration count and swarm size")
    weights = {
        "inertia": inertia,
        "cognitive": cognitive,
        "social": social,
    }
    for name, weight in weights.items():
        if isinstance(weight, bool) or not np.isfinite(weight) or float(weight) < 0.0:
            raise ValueError(f"{name} must be a finite number >= 0, got {weight!r}")
    rng = np.random.default_rng(seed)
    position = rng.uniform(low, high, size=(swarm, 3))
    position[:, ~free] = low[~free]
    velocity = rng.uniform(-1.0, 1.0, size=(swarm, 3)) * width
    velocity[:, ~free] = 0.0
    personal = position.copy()
    personal_ise = np.full(swarm, np.inf)
    best_ise = np.inf
    best = None
    bar = tqdm(total=iterations, desc="tune", unit="iter", disable=None)
    try:
        for step in range(iterations):
            for index in range(swarm):
                gains, ise, response, output = score(position[index])
                if ise < personal_ise[index]:
                    personal_ise[index] = ise
                    personal[index] = position[index]
                if ise < best_ise:
                    best_ise = ise
                    best = (gains, ise, response, output)
            bar.update(1)
            bar.set_postfix(ise=f"{best_ise:.4g}")
            if step == iterations - 1:
                break
            global_pos = personal[int(np.argmin(personal_ise))]
            velocity = (
                float(inertia) * velocity
                + float(cognitive) * rng.random((swarm, 3)) * (personal - position)
                + float(social) * rng.random((swarm, 3)) * (global_pos - position)
            )
            np.clip(velocity, -width, width, out=velocity)
            velocity[:, ~free] = 0.0
            position = np.clip(position + velocity, low, high)
            position[:, ~free] = low[~free]
    finally:
        bar.close()
    gains, ise, response, output = best
    return Solution(gains, ise, reference, response, output)


def _pygad(
    score,
    spans,
    reference,
    generations: int,
    population: int,
    parents: int | None,
    selection: str,
    crossover: str,
    mutation: str,
    mutation_by_replacement: bool,
    elitism: int,
    seed: int | None,
) -> Solution:
    """PyGAD search of the same box scored by ``score``.

    Parameters
    ----------
    score : callable
        ``score(genes) -> (gains, ise, response, output)``. Fitness is the
        negative ISE.
    spans : dict of str to tuple of float
        ``kp``, ``ki``, and ``kd`` intervals. A fixed gain is a single gene value.
    reference : numpy.ndarray
        Setpoint stored on the returned solution.
    generations : int
        Number of generations.
    population : int
        Number of solutions per generation.
    parents : int or None
        Parents mated each generation. ``None`` uses one third of ``population``.
    selection, crossover, mutation : str
        PyGAD operator names.
    mutation_by_replacement : bool
        Replace a mutated gene from its range.
    elitism : int
        Best solutions copied unchanged into the next generation.
    seed : int or None
        Seed passed to PyGAD.

    Returns
    -------
    Solution
        Best generation member, rescored on ``reference``.

    Raises
    ------
    ValueError
        A PyGAD operator name is unknown, or ``parents`` or ``elitism`` does
        not fit in ``population``.
    """

    def fitness(_ga, genes, _index) -> float:
        """PyGAD fitness. The library maximises, so this is negative ISE.

        Parameters
        ----------
        _ga : pygad.GA
            Unused search object.
        genes : numpy.ndarray
            Candidate ``[kp, ki, kd]``.
        _index : int
            Unused solution index.

        Returns
        -------
        float
            Negative integral of squared error.
        """
        _gains, ise, _y, _u = score(genes)
        return -ise

    mating = _ga_count(population, parents, "parents", minimum=2)
    selection = _ga_name(selection, "selection", _SELECTION)
    crossover = _ga_name(crossover, "crossover", _CROSSOVER)
    mutation = _ga_name(mutation, "mutation", _MUTATION)
    if elitism < 0 or elitism >= population:
        raise ValueError(f"elitism must be from 0 to {population - 1}, got {elitism}")
    bar = tqdm(total=generations, desc="tune", unit="gen", disable=None)

    def on_generation(ga) -> None:
        """Advance one generation and show the best ISE so far."""
        ise = -float(np.max(ga.last_generation_fitness))
        bar.update(1)
        bar.set_postfix(ise=f"{ise:.4g}")

    try:
        ga = pygad.GA(
            num_generations=generations,
            num_parents_mating=mating,
            fitness_func=fitness,
            sol_per_pop=population,
            num_genes=3,
            gene_space=[_gene(*spans[name]) for name in ("kp", "ki", "kd")],
            gene_type=float,
            parent_selection_type=selection,
            crossover_type=crossover,
            keep_elitism=elitism,
            mutation_type=mutation,
            mutation_by_replacement=mutation_by_replacement,
            suppress_warnings=True,
            random_seed=seed,
            logger=log,
            on_generation=on_generation,
        )
        ga.run()
    finally:
        bar.close()
    best, _fitness, _ = ga.best_solution()
    gains, ise, response, output = score(best)
    return Solution(gains, ise, reference, response, output)


_SELECTION = ("sss", "rws", "sus", "rank", "random", "tournament")
_CROSSOVER = ("single_point", "two_points", "uniform", "scattered")
_MUTATION = ("random", "swap", "inversion", "scramble")


def _ga_name(value, name: str, allowed: tuple[str, ...]) -> str:
    """Normalize one PyGAD operator name.

    Parameters
    ----------
    value : str
        Operator name.
    name : str
        Parameter name, used in the error message.
    allowed : tuple of str
        Accepted names.

    Returns
    -------
    str
        Lower-case operator name.

    Raises
    ------
    ValueError
        ``value`` is not one of ``allowed``.
    """
    text = str(value).strip().lower()
    if text not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}, got {value!r}")
    return text


def _ga_count(population: int, value: int | None, name: str, *, minimum: int) -> int:
    """Parent count that leaves room for offspring.

    Parameters
    ----------
    population : int
        Number of solutions per generation.
    value : int or None
        Requested count. ``None`` uses one third of ``population``, at least 2,
        and clamps it below ``population``.
    name : str
        Parameter name, used in the error message.
    minimum : int
        Smallest explicit count.

    Returns
    -------
    int
        Count passed to PyGAD.

    Raises
    ------
    ValueError
        An explicit count is below ``minimum`` or not smaller than ``population``.
    """
    if value is None:
        return min(max(2, population // 3), population - 1)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be from {minimum} to {population - 1}, got {value}")
    if value < minimum or value >= population:
        raise ValueError(f"{name} must be from {minimum} to {population - 1}, got {value}")
    return value


def _span(values) -> tuple[float, float]:
    """Turn a gain entry into a closed interval.

    Parameters
    ----------
    values : sequence of float
        One number fixes the gain. Two or more numbers use the minimum and
        maximum.

    Returns
    -------
    tuple of float
        ``(low, high)``.
    """
    vals = tuple(float(item) for item in values)
    if len(vals) == 1:
        return vals[0], vals[0]
    return min(vals), max(vals)


def _box_starts(bounds: list[tuple[float, float]]) -> list[np.ndarray]:
    """Low end, middle, and high end of every free gain.

    Parameters
    ----------
    bounds : list of tuple of float
        ``(low, high)`` for each free gain.

    Returns
    -------
    list of numpy.ndarray
        Cartesian product of the three sample points on each axis.
    """
    axes = [(lo, 0.5 * (lo + hi), hi) for lo, hi in bounds]
    return [np.asarray(point, dtype=float) for point in product(*axes)]


def _gene(lo: float, hi: float):
    """PyGAD gene space for one gain.

    Parameters
    ----------
    lo : float
        Lower bound.
    hi : float
        Upper bound.

    Returns
    -------
    float or dict
        The fixed value when the interval is shorter than ``1e-12``, otherwise
        ``{"low": lo, "high": hi}``.
    """
    if hi - lo <= 1e-12:
        return float(lo)
    return {"low": float(lo), "high": float(hi)}


def _clip(genes: np.ndarray, spans: dict[str, tuple[float, float]]) -> Gains:
    """Clip a gene vector onto the gain intervals.

    Parameters
    ----------
    genes : numpy.ndarray
        Candidate ``[kp, ki, kd]``.
    spans : dict of str to tuple of float
        Inclusive ``(low, high)`` for each gain.

    Returns
    -------
    pid.controller.Gains
        Clipped gains.
    """
    values = {}
    for name, gene in zip(("kp", "ki", "kd"), genes):
        lo, hi = spans[name]
        values[name] = float(np.clip(gene, lo, hi))
    return Gains(values["kp"], values["ki"], values["kd"])


def _final_errors(reference: np.ndarray, response: np.ndarray) -> list[float]:
    """Absolute error at the end of each hold.

    Parameters
    ----------
    reference : numpy.ndarray
        Piecewise-constant setpoint.
    response : numpy.ndarray
        Simulated output, the same length as ``reference``.

    Returns
    -------
    list of float
        One ``|setpoint - y|`` per hold. Empty when ``reference`` is empty.
    """
    if len(reference) == 0:
        return []
    changes = np.flatnonzero(np.diff(reference)) + 1
    bounds = [0, *changes.tolist(), len(reference)]
    return [abs(float(reference[end - 1] - response[end - 1])) for end in bounds[1:]]
