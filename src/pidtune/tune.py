"""Tuning of the PID defined in :mod:`pid`.

``method`` selects the search. ``powell`` is a local bound-constrained search
started from the low, middle, and high end of each free gain. ``pygad`` is a
genetic search of the same box. Both score a candidate by the integral of
squared setpoint error from a simulation of :meth:`pid.controller.PID.step`.
"""

import logging
from dataclasses import dataclass
from itertools import product

import numpy as np
import pygad
from scipy.optimize import minimize

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
    seed: int | None = 1,
) -> Solution:
    """Search ``kp``, ``ki``, and ``kd``.

    Each value in ``levels`` is held for ``hold`` seconds. ``powell`` and
    ``pygad`` minimize the same simulated ISE and stay inside the caller's
    gain limits. Dead zone and accumulator live in :mod:`pid` and are not
    part of this search.

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
    method : {"powell", "pygad"}, optional
        Search algorithm. Default is ``"pygad"``.
    generations : int, optional
        PyGAD generations. Ignored by Powell. Default is 40.
    population : int, optional
        PyGAD population. Ignored by Powell. Default is 24.
    seed : int or None, optional
        PyGAD seed. Ignored by Powell. Default is 1.

    Returns
    -------
    Solution
        Gains with the smallest simulated ISE, and the waveform they produced.

    Raises
    ------
    ValueError
        ``method`` is not ``"powell"`` or ``"pygad"``.
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
        return _powell(score, spans, reference)
    return _pygad(score, spans, reference, generations, population, seed)


def _method(value) -> str:
    """Normalize a search-method name.

    Parameters
    ----------
    value : str
        Method name.

    Returns
    -------
    str
        ``"powell"`` or ``"pygad"``.

    Raises
    ------
    ValueError
        ``value`` is not one of those two names.
    """
    name = str(value).strip().lower()
    if name not in ("powell", "pygad"):
        raise ValueError(f"tune method must be 'powell' or 'pygad', got {value!r}")
    return name


def _powell(score, spans, reference) -> Solution:
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

    Returns
    -------
    Solution
        Visited candidate with the smallest ISE.
    """
    names = ("kp", "ki", "kd")
    free = [name for name in names if spans[name][1] - spans[name][0] > 1e-12]
    fixed = {name: spans[name][0] for name in names if name not in free}
    bounds = [spans[name] for name in free]
    visited: list[tuple[Gains, float, np.ndarray, np.ndarray]] = []

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
        values = dict(fixed)
        values.update(zip(free, point))
        result = score(np.array([values[name] for name in names]))
        visited.append(result)
        return result[1]

    starts = _box_starts(bounds)
    best_start = min(starts, key=evaluate)
    minimize(
        evaluate,
        best_start,
        method="Powell",
        bounds=bounds,
        options={"maxfev": 40, "xtol": 1e-3, "ftol": 1e-3},
    )
    gains, ise, response, output = min(visited, key=lambda item: item[1])
    return Solution(gains, ise, reference, response, output)


def _pygad(
    score,
    spans,
    reference,
    generations: int,
    population: int,
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
    seed : int or None
        Seed passed to PyGAD.

    Returns
    -------
    Solution
        Best generation member, rescored on ``reference``.
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

    parents = max(2, population // 3)
    ga = pygad.GA(
        num_generations=generations,
        num_parents_mating=min(parents, population - 1),
        fitness_func=fitness,
        sol_per_pop=population,
        num_genes=3,
        gene_space=[_gene(*spans[name]) for name in ("kp", "ki", "kd")],
        gene_type=float,
        parent_selection_type="sss",
        keep_elitism=1,
        mutation_type="random",
        mutation_by_replacement=True,
        suppress_warnings=True,
        random_seed=seed,
        logger=log,
    )
    ga.run()
    best, _fitness, _ = ga.best_solution()
    gains, ise, response, output = score(best)
    return Solution(gains, ise, reference, response, output)


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
