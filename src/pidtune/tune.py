"""Tuning of the firmware PID.

``method`` selects the search. ``powell`` is a local bound-constrained search
started from the low, middle, and high end of each free gain. ``pygad`` is a
genetic search of the same box. Both score a candidate by the integral of
squared setpoint error from a simulation of ``PID.cpp``.
"""

import logging
from dataclasses import dataclass, replace
from itertools import product

import control
import numpy as np
import pygad
from scipy.optimize import minimize

from plantid.pid import DT, ENCODER_STEP, Gains, Limits, ki_that_holds, simulate, steps
from plantid.plant import load

log = logging.getLogger("pidtune")


def load_plant(path) -> control.NonlinearIOSystem:
    """Read a plant JSON file written by ``identify.py``."""
    return load(path)


@dataclass(frozen=True)
class Solution:
    """Gains chosen by the search, and the waveform they were scored on."""

    gains: Gains
    ise: float
    reference: np.ndarray
    response: np.ndarray
    output: np.ndarray

    def __str__(self) -> str:
        gains = self.gains
        worst = max(_final_errors(self.reference, self.response), default=0.0)
        return (
            f"Kp = {gains.kp:.4g}   Ki = {gains.ki:.4g}   Kd = {gains.kd:.4g}\n"
            f"ISE = {self.ise:.4g}   |e|_end {worst:.3f}"
        )


def tune(
    plant: control.NonlinearIOSystem,
    *,
    levels,
    hold: float,
    dt: float = DT,
    kp: tuple[float, float],
    ki: tuple[float, float],
    kd: tuple[float, float] = (0.0, 0.0),
    limits: Limits | None = None,
    dead_zone: float | None = None,
    encoder_step: float | None = ENCODER_STEP,
    we: float = 1.0,
    allow_steady_state_error: bool = False,
    method: str = "pygad",
    generations: int = 40,
    population: int = 24,
    seed: int | None = 1,
) -> Solution:
    """Search ``kp``, ``ki`` and ``kd``.

    ``levels`` is the setpoint waveform. Each value is held for ``hold`` seconds.
    ``method`` is ``powell`` or ``pygad``. Both minimize the same simulated ISE
    and stay inside the caller's gain limits.

    ``allow_steady_state_error`` left false raises the ``Ki`` floor, inside the
    caller's range, to the gain that can hold every level. True searches the
    whole supplied ``Ki`` range and keeps a shelf when that scores a lower ISE.
    """
    method = _method(method)
    limits = _with_dead_zone(limits, dead_zone)
    reference = steps(levels, hold, dt)
    spans = {"kp": _span(kp), "ki": _span(ki), "kd": _span(kd)}
    if not allow_steady_state_error:
        lo, hi = spans["ki"]
        floor = ki_that_holds(plant, levels, limits, dt)
        spans["ki"] = (min(hi, max(lo, floor)), hi)

    def score(genes: np.ndarray) -> tuple[Gains, float, np.ndarray, np.ndarray]:
        gains = _clip(genes, spans)
        y, u = simulate(
            plant, reference, gains, dt=dt, limits=limits, encoder_step=encoder_step
        )
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
    name = str(value).strip().lower()
    if name not in ("powell", "pygad"):
        raise ValueError(f"tune method must be 'powell' or 'pygad', got {value!r}")
    return name


def _powell(score, spans, reference) -> Solution:
    """Powell from the low, middle, and high end of every free gain.

    The returned point is the visited candidate with the smallest ISE. Powell's
    own final point is one of those visits.
    """
    names = ("kp", "ki", "kd")
    free = [name for name in names if spans[name][1] - spans[name][0] > 1e-12]
    fixed = {name: spans[name][0] for name in names if name not in free}
    bounds = [spans[name] for name in free]
    visited: list[tuple[Gains, float, np.ndarray, np.ndarray]] = []

    def evaluate(point: np.ndarray) -> float:
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


def _pygad(score, spans, reference, generations: int, population: int, seed: int | None) -> Solution:
    def fitness(_ga, genes, _index) -> float:
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


def _with_dead_zone(limits: Limits | None, dead_zone: float | None) -> Limits:
    base = limits if limits is not None else Limits()
    if dead_zone is None:
        return base
    return replace(base, dead_zone=float(dead_zone))


def _span(values) -> tuple[float, float]:
    vals = tuple(float(item) for item in values)
    if len(vals) == 1:
        return vals[0], vals[0]
    return min(vals), max(vals)


def _box_starts(bounds: list[tuple[float, float]]) -> list[np.ndarray]:
    """Low end, middle and high end of every free gain."""
    axes = [(lo, 0.5 * (lo + hi), hi) for lo, hi in bounds]
    return [np.asarray(point, dtype=float) for point in product(*axes)]


def _gene(lo: float, hi: float):
    if hi - lo <= 1e-12:
        return float(lo)
    return {"low": float(lo), "high": float(hi)}


def _clip(genes: np.ndarray, spans: dict[str, tuple[float, float]]) -> Gains:
    values = {}
    for name, gene in zip(("kp", "ki", "kd"), genes):
        lo, hi = spans[name]
        values[name] = float(np.clip(gene, lo, hi))
    return Gains(values["kp"], values["ki"], values["kd"])


def _final_errors(reference: np.ndarray, response: np.ndarray) -> list[float]:
    if len(reference) == 0:
        return []
    changes = np.flatnonzero(np.diff(reference)) + 1
    bounds = [0, *changes.tolist(), len(reference)]
    return [abs(float(reference[end - 1] - response[end - 1])) for end in bounds[1:]]
