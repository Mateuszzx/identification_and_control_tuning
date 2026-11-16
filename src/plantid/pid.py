"""Digital PID from PID.cpp, and a step-size benchmark for its gains.

The controller is one iteration of ``PID::operator()``:

1. ``error = setpoint - measurement``, cleared when ``|error| < dead_zone``
2. ``acc = clamp(error + (1 - smoothing) * acc, ±anti_windup)``
3. ``p = Kp * error``, ``i = acc * dt * Ki``, ``d = Kd * (error - last_error) / dt``
4. ``out`` is lifted to ``±min_output`` and clipped to ``±max_output``

The default benchmark is the logged turn profile: rest at 0, then
−30, 0, −60, 0, −90, 0, +30, 0, +60, 0, +90, 0.
"""

from dataclasses import dataclass, field, replace
from itertools import product

import numpy as np
import polars as pl
from scipy.optimize import minimize

import control

from plantid.plant import initial_state, step

DT = 15e-3
ENCODER_STEP = 360.0 / 2048.0
LOG_STEPS = (0.0, -30.0, 0.0, -60.0, 0.0, -90.0, 0.0, 30.0, 0.0, 60.0, 0.0, 90.0, 0.0)


@dataclass(frozen=True)
class Gains:
    """Proportional, integral and derivative gains."""

    kp: float
    ki: float = 0.0
    kd: float = 0.0


@dataclass(frozen=True)
class Limits:
    """``PID_Config`` fields other than the three gains."""

    max_output: float = 100.0
    min_output: float = 0.0
    anti_windup: float = 100.0
    dead_zone: float = 0.0
    smoothing: float = 0.0


@dataclass
class FirmwarePID:
    """Stateful copy of ``kalman_electronics::regulators::PID``."""

    gains: Gains
    limits: Limits = field(default_factory=Limits)
    acc: float = 0.0
    last_error: float = 0.0

    def reset(self) -> None:
        self.acc = 0.0
        self.last_error = 0.0

    def step(self, setpoint: float, measurement: float, dt: float) -> float:
        error = float(setpoint - measurement)
        if abs(error) < self.limits.dead_zone:
            error = 0.0
        acc = error + (1.0 - self.limits.smoothing) * self.acc
        acc = float(np.clip(acc, -self.limits.anti_windup, self.limits.anti_windup))
        d_term = 0.0 if dt <= 0.0 else self.gains.kd * (error - self.last_error) / dt
        out = self.gains.kp * error + acc * dt * self.gains.ki + d_term
        if abs(out) < self.limits.min_output:
            out = float(np.sign(out) * self.limits.min_output)
        if abs(out) > self.limits.max_output:
            out = float(np.sign(out) * self.limits.max_output)
        self.acc = acc
        self.last_error = error
        return out


@dataclass(frozen=True)
class StepScore:
    """Error on one hold of the benchmark."""

    setpoint: float
    itae: float
    overshoot_deg: float
    abs_final_deg: float


@dataclass(frozen=True)
class Score:
    """Aggregate of :class:`StepScore` over the whole reference."""

    itae: float
    overshoot_deg: float
    mean_abs_final_deg: float
    saturated: float
    steps: tuple[StepScore, ...]


class Trajectory(tuple):
    """``(y, u)`` from :func:`simulate`. Unpacks like a pair."""

    def __new__(cls, y: np.ndarray, u: np.ndarray):
        return super().__new__(cls, (y, u))

    @property
    def y(self) -> np.ndarray:
        return self[0]

    @property
    def u(self) -> np.ndarray:
        return self[1]


@dataclass
class Tuning:
    """Gains chosen on the multi-step benchmark, plus the search table."""

    gains: Gains
    score: Score
    grid: pl.DataFrame
    steps: pl.DataFrame
    dt: float
    limits: Limits
    fc: float
    we: float
    cost: float
    meets_spec: bool
    reference: np.ndarray
    response: np.ndarray
    output: np.ndarray

    @property
    def i_max(self) -> float:
        """Largest integral contribution, ``anti_windup * dt * Ki``."""
        return self.limits.anti_windup * self.dt * self.gains.ki

    def __str__(self) -> str:
        g = self.gains
        mark = "" if self.meets_spec else "  (poza progiem overshoot / |e|)"
        worst = max(step.abs_final_deg for step in self.score.steps)
        return (
            f"Kp = {g.kp:.4g}   Ki = {g.ki:.4g}   Kd = {g.kd:.4g}{mark}\n"
            f"ISE = {self.cost:.4g}   We = {self.we:g}\n"
            f"i_max = AntiWindup * dt * Ki = {self.i_max:.2f}   Fc = {self.fc:.2f}\n"
            f"ITAE {self.score.itae:.0f}   overshoot {self.score.overshoot_deg:.2f}"
            f"   |e|_end {worst:.3f}"
        )


def steps(levels, hold: float, dt: float) -> np.ndarray:
    """Piecewise-constant reference. Each level is held for ``hold`` seconds."""
    count = int(round(hold / dt))
    if count < 1:
        raise ValueError("hold must cover at least one sample")
    return np.repeat(np.asarray(list(levels), dtype=float), count)


def simulate(
    plant: control.NonlinearIOSystem,
    reference: np.ndarray,
    gains: Gains,
    *,
    dt: float,
    limits: Limits | None = None,
    dead_zone: float | None = None,
    encoder_step: float | None = ENCODER_STEP,
    y0: float = 0.0,
) -> Trajectory:
    """Run ``plant`` under :class:`FirmwarePID`. Measurement is encoder-quantised.

    ``dead_zone`` is the firmware ``DeadZone`` in degrees. It overrides
    ``limits.dead_zone`` when given.
    """
    limits = _with_dead_zone(limits, dead_zone)
    pid = FirmwarePID(gains, limits)
    state = initial_state(plant, y0)
    y_now = float(y0)
    y = np.empty(len(reference))
    u = np.empty(len(reference))
    for i, setpoint in enumerate(reference):
        measured = _quantise(y_now, encoder_step)
        command = pid.step(float(setpoint), measured, dt)
        state, y_now = step(plant, state, command, dt)
        y[i] = y_now
        u[i] = command
    return Trajectory(y, u)


def tune(
    plant: control.NonlinearIOSystem,
    *,
    dt: float = DT,
    levels=LOG_STEPS,
    hold: float = 6.0,
    kp: tuple[float, float],
    ki: tuple[float, float],
    kd: tuple[float, float] = (0.0, 0.0),
    limits: Limits | None = None,
    dead_zone: float | None = None,
    encoder_step: float | None = ENCODER_STEP,
    we: float = 1.0,
    max_overshoot: float = 1.0,
    max_final_error: float = 0.2,
    allow_steady_state_error: bool = False,
) -> Tuning:
    """Minimize the integral of squared setpoint error.

    ``plant`` is a python-control :class:`~control.NonlinearIOSystem` from
    :func:`plantid.io_system`. This function does not identify it.

    ``kp``, ``ki`` and ``kd`` are the search limits ``(low, high)``. A single
    number, or equal ends, fixes that gain. ``kd`` defaults to ``(0, 0)``, so
    the derivative stays off until a range is passed.

    The cost is ISE, ``We * sum((setpoint - y)**2) * dt``. With
    ``allow_steady_state_error`` left false, ``Ki`` is not left below the value
    that makes ``i_max = anti_windup * dt * Ki`` cover the holding command,
    ``Fc + |setpoint| / K`` for an inertial plant and ``Fc`` for an integrator.
    The search never goes above the caller's upper limit. If that limit is too
    low to cover the holding command, ``Ki`` stays at the limit and a shelf
    remains. Set ``allow_steady_state_error`` to search the whole ``Ki`` range
    and accept that shelf. ``max_final_error`` then only reports whether the
    hold ended inside the band. The search evaluates the low end, the middle
    and the high end of every free gain, then polishes the best of those points.

    ``max_overshoot`` stays a pass/fail check reported as ``meets_spec``.

    ``dead_zone`` is the firmware ``DeadZone`` and overrides ``limits.dead_zone``.
    """
    limits = _with_dead_zone(limits, dead_zone)
    reference = steps(levels, hold, dt)
    count = len(reference) // len(tuple(levels))
    ceiling = _ise(reference, np.full(len(reference), _runaway(reference)), dt, we) + 1.0
    ki_lo, ki_hi = _span(ki)
    if not allow_steady_state_error:
        ki_hold = _ki_that_holds(plant, levels, limits, dt)
        ki_lo = min(ki_hi, max(ki_lo, ki_hold))
    spans = {
        "kp": _span(kp),
        "ki": (ki_lo, ki_hi),
        "kd": _span(kd),
    }
    rows: list[dict[str, float]] = []

    def evaluate(gains: Gains) -> float:
        y, u = simulate(plant, reference, gains, dt=dt, limits=limits, encoder_step=encoder_step)
        scored = _score(reference, y, u, dt=dt, count=count, max_output=limits.max_output)
        ise = _ise(reference, y, dt, we)
        excess = _end_excess(
            reference,
            y,
            count,
            max_final_error,
            plant,
            gains,
            limits,
            dt,
            count_resting=not allow_steady_state_error,
        )
        rows.append({**_row(gains, scored), "ise": ise, "excess": excess})
        return ise if excess <= 0.0 else ceiling + excess

    free = [(name, lo, hi) for name, (lo, hi) in spans.items() if hi - lo > 1e-12]
    fixed = {name: lo for name, (lo, hi) in spans.items() if hi - lo <= 1e-12}

    def decode(x: np.ndarray) -> Gains:
        values = dict(fixed)
        for (name, _lo, _hi), item in zip(free, x):
            values[name] = float(item)
        return Gains(values["kp"], values["ki"], values["kd"])

    if free:
        bounds = [(lo, hi) for _name, lo, hi in free]
        starts = _box_starts(bounds)
        best_x = min(starts, key=lambda point: evaluate(decode(point)))
        minimize(
            lambda x: evaluate(decode(x)),
            best_x,
            method="Powell",
            bounds=bounds,
            options={"maxfev": 40, "xtol": 1e-3, "ftol": 1e-3},
        )
    else:
        evaluate(decode(np.array([])))

    chosen = min(rows, key=lambda row: (row["excess"] > 0.0, row["excess"], row["ise"]))
    grid = pl.DataFrame(rows).drop(["ise", "excess"]).sort(["itae", "overshoot_deg", "kp"])
    feasible = chosen["overshoot_deg"] < max_overshoot and chosen["excess"] <= 0.0
    gains = Gains(chosen["kp"], chosen["ki"], chosen["kd"])
    response, output = simulate(
        plant, reference, gains, dt=dt, limits=limits, encoder_step=encoder_step
    )
    score = _score(reference, response, output, dt=dt, count=count, max_output=limits.max_output)
    baseline = Gains(1.0, 0.0, 0.0)
    y_base, u_base = simulate(
        plant, reference, baseline, dt=dt, limits=limits, encoder_step=encoder_step
    )
    base_score = _score(reference, y_base, u_base, dt=dt, count=count, max_output=limits.max_output)
    return Tuning(
        gains=gains,
        score=score,
        grid=grid,
        steps=pl.concat([_step_frame(baseline, base_score), _step_frame(gains, score)]),
        dt=dt,
        limits=limits,
        fc=float(plant.params["fc"]),
        we=float(we),
        cost=float(chosen["ise"]),
        meets_spec=feasible,
        reference=reference,
        response=response,
        output=output,
    )


def _with_dead_zone(limits: Limits | None, dead_zone: float | None) -> Limits:
    base = limits if limits is not None else Limits()
    if dead_zone is None:
        return base
    return replace(base, dead_zone=float(dead_zone))


def _quantise(y: float, encoder_step: float | None) -> float:
    if encoder_step is None or encoder_step <= 0.0:
        return float(y)
    return float(np.round(y / encoder_step) * encoder_step)


def _span(values) -> tuple[float, float]:
    """``(low, high)`` is the search interval. One number fixes the gain."""
    vals = tuple(float(item) for item in values)
    if len(vals) == 1:
        return vals[0], vals[0]
    return min(vals), max(vals)


def _row(gains: Gains, scored: Score) -> dict[str, float]:
    return {
        "kp": gains.kp,
        "ki": gains.ki,
        "kd": gains.kd,
        "itae": scored.itae,
        "overshoot_deg": scored.overshoot_deg,
        "mean_abs_final_deg": scored.mean_abs_final_deg,
        "saturated": scored.saturated,
    }


def _box_starts(bounds: list[tuple[float, float]]) -> list[np.ndarray]:
    """Low end, middle and high end of every free gain."""
    axes = []
    for lo, hi in bounds:
        axes.append((lo, 0.5 * (lo + hi), hi))
    return [np.asarray(point, dtype=float) for point in product(*axes)]


def _ise(reference, y, dt: float, we: float) -> float:
    error = np.asarray(reference, dtype=float) - np.asarray(y, dtype=float)
    return float(np.sum(error * error) * dt * we)


def _runaway(reference: np.ndarray) -> float:
    """A position far enough that a clipped actuator cannot reach it in one hold."""
    return float(np.max(np.abs(reference))) * 100.0 + 1.0


def _end_excess(
    reference,
    y,
    count: int,
    limit: float,
    plant: control.NonlinearIOSystem,
    gains: Gains,
    limits: Limits,
    dt: float,
    *,
    count_resting: bool,
) -> float:
    """Squared miss beyond ``limit``. Zero means the hold ends on the setpoint and can stay there.

    ``count_resting`` is false when a steady-state shelf is allowed. The end of
    the hold is then not required to sit inside ``limit`` either, so a small
    shelf does not outrank the squared error of the step.
    """
    if not count_resting:
        return 0.0
    error = np.asarray(reference, dtype=float) - np.asarray(y, dtype=float)
    total = 0.0
    nseg = len(reference) // count
    for i in range(nseg):
        setpoint = float(reference[i * count])
        resting = _resting_error(plant, setpoint, gains, limits, dt)
        if resting > 1e-6:
            total += resting * resting
        reached = abs(float(error[(i + 1) * count - 1])) - limit
        if reached > 0.0:
            total += reached * reached
    return total


def _holding_command(plant: control.NonlinearIOSystem, setpoint: float) -> float:
    """Actuator command that keeps ``setpoint`` once the plant has stopped.

    An inertial plant needs ``Fc + |y| / K``. An integrator only has to cover
    Coulomb friction.
    """
    fc = float(plant.params["fc"])
    if plant.params["kind"] == "integrator" or float(plant.params["gain"]) <= 0.0:
        return fc
    return fc + abs(float(setpoint)) / float(plant.params["gain"])


def ki_that_holds(plant: control.NonlinearIOSystem, levels, limits: Limits, dt: float) -> float:
    """Smallest ``Ki`` for which the integrator alone can hold every level."""
    return _ki_that_holds(plant, levels, limits, dt)


def _ki_that_holds(plant: control.NonlinearIOSystem, levels, limits: Limits, dt: float) -> float:
    """Smallest ``Ki`` for which the integrator alone can hold every level."""
    needed = max(_holding_command(plant, float(level)) for level in levels)
    if needed <= 0.0:
        return 0.0
    rate = limits.anti_windup * dt
    if rate <= 0.0:
        raise ValueError("anti_windup and dt must be positive to remove steady-state error")
    return needed / rate


def _resting_error(plant: control.NonlinearIOSystem, setpoint: float, gains: Gains, limits: Limits, dt: float) -> float:
    """Position error left when the integrator is at its limit and the plant has stopped.

    Whatever the integral cannot supply is left to ``Kp``, which takes a
    proportional offset. Enough integral leaves no offset at any ``Kp``.
    """
    i_max = limits.anti_windup * dt * max(gains.ki, 0.0)
    short = _holding_command(plant, setpoint) - i_max
    if short <= 1e-9:
        return 0.0
    if gains.kp <= 1e-12:
        return abs(float(setpoint))
    return short / gains.kp


def _score(reference, y, u, *, dt: float, count: int, max_output: float) -> Score:
    error = reference - y
    itae = 0.0
    overshoot = 0.0
    finals: list[float] = []
    pieces: list[StepScore] = []
    nseg = len(reference) // count
    for i in range(nseg):
        sl = slice(i * count, (i + 1) * count)
        elapsed = np.arange(count) * dt
        err = error[sl]
        part = float(np.sum(elapsed * np.abs(err)) * dt)
        target = float(reference[i * count])
        yseg = y[sl]
        y0 = float(yseg[0])
        peak = _overshoot(yseg, y0, target)
        final = abs(float(err[-1]))
        itae += part
        overshoot += peak
        finals.append(final)
        pieces.append(StepScore(target, part, peak, final))
    return Score(
        itae=itae,
        overshoot_deg=overshoot,
        mean_abs_final_deg=float(np.mean(finals)),
        saturated=float(np.mean(np.abs(u) >= max_output - 1e-6)),
        steps=tuple(pieces),
    )


def _overshoot(yseg: np.ndarray, y0: float, target: float) -> float:
    if target > y0:
        return max(0.0, float(np.max(yseg) - target))
    if target < y0:
        return max(0.0, float(target - np.min(yseg)))
    return 0.0


def _step_frame(gains: Gains, score: Score) -> pl.DataFrame:
    label = f"Kp={gains.kp:g}, Ki={gains.ki:g}, Kd={gains.kd:g}"
    return pl.DataFrame(
        {
            "regulator": [label] * len(score.steps),
            "setpoint": [item.setpoint for item in score.steps],
            "itae": [item.itae for item in score.steps],
            "overshoot_deg": [item.overshoot_deg for item in score.steps],
            "abs_final_deg": [item.abs_final_deg for item in score.steps],
        }
    )
