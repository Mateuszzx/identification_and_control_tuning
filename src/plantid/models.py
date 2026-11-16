"""Catalog of plant models.

An inertial model of order ``n`` is

    G(s) = K / ((τ1 s + 1) ... (τn s + 1))

An integrator with ``n`` lags is

    G(s) = Kv / (s (τ1 s + 1) ... (τn s + 1))

``friction`` puts a Coulomb dead-zone on the plant input.
"""

from dataclasses import dataclass
from typing import Literal

Kind = Literal["inertial", "integrator"]


@dataclass(frozen=True)
class Model:
    """One candidate plant. Pass a list of these to :func:`plantid.search`."""

    name: str
    kind: Kind
    order: int
    friction: bool

    def __post_init__(self) -> None:
        if self.order < 0:
            raise ValueError("order must be >= 0")
        if self.kind == "inertial" and self.order < 1:
            raise ValueError("an inertial model needs order >= 1")


def inertial(order: int, *, friction: bool = False, name: str | None = None) -> Model:
    """Inertial plant ``K / Π(τ s + 1)`` of the given order."""
    label = name or f"inercyjny {order}" + (" + tarcie" if friction else "")
    return Model(label, "inertial", order, friction)


def integrator(lags: int = 0, *, friction: bool = True, name: str | None = None) -> Model:
    """Integrator ``Kv / s`` with ``lags`` extra real poles, optionally with friction."""
    if name is not None:
        label = name
    elif lags == 0:
        label = "całkujący" + (" + tarcie" if friction else "")
    else:
        label = f"całkujący, inercja {lags}" + (" + tarcie" if friction else "")
    return Model(label, "integrator", lags, friction)


def default_models() -> list[Model]:
    """Inertial orders 1–3, with and without friction, plus integrators with 0–2 lags."""
    models = [
        inertial(order, friction=friction) for order in (1, 2, 3) for friction in (False, True)
    ]
    models.extend(integrator(lags, friction=True) for lags in (0, 1, 2))
    return models
