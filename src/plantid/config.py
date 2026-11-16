"""Read ``config/config.yaml`` for identification and PID tuning."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class IdentifySetup:
    """Log to fit, and where the fitted plant is written."""

    log: Path
    out: Path


@dataclass(frozen=True)
class TuneSetup:
    """Waveform, firmware limits, and the search bounds."""

    plant: Path
    dt: float
    dead_zone: float
    hold: float
    levels: tuple[float, ...]
    kp: tuple[float, float]
    ki: tuple[float, float]
    kd: tuple[float, float]
    we: float
    allow_steady_state_error: bool
    method: str
    generations: int
    population: int


@dataclass(frozen=True)
class Config:
    identify: IdentifySetup
    tune: TuneSetup


def load_config(path, *, root=None) -> Config:
    """Load a YAML config.

    Relative paths are resolved from ``root``. When ``root`` is omitted they
    are resolved from the directory that contains the config file.
    """
    path = Path(path)
    base = Path(root) if root is not None else path.parent
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    return Config(
        identify=_identify(_section(data, "identify", path), base),
        tune=_tune(_section(data, "tune", path), base),
    )


def _section(data: dict, name: str, path: Path) -> dict:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is missing the {name} section")
    return value


def _identify(section: dict, root: Path) -> IdentifySetup:
    return IdentifySetup(log=_file(section, "log", root), out=_file(section, "out", root))


def _tune(section: dict, root: Path) -> TuneSetup:
    return TuneSetup(
        plant=_file(section, "plant", root),
        dt=float(section["dt"]),
        dead_zone=float(section["dead_zone"]),
        hold=float(section["hold"]),
        levels=tuple(float(level) for level in section["levels"]),
        kp=_pair(section["kp"], "kp"),
        ki=_pair(section["ki"], "ki"),
        kd=_pair(section["kd"], "kd"),
        we=float(section["we"]),
        allow_steady_state_error=bool(section["allow_steady_state_error"]),
        method=_method(section.get("method", "pygad")),
        generations=int(section["generations"]),
        population=int(section["population"]),
    )


def _file(section: dict, key: str, root: Path) -> Path:
    candidate = Path(str(section[key]))
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def _method(value) -> str:
    name = str(value).strip().lower()
    if name not in ("powell", "pygad"):
        raise ValueError(f"tune method must be 'powell' or 'pygad', got {value!r}")
    return name


def _pair(value, name: str) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        number = float(value)
        return number, number
    if isinstance(value, (list, tuple)) and len(value) in (1, 2):
        vals = tuple(float(item) for item in value)
        if len(vals) == 1:
            return vals[0], vals[0]
        return vals
    raise ValueError(f"{name} must be a number or one or two numbers")
