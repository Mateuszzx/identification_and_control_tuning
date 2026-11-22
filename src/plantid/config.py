"""Read ``config/config.yaml`` for identification and PID tuning."""

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path("config") / "config.yaml"


@dataclass(frozen=True)
class IdentifySetup:
    """Log to fit, and where the fitted plant is written.

    Attributes
    ----------
    log : pathlib.Path
        Closed-loop CSV log.
    out : pathlib.Path
        JSON file that receives the fitted plant.
    kp : float
        Proportional gain of the loop that collected the log. Not fitted.
    models : pathlib.Path
        YAML list of plant structures to fit.
    """

    log: Path
    out: Path
    kp: float
    models: Path


@dataclass(frozen=True)
class TuneSetup:
    """Waveform and the search bounds.

    Attributes
    ----------
    plant : pathlib.Path
        Plant JSON written by identification.
    dt : float
        Sample time in seconds.
    hold : float
        Seconds each setpoint level is held.
    levels : tuple of float
        Setpoint values, in order.
    kp, ki, kd : tuple of float
        ``(low, high)`` search interval. Equal ends fix the gain.
    we : float
        Weight on the integral of squared setpoint error.
    method : str
        ``"powell"`` or ``"pygad"``.
    generations : int
        Generations used by PyGAD.
    population : int
        Population size used by PyGAD.
    """

    plant: Path
    dt: float
    hold: float
    levels: tuple[float, ...]
    kp: tuple[float, float]
    ki: tuple[float, float]
    kd: tuple[float, float]
    we: float
    method: str
    generations: int
    population: int


@dataclass(frozen=True)
class Config:
    """Identification setup and tuning setup from one YAML file.

    Attributes
    ----------
    identify : IdentifySetup
        Log path, plant output path, recorded proportional gain, and models file.
    tune : TuneSetup
        Waveform, limits, and search settings.
    """

    identify: IdentifySetup
    tune: TuneSetup


def repository_root(start: Path) -> Path:
    """Directory containing ``pyproject.toml``, walking up from ``start``.

    Parameters
    ----------
    start : pathlib.Path
        File or directory to start from.

    Returns
    -------
    pathlib.Path
        Nearest ancestor, including ``start`` itself, that contains
        ``pyproject.toml``.

    Raises
    ------
    FileNotFoundError
        No ancestor contains ``pyproject.toml``.
    """
    start = Path(start).resolve()
    chain = start.parents if start.is_file() else (start, *start.parents)
    for candidate in chain:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"no pyproject.toml above {start}")


def resolve_config(path: str | Path | None, root: Path) -> Path:
    """Absolute config path. A relative path starts at ``root``.

    Parameters
    ----------
    path : str or pathlib.Path or None
        Config path. ``None`` selects :data:`DEFAULT_CONFIG`.
    root : pathlib.Path
        Repository root used for a relative ``path``.

    Returns
    -------
    pathlib.Path
        Absolute config path. The file is not required to exist yet.
    """
    candidate = DEFAULT_CONFIG if path is None else Path(path)
    if not candidate.is_absolute():
        candidate = Path(root) / candidate
    return candidate.resolve()


def load_config(path, *, root=None) -> Config:
    """Load a YAML config.

    Relative paths inside the file are resolved from ``root``. When ``root``
    is omitted they are resolved from the directory that contains the file.

    Parameters
    ----------
    path : str or pathlib.Path
        YAML file to read.
    root : pathlib.Path, optional
        Base for relative paths stored in the file.

    Returns
    -------
    Config
        Parsed identification and tuning setup.

    Raises
    ------
    ValueError
        The file is not a mapping, a required section is missing, a gain
        bound is malformed, or ``method`` is not ``powell`` or ``pygad``.
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
    """Return one mapping section of the config file.

    Parameters
    ----------
    data : dict
        Parsed YAML document.
    name : str
        Section name, ``"identify"`` or ``"tune"``.
    path : pathlib.Path
        Config path, used in the error message.

    Returns
    -------
    dict
        The requested section.

    Raises
    ------
    ValueError
        The section is missing or is not a mapping.
    """
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is missing the {name} section")
    return value


def _identify(section: dict, root: Path) -> IdentifySetup:
    """Build the identification setup.

    Parameters
    ----------
    section : dict
        ``identify`` mapping from the YAML file.
    root : pathlib.Path
        Base for relative paths.

    Returns
    -------
    IdentifySetup
        Log path, plant output path, recorded proportional gain, and models file.
    """
    return IdentifySetup(
        log=_file(section, "log", root),
        out=_file(section, "out", root),
        kp=float(section["kp"]),
        models=_file(section, "models", root),
    )


def _tune(section: dict, root: Path) -> TuneSetup:
    """Build the tuning setup.

    Parameters
    ----------
    section : dict
        ``tune`` mapping from the YAML file.
    root : pathlib.Path
        Base for the plant path.

    Returns
    -------
    TuneSetup
        Waveform, limits, and search settings. A missing ``method`` becomes
        ``"pygad"``.
    """
    return TuneSetup(
        plant=_file(section, "plant", root),
        dt=float(section["dt"]),
        hold=float(section["hold"]),
        levels=tuple(float(level) for level in section["levels"]),
        kp=_pair(section["kp"], "kp"),
        ki=_pair(section["ki"], "ki"),
        kd=_pair(section["kd"], "kd"),
        we=float(section["we"]),
        method=_method(section.get("method", "pygad")),
        generations=int(section["generations"]),
        population=int(section["population"]),
    )


def _file(section: dict, key: str, root: Path) -> Path:
    """Resolve one path entry.

    Parameters
    ----------
    section : dict
        Mapping that holds the path.
    key : str
        Entry name.
    root : pathlib.Path
        Base for a relative path.

    Returns
    -------
    pathlib.Path
        Absolute path.
    """
    candidate = Path(str(section[key]))
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def _method(value) -> str:
    """Normalize a search-method name.

    Parameters
    ----------
    value : str
        Method name from the config.

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


def _pair(value, name: str) -> tuple[float, float]:
    """Turn a gain entry into a closed interval.

    Parameters
    ----------
    value : float or sequence of float
        One number fixes the gain. Two numbers are the low and high ends.
    name : str
        Gain name, used in the error message.

    Returns
    -------
    tuple of float
        ``(low, high)``. A single number is repeated.

    Raises
    ------
    ValueError
        ``value`` is not a number or a sequence of one or two numbers.
    """
    if isinstance(value, (int, float)):
        number = float(value)
        return number, number
    if isinstance(value, (list, tuple)) and len(value) in (1, 2):
        vals = tuple(float(item) for item in value)
        if len(vals) == 1:
            return vals[0], vals[0]
        return vals
    raise ValueError(f"{name} must be a number or one or two numbers")
