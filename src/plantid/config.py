"""Read ``config/config.yaml`` for identification and PID tuning."""

import math
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
    max_nfev : int
        Residual evaluations allowed while fitting each structure.
    """

    log: Path
    out: Path
    kp: float
    models: Path
    max_nfev: int


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
        ``"powell"``, ``"pygad"``, or ``"pso"``.
    generations : int
        PyGAD generations, or particle-swarm iterations.
    population : int
        PyGAD population, or particle-swarm size.
    parents : int
        Parents mated in each PyGAD generation.
    selection : str
        PyGAD parent selection. One of ``sss``, ``rws``, ``sus``, ``rank``,
        ``random``, and ``tournament``.
    crossover : str
        PyGAD crossover. One of ``single_point``, ``two_points``, ``uniform``,
        and ``scattered``.
    mutation : str
        PyGAD mutation. One of ``random``, ``swap``, ``inversion``, and ``scramble``.
    mutation_by_replacement : bool
        Replace a gene with a new value drawn from its range.
    elitism : int
        Best solutions copied unchanged into the next generation.
    seed : int or None
        PyGAD and particle-swarm seed. ``None`` leaves the generator unseeded.
    inertia : float
        Particle-swarm velocity weight.
    cognitive : float
        Pull toward each particle's own best position.
    social : float
        Pull toward the swarm's best position.
    maxfev : int
        Function evaluations allowed after Powell's starting points.
    xtol : float
        Powell step tolerance.
    ftol : float
        Powell cost tolerance.
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
    parents: int
    selection: str
    crossover: str
    mutation: str
    mutation_by_replacement: bool
    elitism: int
    seed: int | None
    inertia: float
    cognitive: float
    social: float
    maxfev: int
    xtol: float
    ftol: float


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
        bound is malformed, or ``method`` is not ``powell``, ``pygad``, or ``pso``.
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
        Log path, plant output path, recorded proportional gain, models file,
        and the residual-evaluation limit.
    """
    return IdentifySetup(
        log=_file(section, "log", root),
        out=_file(section, "out", root),
        kp=float(section["kp"]),
        models=_file(section, "models", root),
        max_nfev=_positive_int(section["max_nfev"], "max_nfev"),
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
        ``"pygad"``. Missing PyGAD and particle-swarm entries use the shipped defaults.
    """
    population = int(section["population"])
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
        population=population,
        parents=_parents(population, section.get("parents")),
        selection=_choice(section.get("selection", "sss"), "selection", _SELECTION),
        crossover=_choice(section.get("crossover", "single_point"), "crossover", _CROSSOVER),
        mutation=_choice(section.get("mutation", "random"), "mutation", _MUTATION),
        mutation_by_replacement=_flag(
            section.get("mutation_by_replacement", True),
            "mutation_by_replacement",
        ),
        elitism=_parents_bound(
            population,
            _positive_int(section.get("elitism", 1), "elitism", minimum=0),
            "elitism",
        ),
        seed=_seed(section.get("seed", 1)),
        inertia=_coefficient(section.get("inertia", 0.7), "inertia"),
        cognitive=_coefficient(section.get("cognitive", 1.5), "cognitive"),
        social=_coefficient(section.get("social", 1.5), "social"),
        maxfev=_positive_int(section.get("maxfev", 40), "maxfev"),
        xtol=_positive_float(section.get("xtol", 1e-3), "xtol"),
        ftol=_positive_float(section.get("ftol", 1e-3), "ftol"),
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


_SELECTION = ("sss", "rws", "sus", "rank", "random", "tournament")
_CROSSOVER = ("single_point", "two_points", "uniform", "scattered")
_MUTATION = ("random", "swap", "inversion", "scramble")


def _coefficient(value, name: str) -> float:
    """Return a finite particle-swarm coefficient at least 0.

    Parameters
    ----------
    value : float
        Coefficient from the config.
    name : str
        Entry name, used in the error message.

    Returns
    -------
    float
        ``value`` as a float.

    Raises
    ------
    ValueError
        ``value`` is not a finite number, or it is negative.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
    return number


def _positive_float(value, name: str) -> float:
    """Return a finite tolerance greater than zero.

    Parameters
    ----------
    value : float
        Tolerance from the config.
    name : str
        Entry name, used in the error message.

    Returns
    -------
    float
        ``value`` as a float.

    Raises
    ------
    ValueError
        ``value`` is not a finite number greater than zero.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
    return number


def _positive_int(value, name: str, *, minimum: int = 1) -> int:
    """Return an integer at or above ``minimum``.

    Parameters
    ----------
    value : int
        Candidate count.
    name : str
        Entry name, used in the error message.
    minimum : int, optional
        Smallest accepted value. Default is 1.

    Returns
    -------
    int
        ``value`` as an integer.

    Raises
    ------
    ValueError
        ``value`` is not an integer, or it is below ``minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _parents(population: int, value) -> int:
    """Parent count. A missing value is one third of the population, at least 2.

    Parameters
    ----------
    population : int
        PyGAD population.
    value : int or None
        Requested parent count. ``None`` uses the default.

    Returns
    -------
    int
        Parents mated in one generation. The default is clamped below ``population``.

    Raises
    ------
    ValueError
        An explicit count is below 2 or not smaller than ``population``.
    """
    if value is None:
        return min(max(2, population // 3), population - 1)
    return _parents_bound(population, _positive_int(value, "parents", minimum=2), "parents")


def _parents_bound(population: int, count: int, name: str) -> int:
    """Reject a count that leaves no room for the rest of the population.

    Parameters
    ----------
    population : int
        PyGAD population.
    count : int
        Candidate count.
    name : str
        Entry name, used in the error message.

    Returns
    -------
    int
        ``count`` when it is smaller than ``population``.

    Raises
    ------
    ValueError
        ``count`` is greater than or equal to ``population``.
    """
    if count >= population:
        raise ValueError(f"{name} must be smaller than population {population}, got {count}")
    return count


def _choice(value, name: str, allowed: tuple[str, ...]) -> str:
    """Normalize one named PyGAD option.

    Parameters
    ----------
    value : str
        Option from the config.
    name : str
        Entry name, used in the error message.
    allowed : tuple of str
        Accepted names.

    Returns
    -------
    str
        Lower-case option.

    Raises
    ------
    ValueError
        ``value`` is not one of ``allowed``.
    """
    text = str(value).strip().lower()
    if text not in allowed:
        names = ", ".join(allowed)
        raise ValueError(f"{name} must be one of {names}, got {value!r}")
    return text


def _flag(value, name: str) -> bool:
    """Return a boolean config flag.

    Parameters
    ----------
    value : bool
        Flag from the config.
    name : str
        Entry name, used in the error message.

    Returns
    -------
    bool
        ``value``.

    Raises
    ------
    ValueError
        ``value`` is not a boolean.
    """
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false, got {value!r}")
    return value


def _seed(value) -> int | None:
    """Return a PyGAD seed.

    Parameters
    ----------
    value : int or None
        Seed from the config. ``None`` leaves the generator unseeded.

    Returns
    -------
    int or None
        The seed.

    Raises
    ------
    ValueError
        ``value`` is not an integer or ``None``.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"seed must be an integer or null, got {value!r}")
    return value


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
