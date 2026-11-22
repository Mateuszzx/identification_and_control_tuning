"""Save the identification and tuning figures."""

from pathlib import Path

import matplotlib
import numpy as np


def save_identify(path, time, setpoint, measurement, prediction, label) -> None:
    """Write the measured output, the fitted response, and the residual.

    Parameters
    ----------
    path : str or pathlib.Path
        PNG destination. Parent directories are created.
    time : array_like
        Sample times.
    setpoint : array_like
        Logged reference.
    measurement : array_like
        Logged output.
    prediction : array_like
        Fitted output, the same length as ``measurement``.
    label : str
        Legend entry for ``prediction``.
    """
    plt = _pyplot()
    time = np.asarray(time, dtype=float)
    measurement = np.asarray(measurement, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    fig, axes = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": (2, 1)}
    )
    axes[0].plot(time, setpoint, label="set_point", lw=1.2)
    axes[0].plot(time, measurement, label="measure_value", lw=1.2)
    axes[0].plot(time, prediction, ls="--", label=label, lw=1.0)
    axes[0].set_ylabel("y")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].plot(time, measurement - prediction, lw=0.8)
    axes[1].axhline(0.0, color="0.5", lw=0.6)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("residuum")
    _write(fig, path)


def save_tune(path, time, reference, response, output) -> None:
    """Write the setpoint, the simulated response, and the controller output.

    Parameters
    ----------
    path : str or pathlib.Path
        PNG destination. Parent directories are created.
    time : array_like
        Sample times.
    reference : array_like
        Setpoint waveform.
    response : array_like
        Simulated plant output.
    output : array_like
        Controller command.
    """
    plt = _pyplot()
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": (2, 1)}
    )
    axes[0].plot(time, reference, label="set_point", lw=1.2)
    axes[0].plot(time, response, ls="--", label="y", lw=1.1)
    axes[0].set_ylabel("y")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].plot(time, output, label="out", lw=1.0)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("out")
    axes[1].legend(loc="best", fontsize=8)
    _write(fig, path)


def _pyplot():
    """Matplotlib pyplot bound to the Agg backend.

    Returns
    -------
    module
        ``matplotlib.pyplot`` after forcing the Agg backend.
    """
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _write(fig, path) -> None:
    """Save a figure and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to write.
    path : str or pathlib.Path
        PNG destination. Parent directories are created.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    fig.clf()
    import matplotlib.pyplot as plt

    plt.close(fig)
