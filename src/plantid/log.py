"""Closed-loop log that identification reads."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from plantid.fit import Fit
from plantid.models import Model


@dataclass
class Log:
    """One proportional-loop recording.

    Attributes
    ----------
    time : numpy.ndarray
        Sample times, sorted.
    setpoint : numpy.ndarray
        Logged reference.
    measurement : numpy.ndarray
        Logged output.
    error : numpy.ndarray
        Logged ``setpoint - measurement``.
    p_term : numpy.ndarray
        Logged proportional term.
    """

    time: np.ndarray
    setpoint: np.ndarray
    measurement: np.ndarray
    error: np.ndarray
    p_term: np.ndarray

    @classmethod
    def read(cls, path) -> "Log":
        """Read a CSV log and sort it by time.

        Parameters
        ----------
        path : str or pathlib.Path
            CSV with ``time_s``, ``set_point``, ``measure_value``, ``error``,
            and ``p_term``.

        Returns
        -------
        Log
            Columns needed to identify the plant.
        """
        frame = pl.read_csv(Path(path)).sort("time_s")
        return cls(
            time=frame["time_s"].to_numpy(),
            setpoint=frame["set_point"].to_numpy(),
            measurement=frame["measure_value"].to_numpy(),
            error=frame["error"].to_numpy(),
            p_term=frame["p_term"].to_numpy(),
        )

    @property
    def kp(self) -> float:
        """Proportional gain recorded in the log.

        ``p_term`` is ``Kp * error`` before the output clip, so the ratio stays
        the collection gain on saturated samples too.

        Returns
        -------
        float
            ``median(p_term / error)`` over samples whose absolute error exceeds ``1e-9``.

        Raises
        ------
        ValueError
            Every error sample is effectively zero.
        """
        return self.gain(self.error, self.p_term)

    @staticmethod
    def gain(error: np.ndarray, p_term: np.ndarray) -> float:
        """Proportional gain ``median(p_term / error)``.

        Parameters
        ----------
        error : numpy.ndarray
            Logged ``setpoint - measurement``.
        p_term : numpy.ndarray
            Logged proportional term, the same length as ``error``.

        Returns
        -------
        float
            Median ratio over samples whose absolute error exceeds ``1e-9``.

        Raises
        ------
        ValueError
            Every error sample is effectively zero.
        """
        error = np.asarray(error, dtype=float)
        p_term = np.asarray(p_term, dtype=float)
        usable = np.abs(error) > 1e-9
        if not np.any(usable):
            raise ValueError("the log has no nonzero error, so P cannot be read")
        return float(np.median(p_term[usable] / error[usable]))

    def identify(self, models: list[Model] | None = None) -> Fit:
        """Fit the catalog to this log.

        Parameters
        ----------
        models : list of Model, optional
            Structures to fit. Default is :meth:`Model.catalog`.

        Returns
        -------
        Fit
            Fits sorted by increasing RMSE. The proportional gain is :attr:`kp`.
        """
        chosen = Model.catalog() if models is None else models
        return Fit.search(self.time, self.setpoint, self.measurement, chosen, kp=self.kp)
