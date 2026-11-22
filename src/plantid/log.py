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
        Logged setpoint ``set_point``, the signal :math:`w(t)`.
    measurement : numpy.ndarray
        Logged sensor reading ``measure_value``, the signal :math:`y_m(t)`.
    """

    time: np.ndarray
    setpoint: np.ndarray
    measurement: np.ndarray

    @classmethod
    def read(cls, path) -> "Log":
        """Read a CSV log and sort it by time.

        Extra columns are ignored. The proportional gain is not in the file;
        pass it to :meth:`identify`.

        Parameters
        ----------
        path : str or pathlib.Path
            CSV with ``time_s``, ``set_point``, and ``measure_value``.

        Returns
        -------
        Log
            Time, setpoint, and measurement.
        """
        frame = pl.read_csv(Path(path)).sort("time_s")
        return cls(
            time=frame["time_s"].to_numpy(),
            setpoint=frame["set_point"].to_numpy(),
            measurement=frame["measure_value"].to_numpy(),
        )

    def identify(self, kp: float, models: list[Model] | None = None) -> Fit:
        """Fit the catalog to this log in a P loop with gain ``kp``.

        Parameters
        ----------
        kp : float
            Proportional gain of the loop that collected the log. This is
            ``identify.kp`` in the config. It is not fitted.
        models : list of Model, optional
            Structures to fit. Default is :meth:`Model.catalog`.

        Returns
        -------
        Fit
            Fits sorted by increasing RMSE. The proportional gain is ``kp``.
        """
        chosen = Model.catalog() if models is None else models
        return Fit.search(self.time, self.setpoint, self.measurement, chosen, kp=float(kp))
