"""Catalog of plant models the identification fits.

An inertial model of order ``n`` is

    G(s) = K / ((τ1 s + 1) ... (τn s + 1))

An integrator with ``n`` lags is

    G(s) = Kv / (s (τ1 s + 1) ... (τn s + 1))

``friction`` puts a Coulomb dead-zone on the plant input. :meth:`Model.fit`
estimates the gain, the time constants, and that threshold.
"""

from dataclasses import dataclass
from typing import Literal

import lmfit
import numpy as np

from plantid.plant import Plant

Kind = Literal["inertial", "integrator"]


@dataclass(frozen=True)
class Model:
    """One candidate plant structure.

    Attributes
    ----------
    name : str
        Label used in the fit table.
    kind : {"inertial", "integrator"}
        Plant family.
    order : int
        Number of real poles besides the integrator, if any.
    friction : bool
        Whether a Coulomb dead-zone is fitted on the input.
    """

    name: str
    kind: Kind
    order: int
    friction: bool

    def __post_init__(self) -> None:
        """Reject an impossible order.

        Raises
        ------
        ValueError
            ``order`` is negative, or an inertial model has order zero.
        """
        if self.order < 0:
            raise ValueError("order must be >= 0")
        if self.kind == "inertial" and self.order < 1:
            raise ValueError("an inertial model needs order >= 1")

    @classmethod
    def inertial(cls, order: int, *, friction: bool = False, name: str | None = None) -> "Model":
        """Inertial plant ``K / Π(τ s + 1)`` of the given order.

        Parameters
        ----------
        order : int
            Number of real poles. Must be at least 1.
        friction : bool, optional
            Fit a Coulomb input dead-zone. Default is false.
        name : str, optional
            Catalog label. The default is the Polish name used in the fit table.

        Returns
        -------
        Model
            Unfitted inertial structure.
        """
        label = name or f"inercyjny {order}" + (" + tarcie" if friction else "")
        return cls(label, "inertial", order, friction)

    @classmethod
    def integrator(
        cls, lags: int = 0, *, friction: bool = True, name: str | None = None
    ) -> "Model":
        """Integrator ``Kv / s`` with extra real poles.

        Parameters
        ----------
        lags : int, optional
            Number of extra real poles. Default is 0.
        friction : bool, optional
            Fit a Coulomb input dead-zone. Default is true.
        name : str, optional
            Catalog label. The default is the Polish name used in the fit table.

        Returns
        -------
        Model
            Unfitted integrator structure.
        """
        if name is not None:
            label = name
        elif lags == 0:
            label = "całkujący" + (" + tarcie" if friction else "")
        else:
            label = f"całkujący, inercja {lags}" + (" + tarcie" if friction else "")
        return cls(label, "integrator", lags, friction)

    @classmethod
    def catalog(cls) -> list["Model"]:
        """Structures searched by identification.

        Returns
        -------
        list of Model
            Inertial orders 1–3, each with and without friction, then integrators
            with 0, 1, and 2 lags and friction.
        """
        models = [
            cls.inertial(order, friction=friction)
            for order in (1, 2, 3)
            for friction in (False, True)
        ]
        models.extend(cls.integrator(lags, friction=True) for lags in (0, 1, 2))
        return models

    def fit(
        self,
        time: np.ndarray,
        setpoint: np.ndarray,
        measurement: np.ndarray,
        *,
        kp: float,
    ):
        """Output-error fit of this structure in a P loop with gain ``kp``.

        lmfit minimizes the squared error between the measured output and the
        simulated P loop. Gain, time constants, and friction are fitted in log space.

        Parameters
        ----------
        time : numpy.ndarray
            Sample times from the log.
        setpoint : numpy.ndarray
            Logged setpoint :math:`w(t)`.
        measurement : numpy.ndarray
            Logged sensor reading :math:`y_m(t)`. The first sample is the initial condition.
        kp : float
            Proportional gain of the loop that collected the log.

        Returns
        -------
        plantid.fit.FitResult
            Fitted plant, its prediction, and the output RMSE.
        """
        from plantid.fit import FitResult

        time = np.asarray(time, dtype=float)
        setpoint = np.asarray(setpoint, dtype=float)
        measurement = np.asarray(measurement, dtype=float)
        y0 = float(measurement[0])
        params = self._parameters()

        def residual(pars: lmfit.Parameters) -> np.ndarray:
            """Output error of the P loop at one parameter vector.

            Parameters
            ----------
            pars : lmfit.Parameters
                Log-space gain, time constants, and optional friction.

            Returns
            -------
            numpy.ndarray
                Simulated output minus the measurement.
            """
            plant = self._plant(self._vector(pars))
            return plant.proportional_response(time, setpoint, kp, y0) - measurement

        result = lmfit.minimize(
            residual,
            params,
            method="least_squares",
            max_nfev=40,
            xtol=1e-10,
            ftol=1e-8,
            gtol=1e-8,
        )
        plant = self._plant(self._vector(result.params))
        prediction = plant.proportional_response(time, setpoint, kp, y0)
        rmse = float(np.sqrt(np.mean(np.square(prediction - measurement))))
        return FitResult(self, plant, rmse, self._at_bound(result.params), prediction, float(kp))

    def _parameters(self) -> lmfit.Parameters:
        """Log-space parameters. Exponentiating them keeps gain, τ, and Fc positive.

        Returns
        -------
        lmfit.Parameters
            ``log_gain``, one ``log_tau_i`` per pole, and ``log_fc`` when the
            model has friction. Bounds are included.
        """
        n = self.order
        if self.kind == "inertial":
            gain0 = np.log(20.0 if self.friction else 8.0)
            tau0 = np.log(np.geomspace(30.0 if self.friction else 8.0, 0.1, n))
        else:
            gain0 = np.log(0.8)
            tau0 = np.log(np.geomspace(0.2, 0.05, n)) if n else np.array([])
        params = lmfit.Parameters()
        params.add("log_gain", value=float(gain0), min=-2.0, max=8.0)
        for index, value in enumerate(tau0):
            params.add(f"log_tau_{index}", value=float(value), min=-7.0, max=8.0)
        if self.friction:
            params.add("log_fc", value=float(np.log(10.0)), min=0.0, max=3.5)
        return params

    def _vector(self, params: lmfit.Parameters) -> np.ndarray:
        """Stack the log parameters in plant-builder order.

        Parameters
        ----------
        params : lmfit.Parameters
            Parameters from :meth:`_parameters`.

        Returns
        -------
        numpy.ndarray
            ``[log_gain, log_tau_0, ..., log_fc]``. ``log_fc`` is omitted without friction.
        """
        values = [params["log_gain"].value]
        values.extend(params[f"log_tau_{index}"].value for index in range(self.order))
        if self.friction:
            values.append(params["log_fc"].value)
        return np.asarray(values, dtype=float)

    @staticmethod
    def _at_bound(params: lmfit.Parameters) -> bool:
        """Report whether any parameter is within ``1e-3`` of a bound.

        Parameters
        ----------
        params : lmfit.Parameters
            Optimized parameters.

        Returns
        -------
        bool
            True when any value is that close to its minimum or maximum.
        """
        for par in params.values():
            if abs(par.value - par.min) < 1e-3 or abs(par.value - par.max) < 1e-3:
                return True
        return False

    def _plant(self, p: np.ndarray) -> Plant:
        """Build a plant from a log-parameter vector.

        Parameters
        ----------
        p : numpy.ndarray
            Vector from :meth:`_vector`.

        Returns
        -------
        Plant
            Plant with positive gain, time constants sorted descending, and Fc.
        """
        gain = float(np.exp(p[0]))
        taus = tuple(sorted((float(np.exp(v)) for v in p[1 : 1 + self.order]), reverse=True))
        fc = float(np.exp(p[-1])) if self.friction else 0.0
        return Plant(self.kind, gain, taus, fc)
