"""Catalog of plant models the identification fits.

A structure is

    G(s) = K (ζ1 s + 1)...(ζm s + 1) / (s^q (τ1 s + 1)...(τn s + 1))

``q`` is the integrator power, ``m`` is the number of real derivative zeros,
and ``n`` is the number of real lags. ``q = 0`` is inertial. ``q = 1`` and
``m = 0`` is a single integrator. ``friction`` puts a Coulomb dead-zone on
the plant input. :meth:`Model.fit`
estimates the gain, the time constants, and that threshold. The structures
to try are read from a YAML file, not listed in this module.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import lmfit
import numpy as np
import yaml
from tqdm.auto import tqdm

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
        ``"integrator"`` when ``integrators`` is positive, otherwise ``"inertial"``.
    order : int
        Number of real lag poles, ``(τ s + 1)``.
    friction : bool
        Whether a Coulomb dead-zone is fitted on the input.
    integrators : int
        Power of ``1/s``. Default is 0.
    zeros : int
        Number of real derivative factors, ``(ζ s + 1)``. Default is 0.
    """

    name: str
    kind: Kind
    order: int
    friction: bool
    integrators: int = 0
    zeros: int = 0

    def __post_init__(self) -> None:
        """Reject a plant that is empty or not strictly proper.

        Raises
        ------
        ValueError
            A count is negative, there is no pole, the zeros are not fewer
            than the poles, or ``kind`` disagrees with ``integrators``.
        """
        if self.order < 0 or self.integrators < 0 or self.zeros < 0:
            raise ValueError("order, integrators, and zeros must be >= 0")
        if self.integrators + self.order < 1:
            raise ValueError("the plant needs at least one pole")
        if self.zeros > self.integrators + self.order:
            raise ValueError("zeros must not outnumber poles, or the plant is improper")
        expected = "integrator" if self.integrators else "inertial"
        if self.kind != expected:
            raise ValueError("kind must be integrator exactly when integrators is positive")

    @classmethod
    def inertial(
        cls,
        order: int,
        *,
        friction: bool = False,
        zeros: int = 0,
        name: str | None = None,
    ) -> "Model":
        """Inertial plant of the given order, with optional real zeros.

        Parameters
        ----------
        order : int
            Number of real poles. Must be at least 1 when there is no integrator.
        friction : bool, optional
            Fit a Coulomb input dead-zone. Default is false.
        zeros : int, optional
            Number of real derivative factors. Default is 0.
        name : str, optional
            Catalog label. The default describes the structure.

        Returns
        -------
        Model
            Unfitted inertial structure.
        """
        return cls._make(0, order, zeros, friction, name)

    @classmethod
    def integrator(
        cls,
        lags: int = 0,
        *,
        friction: bool = False,
        zeros: int = 0,
        integrators: int = 1,
        name: str | None = None,
    ) -> "Model":
        """Integrator of the given power, with extra real poles and zeros.

        Parameters
        ----------
        lags : int, optional
            Number of extra real poles. Default is 0.
        friction : bool, optional
            Fit a Coulomb input dead-zone. Default is false.
        zeros : int, optional
            Number of real derivative factors. Default is 0.
        integrators : int, optional
            Power of ``1/s``. Default is 1.
        name : str, optional
            Catalog label. The default describes the structure.

        Returns
        -------
        Model
            Unfitted integrator structure.
        """
        return cls._make(integrators, lags, zeros, friction, name)

    @classmethod
    def _make(
        cls,
        integrators: int,
        order: int,
        zeros: int,
        friction: bool,
        name: str | None,
    ) -> "Model":
        """Build one structure, naming it when ``name`` is omitted.

        Parameters
        ----------
        integrators : int
            Power of ``1/s``.
        order : int
            Number of real lag poles.
        zeros : int
            Number of real derivative factors.
        friction : bool
            Whether the input dead-zone is fitted.
        name : str or None
            Catalog label. ``None`` builds one from the counts.

        Returns
        -------
        Model
            Unfitted structure.
        """
        kind = "integrator" if integrators else "inertial"
        if name is None:
            if integrators == 0:
                label = f"inertial {order}"
            elif integrators == 1 and order == 0:
                label = "integrator"
            elif integrators == 1:
                label = f"integrator, lag {order}"
            else:
                label = f"integrator {integrators}, lag {order}"
            if zeros:
                label += f", derivative {zeros}"
            if friction:
                label += " + friction"
        else:
            label = name
        return cls(label, kind, order, friction, integrators, zeros)

    @classmethod
    def read(cls, path) -> list["Model"]:
        """Read the structures identification should try.

        Parameters
        ----------
        path : str or pathlib.Path
            YAML list. Each item has ``order`` and ``friction``. ``kind`` is
            ``inertial`` or ``integrator``. ``integrators`` is the power of
            ``1/s`` and overrides ``kind``. ``zeros`` is the number of real
            derivative factors. ``name`` is optional.

        Returns
        -------
        list of Model
            One model per list item, in file order.

        Raises
        ------
        ValueError
            The file is empty, is not a list, or an item is not a usable model.
        """
        source = Path(path)
        data = yaml.safe_load(source.read_text())
        if not isinstance(data, list) or not data:
            raise ValueError(f"{source} must be a non-empty list of models")
        return [cls._entry(item, source) for item in data]

    @classmethod
    def _entry(cls, item, path: Path) -> "Model":
        """Build one model from a YAML mapping.

        Parameters
        ----------
        item : object
            One list item from the models file.
        path : pathlib.Path
            File path, used in error messages.

        Returns
        -------
        Model
            Unfitted structure.

        Raises
        ------
        ValueError
            ``item`` is not a mapping, or ``kind`` or ``order`` is unusable.
        """
        if not isinstance(item, dict):
            raise ValueError(f"{path} has an entry that is not a mapping")
        try:
            order = int(item["order"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path} needs an integer order") from exc
        kind = item.get("kind")
        if "integrators" in item:
            try:
                integrators = int(item["integrators"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path} needs an integer integrators count") from exc
        elif kind == "integrator":
            integrators = 1
        elif kind in (None, "inertial"):
            integrators = 0
        else:
            raise ValueError(f"{path} has unknown kind {kind!r}")
        try:
            zeros = int(item.get("zeros", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} needs an integer zeros count") from exc
        friction = bool(item.get("friction", False))
        name = item.get("name")
        name = None if name is None else str(name)
        return cls._make(integrators, order, zeros, friction, name)

    def fit(
        self,
        time: np.ndarray,
        setpoint: np.ndarray,
        measurement: np.ndarray,
        *,
        kp: float,
        max_nfev: int = 40,
    ):
        """Output-error fit of this structure in a P loop with gain ``kp``.

        lmfit minimizes the squared error between the measured output and the
        simulated P loop. Gain, time constants, and friction are fitted in log space.
        An interactive terminal counts residual evaluations. The bar stays
        hidden when stderr is not a terminal.

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
        max_nfev : int, optional
            Residual evaluations allowed for this fit. Default is 40.

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

        limit = int(max_nfev)
        bar = tqdm(total=limit, desc=self.name, unit="eval", leave=False, disable=None)

        def _progress(_pars, nfev, _resid, *_args, **_kws) -> None:
            """Advance the residual bar. ``nfev`` is capped at ``limit``."""
            bar.n = min(int(nfev), limit)
            bar.refresh()

        try:
            result = lmfit.minimize(
                residual,
                params,
                method="least_squares",
                max_nfev=limit,
                xtol=1e-10,
                ftol=1e-8,
                gtol=1e-8,
                iter_cb=_progress,
            )
        finally:
            bar.close()
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
        if self.integrators == 0:
            gain0 = np.log(20.0 if self.friction else 8.0)
            tau0 = np.log(np.geomspace(30.0 if self.friction else 8.0, 0.1, n))
        else:
            gain0 = np.log(0.8)
            tau0 = np.log(np.geomspace(0.2, 0.05, n)) if n else np.array([])
        zero0 = np.log(np.geomspace(0.3, 0.05, self.zeros)) if self.zeros else np.array([])
        params = lmfit.Parameters()
        params.add("log_gain", value=float(gain0), min=-2.0, max=8.0)
        for index, value in enumerate(tau0):
            params.add(f"log_tau_{index}", value=float(value), min=-7.0, max=8.0)
        for index, value in enumerate(zero0):
            params.add(f"log_zero_{index}", value=float(value), min=-7.0, max=8.0)
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
            ``[log_gain, log_tau_*, log_zero_*, log_fc]``. ``log_fc`` is omitted without friction.
        """
        values = [params["log_gain"].value]
        values.extend(params[f"log_tau_{index}"].value for index in range(self.order))
        values.extend(params[f"log_zero_{index}"].value for index in range(self.zeros))
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
        cursor = 1
        pole_logs = p[cursor : cursor + self.order]
        taus = tuple(sorted((float(np.exp(v)) for v in pole_logs), reverse=True))
        cursor += self.order
        zeros = tuple(
            sorted((float(np.exp(v)) for v in p[cursor : cursor + self.zeros]), reverse=True)
        )
        cursor += self.zeros
        fc = float(np.exp(p[cursor])) if self.friction else 0.0
        return Plant(self.kind, gain, taus, fc, integrators=self.integrators, zeros=zeros)
