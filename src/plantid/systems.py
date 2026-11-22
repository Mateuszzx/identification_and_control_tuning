"""Build python-control transfer functions and a matching state-space form.

The state-space form is the phase-variable companion form of the same
polynomial, with the output equal to the last state. That makes the initial
condition ``y(0) = y0``, ``ẏ(0) = 0`` a one-line assignment.
"""

import control
import numpy as np


def companion(kind: str, gain: float, taus: tuple[float, ...]):
    """Phase-variable realization of the linear plant.

    The output is the last state, so ``C = [0 ... 1]``.

    Parameters
    ----------
    kind : {"inertial", "integrator"}
        ``"integrator"`` adds a free pole at the origin.
    gain : float
        Numerator gain, ``K`` or ``Kv``.
    taus : tuple of float
        Extra real time constants. Each contributes a factor ``(tau s + 1)``.

    Returns
    -------
    system : control.TransferFunction
        Linear plant ``G(s)``.
    a : numpy.ndarray
        State matrix of shape ``(n, n)``.
    b : numpy.ndarray
        Input vector of shape ``(n,)``.
    c : numpy.ndarray
        Output row of shape ``(n,)``, with ``c[-1] == 1``.
    """
    denominator = np.array([1.0, 0.0]) if kind == "integrator" else np.array([1.0])
    for tau in taus:
        denominator = np.convolve(denominator, [float(tau), 1.0])
    lead = float(denominator[0])
    coeff = denominator[1:] / lead
    n = coeff.size
    a = np.zeros((n, n))
    a[0, :] = -coeff
    if n > 1:
        a[1:, :-1] = np.eye(n - 1)
    b = np.zeros(n)
    b[0] = float(gain) / lead
    c = np.zeros(n)
    c[-1] = 1.0
    system = control.tf([float(gain)], denominator)
    return system, a, b, c


def closed_loop(system: control.TransferFunction, kp: float) -> control.TransferFunction:
    """Linear loop ``Kp G / (1 + Kp G)``.

    Friction is not part of this system.

    Parameters
    ----------
    system : control.TransferFunction
        Linear plant.
    kp : float
        Proportional gain wrapped around the plant.

    Returns
    -------
    control.TransferFunction
        Unity-feedback closed loop.
    """
    return control.feedback(kp * system, 1)
