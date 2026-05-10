"""Perfusion volume bookkeeping for ambr®250-style constant-volume operation.

In perfusion culture, the reactor working volume is kept approximately constant
by balancing the perfusion (feed) rate against the sum of the bleed rate and the
harvest (permeate) rate:

.. math::
    \\frac{\\mathrm{d}V}{\\mathrm{d}t} = F_{\\text{feed}} - F_{\\text{bleed}} - F_{\\text{harvest}} \\approx 0

Hence:

.. math::
    F_{\\text{harvest}} = F_{\\text{feed}} - F_{\\text{bleed}}

When a VCV setpoint is exceeded, the bleed rate is transiently increased to
remove excess biomass (and cells that are retained by the ATF/TFF filter).

References
----------
.. [Gadiyar2026] Gadiyar et al. (2026), §2 ("Perfusion process description").
"""

from __future__ import annotations

import torch
from torch import Tensor


def perfusion_volume_step(
    V_t: Tensor | float,
    F_feed_L_per_h: float | Tensor,
    F_bleed_L_per_h: float | Tensor,
    F_harvest_L_per_h: float | Tensor,
    dt_hours: float = 24.0,
) -> Tensor:
    """Compute the reactor volume at time *t+1* from volumetric flow rates.

    For constant-volume operation, the caller should ensure that
    ``F_feed ≈ F_bleed + F_harvest``, in which case this function returns
    a volume very close to :math:`V_t`.

    Parameters
    ----------
    V_t:
        Working volume at time *t* [L].
    F_feed_L_per_h:
        Volumetric perfusion (feed) rate [L h⁻¹].
    F_bleed_L_per_h:
        Volumetric bleed rate [L h⁻¹].
    F_harvest_L_per_h:
        Volumetric permeate (harvest) rate [L h⁻¹].
    dt_hours:
        Time step [h]. Default 24.0.

    Returns
    -------
    Tensor
        Scalar tensor with the volume at *t+1* [L].

    Notes
    -----
    :math:`V_{t+1} = V_t + \\Delta t (F_{\\text{feed}} - F_{\\text{bleed}} - F_{\\text{harvest}})`

    If the result is negative (physically impossible), a ``RuntimeError`` is
    raised.  This should never occur under correct operating conditions.

    Examples
    --------
    >>> import torch
    >>> V_next = perfusion_volume_step(0.250, 0.250/24, 0.250*0.15/24, 0.250*0.85/24)
    >>> abs(float(V_next) - 0.250) < 1e-9
    True
    """
    V = torch.as_tensor(V_t, dtype=torch.float64)
    F_in = torch.as_tensor(F_feed_L_per_h, dtype=torch.float64)
    F_out = torch.as_tensor(F_bleed_L_per_h, dtype=torch.float64) + torch.as_tensor(
        F_harvest_L_per_h, dtype=torch.float64
    )
    V_next = V + dt_hours * (F_in - F_out)
    if float(V_next) < 0.0:
        msg = (
            f"Volume became negative ({float(V_next):.6f} L) at step. "
            "Check that perfusion rate ≥ bleed + harvest."
        )
        raise RuntimeError(msg)
    return V_next


def constant_volume_harvest_rate(
    perfusion_rate_L_per_h: float | Tensor,
    bleed_rate_L_per_h: float | Tensor,
) -> Tensor:
    """Return the harvest rate that maintains constant volume.

    .. math::
        F_{\\text{harvest}} = F_{\\text{feed}} - F_{\\text{bleed}}

    Parameters
    ----------
    perfusion_rate_L_per_h:
        Volumetric perfusion rate [L h⁻¹].
    bleed_rate_L_per_h:
        Volumetric bleed rate [L h⁻¹].

    Returns
    -------
    Tensor
        Scalar — harvest rate [L h⁻¹].

    Raises
    ------
    ValueError
        If bleed_rate > perfusion_rate (physically impossible steady state).

    Examples
    --------
    >>> import torch
    >>> F_h = constant_volume_harvest_rate(0.250/24, 0.250*0.15/24)
    >>> round(float(F_h / (0.250 / 24)), 4)
    0.85
    """
    F_feed = torch.as_tensor(perfusion_rate_L_per_h, dtype=torch.float64)
    F_bleed = torch.as_tensor(bleed_rate_L_per_h, dtype=torch.float64)
    F_harvest = F_feed - F_bleed
    if float(F_harvest) < -1e-9:
        msg = (
            f"Bleed rate ({float(F_bleed):.4f} L/h) exceeds "
            f"perfusion rate ({float(F_feed):.4f} L/h). "
            "This violates constant-volume operation."
        )
        raise ValueError(msg)
    return torch.clamp(F_harvest, min=0.0)


def bleed_trigger(
    vcv_current: float | Tensor,
    vcv_setpoint: float,
    bleed_rate_base_vvd: float,
    bleed_rate_max_vvd: float,
    k_bleed: float = 0.5,
) -> float:
    """Compute a VCV-controlled bleed rate (proportional controller).

    When ``vcv_current > vcv_setpoint``, the bleed rate is increased
    proportionally to the deviation.  This mimics the simple feedback
    controller used in ambr®250 automated perfusion protocols.

    .. math::
        F_{\\text{bleed}} = F_{\\text{base}} + k_{\\text{bleed}}
                            \\cdot \\max(\\text{VCV} - \\text{VCV}_{\\text{sp}}, 0)

    Parameters
    ----------
    vcv_current:
        Current volumetric cell volume [%].
    vcv_setpoint:
        Target VCV [%].
    bleed_rate_base_vvd:
        Base bleed rate [vvd] applied even at setpoint.
    bleed_rate_max_vvd:
        Maximum allowable bleed rate [vvd].
    k_bleed:
        Proportional gain [vvd per % VCV deviation].

    Returns
    -------
    float
        Bleed rate [vvd], clamped to ``[bleed_rate_base_vvd, bleed_rate_max_vvd]``.

    Examples
    --------
    >>> round(bleed_trigger(35.0, 30.0, 0.10, 0.40, k_bleed=0.05), 4)
    0.35
    """
    vcv = float(vcv_current)
    deviation = max(vcv - vcv_setpoint, 0.0)
    rate = bleed_rate_base_vvd + k_bleed * deviation
    return float(min(max(rate, bleed_rate_base_vvd), bleed_rate_max_vvd))
