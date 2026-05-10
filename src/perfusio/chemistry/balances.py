"""Discrete mass-balance implementation for perfusion bioreactors.

Implements Equations 2–4 of Gadiyar et al. (2026) in vectorised, batch-aware
PyTorch code.  The fundamental relation is:

.. math::
    \\frac{\\mathrm{d}(V \\cdot c_k)}{\\mathrm{d}t}
    = u_{f,k} - u_{b,k} - u_{p,k} + R_k \\cdot V

where

- :math:`c_k` is the concentration of species :math:`k` [g L⁻¹ or mmol L⁻¹],
- :math:`V` is the working volume [L],
- :math:`u_{f,k}` is the molar/mass feed rate [mmol h⁻¹ or g h⁻¹],
- :math:`u_{b,k}` is the bleed removal rate,
- :math:`u_{p,k}` is the permeate (harvest) removal rate,
- :math:`R_k` is the volumetric net production rate [mmol L⁻¹ h⁻¹ or g L⁻¹ h⁻¹].

In the discrete daily step form (Eq. 3 of the paper, rearranged):

.. math::
    R_k(s_i) = \\frac{c_{i+1,k} - c_{i,k}}{\\Delta t}
               - \\frac{1}{V_i}
                 \\left[ u_{f,k} - u_{b,k} - u_{p,k}
                         - c_{i,k} \\frac{\\Delta V}{\\Delta t} \\right]

This module computes :math:`R_k` analytically from observed trajectories
(for GP training) and also integrates :math:`c_{i+1}` forward given
:math:`R_k` (for forecast rollout).

References
----------
.. [Gadiyar2026] Gadiyar et al. (2026), Equations 2–4.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from perfusio.chemistry.species import Species
from perfusio.states import Trajectory


class DiscreteMassBalance:
    """Vectorised implementation of Gadiyar et al. (2026), Eq. 3.

    For each species :math:`k`:

    .. math::
        R_k(s_i) = \\frac{c_{i+1,k} - c_{i,k}}{\\Delta t}
                   - \\frac{1}{V_i}
                     \\left[ u_{f,k} - u_{b,k} - u_{p,k}
                             - c_{i,k} \\frac{\\Delta V}{\\Delta t} \\right]

    Parameters
    ----------
    species:
        Ordered sequence of :class:`~perfusio.chemistry.species.Species`
        objects whose concentrations are tracked by the balance.
    dt_hours:
        Discrete time step in hours.  Default is 24 h (daily sampling).

    Notes
    -----
    **Volume convention**:  For ambr®250 perfusion reactors in constant-volume
    mode, :math:`\\Delta V / \\Delta t = 0` because the harvest rate equals
    ``perfusion_rate − bleed_rate``.  The term is included here for generality
    but will be zero in most practical use cases.

    **Units**:  All concentrations must be in the *same* base unit throughout
    a given trajectory (e.g. all in mmol L⁻¹).  The balances do **not**
    convert units.  VCD is conventionally in 10⁶ cells mL⁻¹; the rate :math:`R_\\text{VCD}`
    will therefore be in 10⁶ cells mL⁻¹ h⁻¹.

    Examples
    --------
    Round-trip test: rates_from_observations then step should reconstruct
    the original trajectory within floating-point tolerance.

    >>> import torch
    >>> from perfusio.chemistry.species import SpeciesRegistry
    >>> reg = SpeciesRegistry.DEFAULT
    >>> species = [reg["VCD"], reg["Glc"]]
    >>> bal = DiscreteMassBalance(species, dt_hours=24.0)
    >>> # Construct a noiseless 5-day trajectory
    >>> T, K = 5, 2
    >>> c = torch.ones(T, K, dtype=torch.float64)
    >>> u_f = torch.zeros(T - 1, K, dtype=torch.float64)
    >>> u_b = torch.zeros(T - 1, K, dtype=torch.float64)
    >>> u_p = torch.zeros(T - 1, K, dtype=torch.float64)
    >>> V = torch.full((T,), 0.250, dtype=torch.float64)
    >>> R = bal.rates_from_observations(c, V, u_f, u_b, u_p)
    >>> c_recon = bal.step(c[0], R[0], V[0], u_f[0], u_b[0], u_p[0], 24.0)
    >>> torch.allclose(c_recon, c[1], atol=1e-10)
    True
    """

    def __init__(
        self,
        species: Sequence[Species],
        dt_hours: float = 24.0,
    ) -> None:
        if dt_hours <= 0:
            msg = f"dt_hours must be positive; got {dt_hours}."
            raise ValueError(msg)
        self._species = list(species)
        self.dt_hours = dt_hours
        self.n_species = len(self._species)

    @property
    def species(self) -> list[Species]:
        """Ordered list of tracked species."""
        return self._species

    def rates_from_observations(
        self,
        c: Tensor,
        V: Tensor,
        u_f: Tensor,
        u_b: Tensor,
        u_p: Tensor,
    ) -> Tensor:
        """Compute net volumetric rates :math:`R_k(s_i)` from observations.

        This implements Eq. 3 of Gadiyar et al. (2026) (rearranged to solve
        for :math:`R`), vectorised over all species and all time steps.

        Parameters
        ----------
        c:
            Concentration tensor, shape ``(T, n_species)`` [same unit as
            the corresponding :class:`~perfusio.chemistry.species.Species`].
        V:
            Working volume tensor, shape ``(T,)`` [L].
        u_f:
            Feed rate tensor, shape ``(T-1, n_species)``
            [concentration-unit × L h⁻¹ = amount h⁻¹].  Set to zero for
            species with no direct feed stream.
        u_b:
            Bleed removal rate, shape ``(T-1, n_species)``.
        u_p:
            Permeate (harvest) removal rate, shape ``(T-1, n_species)``.

        Returns
        -------
        Tensor
            Shape ``(T-1, n_species)`` — one rate vector per time interval.
            Units are [concentration-unit h⁻¹].

        Raises
        ------
        ValueError
            If tensor shapes are inconsistent.

        Notes
        -----
        The derivation follows directly from Eq. 3 of the paper.  Rearranging
        the discrete balance:

        .. math::
            R_k = \\frac{c_{i+1,k} - c_{i,k}}{\\Delta t}
                  - \\frac{u_{f,k} - u_{b,k} - u_{p,k}}{V_i}
                  + c_{i,k} \\frac{V_{i+1} - V_i}{V_i \\Delta t}

        The last term vanishes in constant-volume mode.
        """
        T = c.shape[0]
        self._validate_shapes(c, V, u_f, u_b, u_p, T)

        # Concentration difference (T-1, K)
        dc = c[1:] - c[:-1]  # (T-1, K)

        # Volume at step i, broadcast to (T-1, 1)
        V_i = V[:-1].unsqueeze(-1)  # (T-1, 1)

        # Volume change term (T-1, 1) — zero in constant-volume mode
        dV = (V[1:] - V[:-1]).unsqueeze(-1)  # (T-1, 1)

        # Net flow term: (u_f - u_b - u_p) is amount h⁻¹, divide by V to get
        # concentration-unit h⁻¹ (Eq. 3, right-hand side, second term)
        net_flow = (u_f - u_b - u_p) / V_i  # (T-1, K)

        # Volume-change dilution term (Eq. 3, last term)
        dil = c[:-1] * dV / (V_i * self.dt_hours)  # (T-1, K)

        # Eq. 3 rearranged for R:
        # R = dc/dt - net_flow + c * dV/(V*dt)
        R = dc / self.dt_hours - net_flow + dil  # (T-1, K)

        return R

    def step(
        self,
        c_t: Tensor,
        R_t: Tensor,
        V_t: Tensor,
        u_f: Tensor,
        u_b: Tensor,
        u_p: Tensor,
        dt: float | None = None,
    ) -> Tensor:
        """Integrate one discrete step forward (Eq. 2 of Gadiyar et al. 2026).

        Computes :math:`c_{t+1}` from :math:`c_t`, :math:`R_t`, and the
        mass-flow terms.

        .. math::
            c_{t+1,k} = c_{t,k} + \\Delta t \\left[
                R_k + \\frac{u_{f,k} - u_{b,k} - u_{p,k}}{V_t}
                - c_{t,k} \\frac{\\Delta V}{V_t \\Delta t}
            \\right]

        Parameters
        ----------
        c_t:
            Concentration at time *t*, shape ``(n_species,)`` or
            ``(B, n_species)`` for a batch.
        R_t:
            Net volumetric rate at time *t*, same shape as *c_t*.
        V_t:
            Working volume at time *t* [L], scalar or shape ``(B,)``.
        u_f:
            Feed rate at time *t*, same shape as *c_t*.
        u_b:
            Bleed rate at time *t*, same shape as *c_t*.
        u_p:
            Permeate rate at time *t*, same shape as *c_t*.
        dt:
            Override for the time step [h].  Defaults to :attr:`dt_hours`.

        Returns
        -------
        Tensor
            :math:`c_{t+1}`, same shape as *c_t*.

        Notes
        -----
        In constant-volume mode, pass ``V_t`` as a scalar and ``dV=0``.
        The term ``c_t * dV / (V_t * dt)`` vanishes automatically because
        the volume change is zero.

        Examples
        --------
        >>> import torch
        >>> from perfusio.chemistry.species import SpeciesRegistry
        >>> reg = SpeciesRegistry.DEFAULT
        >>> bal = DiscreteMassBalance([reg["VCD"], reg["Glc"]], dt_hours=24.0)
        >>> c = torch.tensor([10.0, 3.0], dtype=torch.float64)
        >>> R = torch.tensor([0.5, -0.1], dtype=torch.float64)
        >>> V = torch.tensor(0.250, dtype=torch.float64)
        >>> u = torch.zeros(2, dtype=torch.float64)
        >>> c_next = bal.step(c, R, V, u, u, u)
        >>> c_next.shape
        torch.Size([2])
        """
        dt = dt if dt is not None else self.dt_hours

        # Broadcast V_t for batch operation
        if c_t.dim() == 1:
            # Single-reactor path
            V_scalar = V_t if V_t.dim() == 0 else V_t.squeeze()
            net_flow = (u_f - u_b - u_p) / V_scalar
            # Constant-volume: dV = 0, last term vanishes
            c_next = c_t + dt * (R_t + net_flow)
        else:
            # Batch path: c_t shape (B, K), V_t shape (B,)
            V_col = V_t.unsqueeze(-1)  # (B, 1)
            net_flow = (u_f - u_b - u_p) / V_col  # (B, K)
            c_next = c_t + dt * (R_t + net_flow)

        return c_next

    def rates_from_trajectory(
        self,
        traj: Trajectory,
        species_indices: list[int] | None = None,
    ) -> Tensor:
        """Convenience wrapper: extract rates from a :class:`~perfusio.states.Trajectory`.

        Parameters
        ----------
        traj:
            A full run trajectory.
        species_indices:
            Column indices to extract from ``traj.species``.  If ``None``,
            all columns are used.

        Returns
        -------
        Tensor
            Shape ``(T-1, n_selected_species)``.
        """
        c = traj.species if species_indices is None else traj.species[:, species_indices]
        V = traj.volume_L

        # Build zero flow tensors (T-1, K) — caller is responsible for
        # providing real flows via a separate method when non-zero
        T_minus_1 = c.shape[0] - 1
        K = c.shape[1]
        zeros = torch.zeros(T_minus_1, K, dtype=c.dtype, device=c.device)

        return self.rates_from_observations(c, V, zeros, zeros, zeros)

    def _validate_shapes(
        self,
        c: Tensor,
        V: Tensor,
        u_f: Tensor,
        u_b: Tensor,
        u_p: Tensor,
        T: int,
    ) -> None:
        """Raise ``ValueError`` if tensor dimensions are incompatible."""
        K = self.n_species
        expected_c = (T, K)
        expected_V = (T,)
        expected_u = (T - 1, K)

        if c.shape != torch.Size(expected_c):
            msg = f"c shape {tuple(c.shape)} != expected {expected_c}."
            raise ValueError(msg)
        if V.shape != torch.Size(expected_V):
            msg = f"V shape {tuple(V.shape)} != expected {expected_V}."
            raise ValueError(msg)
        for name, u in [("u_f", u_f), ("u_b", u_b), ("u_p", u_p)]:
            if u.shape != torch.Size(expected_u):
                msg = f"{name} shape {tuple(u.shape)} != expected {expected_u}."
                raise ValueError(msg)


class FlowRateCalculator:
    """Compute molar/mass flow rates for feed, bleed, and permeate streams.

    Translates volumetric flow settings (in vessel volumes per day, vvd) into
    the per-species flow rates needed by :class:`DiscreteMassBalance`.

    Parameters
    ----------
    volume_L:
        Nominal reactor working volume [L].  For ambr®250, 0.250 L.
    dt_hours:
        Discrete time step [h].  Default 24.0 (daily).

    Notes
    -----
    In constant-volume perfusion mode:

    .. math::
        u_{b,k} = \\text{bleed\\_rate} \\cdot V \\cdot c_k / \\Delta t

    .. math::
        u_{p,k} = (\\text{perf\\_rate} - \\text{bleed\\_rate})
                   \\cdot V \\cdot c_k \\cdot (1 - \\sigma_k) / \\Delta t

    where :math:`\\sigma_k \\in [0,1]` is the sieving coefficient for species
    :math:`k` through the alternating tangential flow (ATF) filter.  For cells,
    :math:`\\sigma_{\\text{cell}} \\approx 0` (perfect retention).  For metabolites,
    :math:`\\sigma \\approx 1` (free passage).
    """

    def __init__(self, volume_L: float = 0.250, dt_hours: float = 24.0) -> None:
        self.volume_L = volume_L
        self.dt_hours = dt_hours

    def compute_flows(
        self,
        c: Tensor,
        perfusion_rate_vvd: float | Tensor,
        bleed_rate_vvd: float | Tensor,
        c_feed: Tensor | None = None,
        sieving_coeffs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute (u_f, u_b, u_p) for a single time step.

        Parameters
        ----------
        c:
            Current concentrations, shape ``(K,)`` [concentration units].
        perfusion_rate_vvd:
            Perfusion rate [vessel volumes per day].
        bleed_rate_vvd:
            Bleed rate [vvd].
        c_feed:
            Feed stream concentrations, shape ``(K,)``.  ``None`` implies zero.
        sieving_coeffs:
            Per-species sieving coefficient through ATF/TFF filter, shape ``(K,)``.
            ``0`` = fully retained (cells), ``1`` = freely passing (metabolites).
            Default: 1.0 for all (no retention).

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(u_f, u_b, u_p)`` each shape ``(K,)`` in
            [concentration-unit × L / h].

        Notes
        -----
        A harvest rate :math:`\\text{harvest} = \\text{perf} - \\text{bleed}` is
        applied only to freely passing species (:math:`\\sigma_k = 1`).  Cell-
        retained species (:math:`\\sigma_k = 0`) only leave via the bleed stream.
        """
        K = c.shape[0]
        dtype, device = c.dtype, c.device

        if sieving_coeffs is None:
            sieving_coeffs = torch.ones(K, dtype=dtype, device=device)

        if c_feed is None:
            c_feed = torch.zeros(K, dtype=dtype, device=device)

        # Convert vvd → h⁻¹ (1 vvd = V L per day = V/24 L per hour)
        V_per_h = self.volume_L / self.dt_hours  # volume exchanged per hour per vvd

        # Feed flow rate [concentration-unit × L h⁻¹]
        perf_rate_L_per_h = float(perfusion_rate_vvd) * V_per_h
        u_f = c_feed * perf_rate_L_per_h  # (K,)

        # Bleed: removes cells + metabolites at bulk concentration
        bleed_rate_L_per_h = float(bleed_rate_vvd) * V_per_h
        u_b = c * bleed_rate_L_per_h  # (K,)

        # Permeate (harvest): removes at concentration * (1 - sigma)
        # harvest = perf - bleed (constant-volume)
        harvest_rate_L_per_h = (float(perfusion_rate_vvd) - float(bleed_rate_vvd)) * V_per_h
        u_p = c * harvest_rate_L_per_h * (1.0 - sieving_coeffs)  # (K,)

        return u_f, u_b, u_p
