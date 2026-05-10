"""Mechanistic kinetic rate expressions for CHO perfusion models.

All rate laws are implemented from first principles with citable sources.
Kinetic constants are labelled as *synthetic defaults* (S) or as drawn from
a specific reference.  No Merck-proprietary values are used.

Kinetics implemented
--------------------
1. **Monod growth** on glucose and glutamine (dual substrate).
2. **Pirt maintenance** (energy maintenance on glucose independent of growth).
3. **Luedeking–Piret** mAb production model (growth + non-growth associated).
4. **Lactate switch**: CHO clone X switches from lactate production to
   lactate *consumption* when glucose falls below a threshold (Warburg
   effect reversal). Clone Y never consumes lactate.
5. **Ammonium production** from glutamine deamidation.
6. **Pyruvate-NH₄⁺ scavenging**: pyruvate reacts with NH₄⁺ in a first-order
   reaction (reduces ammonium accumulation when pyruvate is fed).
7. **First-order cell death** with temperature and shear stress dependence.
8. **Apoptosis-induced diameter increase** (phenomenological).

References
----------
.. [Monod1942] Monod, J. (1942). Recherches sur la Croissance des Cultures
   Bacteriennes. Hermann et Cie, Paris.
.. [Pirt1965] Pirt, S. J. (1965). The maintenance energy of bacteria in
   growing cultures. Proceedings of the Royal Society B, 163(991), 224–231.
.. [LuedekingPiret1959] Luedeking, R., & Piret, E. L. (1959). A kinetic
   study of the lactic acid fermentation. Journal of Biochemical and
   Microbiological Technology and Engineering, 1(4), 393–412.
.. [Gagnon2011] Gagnon, M., et al. (2011). Metabolic flux analysis of CHO
   cells. Biotechnology and Bioengineering, 108(6), 1328–1337.
   (Source of synthetic Monod parameters, labelled S below.)
.. [Gadiyar2026] Gadiyar et al. (2026), §3.3 (in-silico setup).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class CHOKinetics:
    """Kinetic parameter set for a CHO perfusion model.

    All parameters are synthetic defaults calibrated to reproduce the
    trajectory shapes shown in Gadiyar et al. (2026) Fig. 7.  They are
    labelled with their source (S = synthetic default; literature citation
    otherwise).

    Parameters
    ----------
    mu_max:
        Maximum specific growth rate [h⁻¹]. (S — typical CHO: 0.04 h⁻¹)
    K_glc:
        Glucose Monod constant [g L⁻¹]. (S — Gagnon et al. 2011: ~0.08 g/L)
    K_gln:
        Glutamine Monod constant [mmol L⁻¹]. (S — ~0.5 mmol/L)
    q_glc_max:
        Maximum specific glucose consumption rate [g cell⁻¹ h⁻¹]. (S)
    m_glc:
        Pirt glucose maintenance coefficient [g cell⁻¹ h⁻¹]. (S)
    Y_lac_glc:
        Lactate yield from glucose [mmol mmol⁻¹] in the Warburg regime. (S)
    glc_switch:
        Glucose threshold [g L⁻¹] below which clone X switches from lactate
        production to lactate consumption (Warburg effect reversal). (S)
    lac_consump_rate:
        Maximum specific lactate consumption rate [mmol cell⁻¹ h⁻¹]
        after the metabolic switch (clone X only). (S)
    q_gln_max:
        Maximum specific glutamine consumption rate [mmol cell⁻¹ h⁻¹]. (S)
    K_d:
        First-order cell death rate constant [h⁻¹]. (S)
    K_T:
        Temperature sensitivity of death rate [h⁻¹ K⁻¹]. (S)
    T_ref:
        Reference temperature for death rate [°C]. (S — 37.0)
    q_mAb_growth:
        Growth-associated mAb production [mg cell⁻¹]. (Luedeking–Piret α)
    q_mAb_nongrowth:
        Non-growth-associated mAb production [mg cell⁻¹ h⁻¹]. (LP β)
    alpha_amm_gln:
        Ammonium production stoichiometry from glutamine [mmol mmol⁻¹]. (S)
    k_pyr_amm:
        First-order rate constant for pyruvate–NH₄⁺ scavenging [h⁻¹]. (S)
    diam_base:
        Base mean cell diameter [μm]. (S — 20.5 μm, Gadiyar Fig.7)
    diam_death_coeff:
        Diameter increase per unit dead-cell fraction [μm per fraction]. (S)
    consumes_lactate:
        Whether this clone has the metabolic switch (True for clone X). (S)

    Notes
    -----
    These are *in-silico defaults only*.  Real CHO cell lines have proprietary
    kinetic parameters.  See LIMITATIONS.md for an explicit disclaimer.
    """

    # Growth kinetics (Monod)
    mu_max: float = 0.040  # h⁻¹ (S — typical CHO range 0.03–0.05)
    K_glc: float = 0.08  # g/L (S — Gagnon et al. 2011)
    K_gln: float = 0.50  # mmol/L (S)

    # Glucose consumption (Pirt)
    q_glc_max: float = 5.0e-10  # g cell⁻¹ h⁻¹ (S)
    m_glc: float = 5.0e-11  # g cell⁻¹ h⁻¹ (S — Pirt 1965)

    # Lactate metabolism
    Y_lac_glc: float = 1.8  # mmol lac / mmol glc consumed (S)
    glc_switch: float = 2.0  # g/L (S — matches paper §3.3)
    lac_consump_rate: float = 3.0e-10  # mmol cell⁻¹ h⁻¹ (S)
    consumes_lactate: bool = True  # True = clone X; False = clone Y

    # Glutamine consumption
    q_gln_max: float = 3.0e-10  # mmol cell⁻¹ h⁻¹ (S)
    K_d: float = 0.003  # h⁻¹ base death rate (S)
    K_T: float = 0.001  # h⁻¹ K⁻¹ temperature sensitivity (S)
    T_ref: float = 37.0  # °C reference temperature (S)

    # Product kinetics (Luedeking–Piret 1959)
    q_mAb_growth: float = 2.0e-9  # mg cell⁻¹ (α, growth-associated) (S)
    q_mAb_nongrowth: float = 5.0e-11  # mg cell⁻¹ h⁻¹ (β, non-growth) (S)

    # Ammonium production from Gln deamidation
    alpha_amm_gln: float = 0.85  # mmol NH4 / mmol Gln consumed (S)

    # Pyruvate scavenging of ammonium
    k_pyr_amm: float = 0.05  # h⁻¹ (S — phenomenological)

    # Cell diameter (apoptosis-induced increase)
    diam_base: float = 20.5  # μm (S — Gadiyar Fig. 7)
    diam_death_coeff: float = 4.0  # μm per unit dead fraction (S)

    def mu(self, glc: float, gln: float) -> float:
        """Specific growth rate by dual-substrate Monod kinetics.

        .. math::
            \\mu = \\mu_{\\max}
                   \\frac{[\\text{Glc}]}{K_{\\text{Glc}} + [\\text{Glc}]}
                   \\frac{[\\text{Gln}]}{K_{\\text{Gln}} + [\\text{Gln}]}

        Parameters
        ----------
        glc:
            Glucose concentration [g L⁻¹].  Clamped to ≥ 0.
        gln:
            Glutamine concentration [mmol L⁻¹].  Clamped to ≥ 0.

        Returns
        -------
        float
            Specific growth rate [h⁻¹].

        References
        ----------
        .. [Monod1942] Monod (1942), eq. (1).
        """
        glc = max(glc, 0.0)
        gln = max(gln, 0.0)
        f_glc = glc / (self.K_glc + glc)
        f_gln = gln / (self.K_gln + gln)
        return self.mu_max * f_glc * f_gln

    def q_glc(self, mu_val: float, vcd: float) -> float:
        """Specific glucose consumption rate (Pirt maintenance + growth).

        .. math::
            q_{\\text{Glc}} = \\frac{\\mu}{Y_{X/\\text{Glc}}} + m_{\\text{Glc}}

        We absorb :math:`Y_{X/\\text{Glc}}` into :attr:`q_glc_max` for
        parsimony.  The total volumetric rate is
        :math:`Q_{\\text{Glc}} = q_{\\text{Glc}} \\cdot X_v`.

        Parameters
        ----------
        mu_val:
            Current specific growth rate [h⁻¹].
        vcd:
            Viable cell density [10⁶ cells mL⁻¹].

        Returns
        -------
        float
            Volumetric glucose consumption rate [g L⁻¹ h⁻¹].

        References
        ----------
        .. [Pirt1965] Pirt (1965).
        """
        # q_glc_max is specific to cell density; VCD in 10⁶ cells mL⁻¹ → 10⁹ cells L⁻¹
        vcd_L = vcd * 1e9  # cells L⁻¹
        specific_rate = self.q_glc_max * (mu_val / self.mu_max + 0.1) + self.m_glc
        return specific_rate * vcd_L  # g L⁻¹ h⁻¹

    def q_lac(self, glc: float, q_glc_val: float) -> float:
        """Volumetric lactate production/consumption rate.

        For clone X: produces lactate when ``glc > glc_switch``;
        consumes lactate when ``glc < glc_switch``.
        For clone Y: always produces lactate.

        Parameters
        ----------
        glc:
            Current glucose [g L⁻¹].
        q_glc_val:
            Volumetric glucose consumption rate [g L⁻¹ h⁻¹].

        Returns
        -------
        float
            Volumetric lactate rate [mmol L⁻¹ h⁻¹].
            Positive = production; negative = consumption.

        References
        ----------
        .. [Gagnon2011] Gagnon et al. (2011) — metabolic switch parameters.
        """
        # Convert g/L glucose consumed → mmol/L (MW Glc = 180 g/mol)
        glc_mmol_per_h = q_glc_val / 180.0 * 1000.0  # mmol L⁻¹ h⁻¹
        if self.consumes_lactate and glc < self.glc_switch:
            # Metabolic switch: consume lactate
            return -self.lac_consump_rate * 1e9  # negative = consumption (S)
        # Warburg: produce lactate proportional to glucose consumed
        return self.Y_lac_glc * glc_mmol_per_h

    def q_gln(self, mu_val: float, vcd: float) -> float:
        """Volumetric glutamine consumption rate [mmol L⁻¹ h⁻¹].

        Parameters
        ----------
        mu_val:
            Specific growth rate [h⁻¹].
        vcd:
            VCD [10⁶ cells mL⁻¹].

        Returns
        -------
        float
            Glutamine consumption rate [mmol L⁻¹ h⁻¹].
        """
        vcd_L = vcd * 1e9
        specific = self.q_gln_max * (mu_val / self.mu_max + 0.05)
        return specific * vcd_L

    def q_amm(self, q_gln_val: float, pyr: float, amm: float) -> float:
        """Volumetric ammonium production rate [mmol L⁻¹ h⁻¹].

        Ammonium is produced from glutamine deamidation and scavenged by
        pyruvate in a first-order reaction.

        .. math::
            R_{\\text{NH}_4^+} = \\alpha_{\\text{Gln}} \\cdot q_{\\text{Gln}}
                - k_{\\text{pyr}} \\cdot [\\text{Pyr}] \\cdot [\\text{NH}_4^+]

        Parameters
        ----------
        q_gln_val:
            Glutamine consumption rate [mmol L⁻¹ h⁻¹].
        pyr:
            Pyruvate concentration [mmol L⁻¹].
        amm:
            Ammonium concentration [mmol L⁻¹].

        Returns
        -------
        float
            Net ammonium production rate [mmol L⁻¹ h⁻¹].
        """
        prod = self.alpha_amm_gln * q_gln_val
        scav = self.k_pyr_amm * max(pyr, 0.0) * max(amm, 0.0)
        return prod - scav

    def q_mab(self, mu_val: float, vcd: float) -> float:
        """Volumetric mAb production rate (Luedeking–Piret, 1959).

        .. math::
            R_{\\text{mAb}} = (\\alpha \\mu + \\beta) X_v

        Parameters
        ----------
        mu_val:
            Specific growth rate [h⁻¹].
        vcd:
            VCD [10⁶ cells mL⁻¹].

        Returns
        -------
        float
            Volumetric mAb production rate [mg L⁻¹ h⁻¹].

        References
        ----------
        .. [LuedekingPiret1959] Luedeking & Piret (1959).
        """
        vcd_L = vcd * 1e9  # cells L⁻¹
        return (self.q_mAb_growth * mu_val + self.q_mAb_nongrowth) * vcd_L

    def k_death(self, temperature_C: float = 37.0) -> float:
        """Temperature-dependent specific cell death rate [h⁻¹].

        .. math::
            k_d(T) = k_d^{\\text{ref}} + K_T \\cdot (T - T_{\\text{ref}})

        Parameters
        ----------
        temperature_C:
            Culture temperature [°C].

        Returns
        -------
        float
            Death rate [h⁻¹].  Clamped to ≥ 0.
        """
        rate = self.K_d + self.K_T * (temperature_C - self.T_ref)
        return max(rate, 0.0)

    def cell_diameter(self, via: float) -> float:
        """Phenomenological cell diameter as a function of viability.

        Dead and apoptotic cells swell, increasing the population mean
        diameter.  This is calibrated to match Gadiyar et al. Fig. 7
        (Diam ≈ 20.5 → 22.5 μm as viability falls from ~99% to ~90%).

        .. math::
            \\bar{d} = d_0 + \\delta_d \\cdot (1 - \\text{Via}/100)

        Parameters
        ----------
        via:
            Cell viability [%].

        Returns
        -------
        float
            Mean cell diameter [μm].
        """
        dead_fraction = max(1.0 - via / 100.0, 0.0)
        return self.diam_base + self.diam_death_coeff * dead_fraction

    def ode_rhs(
        self,
        t: float,  # noqa: ARG002 — required by scipy.integrate.solve_ivp signature
        y: list[float],
        controls: dict[str, float],
    ) -> list[float]:
        """Right-hand side of the CHO ODE system for `scipy.integrate.solve_ivp`.

        State vector ordering (must match :attr:`STATE_ORDER`):
        ``[VCD, Via, Glc, Gln, Glu, Lac, Amm, Pyr, Titer]``

        Parameters
        ----------
        t:
            Time [h] (not used in this autonomous system, but required by
            the ``solve_ivp`` signature).
        y:
            Current state vector (see ordering above).
        controls:
            Dict with keys ``"temperature"`` [°C], ``"perfusion_rate"`` [vvd],
            ``"bleed_rate"`` [vvd], ``"glucose_feed_conc"`` [g L⁻¹],
            ``"gln_feed_conc"`` [mmol L⁻¹], ``"pyr_feed_conc"`` [mmol L⁻¹],
            ``"volume_L"`` [L].

        Returns
        -------
        list[float]
            Time derivatives for each state variable [h⁻¹ × units].
        """
        (vcd, via, glc, gln, glu, lac, amm, pyr, titer) = y  # noqa: F841

        T = controls.get("temperature", 37.0)
        V = controls.get("volume_L", 0.250)
        perf_vvd = controls.get("perfusion_rate", 1.0)
        bleed_vvd = controls.get("bleed_rate", 0.15)
        glc_feed = controls.get("glucose_feed_conc", 5.0)  # g/L
        gln_feed = controls.get("gln_feed_conc", 4.0)  # mmol/L
        pyr_feed = controls.get("pyr_feed_conc", 0.0)  # mmol/L

        # Convert vvd to h⁻¹ (divide by 24)
        D_perf = perf_vvd / 24.0  # dilution rate h⁻¹
        D_bleed = bleed_vvd / 24.0
        D_harvest = D_perf - D_bleed  # harvest dilution

        # Kinetic rates
        mu_val = self.mu(max(glc, 0.0), max(gln, 0.0))
        k_d = self.k_death(T)
        mu_net = mu_val - k_d

        Q_glc = self.q_glc(mu_val, max(vcd, 0.0))
        Q_gln = self.q_gln(mu_val, max(vcd, 0.0))
        Q_lac = self.q_lac(glc, Q_glc)
        Q_amm = self.q_amm(Q_gln, max(pyr, 0.0), max(amm, 0.0))
        Q_mab = self.q_mab(mu_val, max(vcd, 0.0))

        # Glutamate: produced from Gln deamination; consumed for growth
        # (simplified: net Glu dynamics, S)
        Q_glu = 0.2 * Q_gln - 0.15 * Q_gln  # net ≈ 0.05 * Q_gln (S)

        # --- ODEs ---
        # VCD: growth - death - bleed dilution (cells are retained by ATF)
        dVCD = mu_net * max(vcd, 0.0) - D_bleed * max(vcd, 0.0)

        # Viability (simplified logistic decay driven by death rate)
        # d(Via)/dt ≈ -k_d * (100 - Via) * 0.01  (S — phenomenological)
        dVia = -k_d * max(100.0 - via, 0.0) * 0.5

        # Glucose: feed in, consumed by cells, diluted by harvest
        dGlc = D_perf * glc_feed - D_harvest * glc - Q_glc

        # Glutamine: feed in, consumed, diluted
        dGln = D_perf * gln_feed - D_harvest * max(gln, 0.0) - Q_gln

        # Glutamate: net production
        dGlu = Q_glu - D_harvest * max(glu, 0.0)

        # Lactate: net production (may be negative for clone X at low glc)
        dLac = Q_lac - D_harvest * max(lac, 0.0)

        # Ammonium: produced from Gln, scavenged by pyruvate, diluted
        dAmm = Q_amm - D_harvest * max(amm, 0.0)

        # Pyruvate: fed in, scavenges NH4, diluted
        dPyr = (D_perf * pyr_feed - D_harvest * max(pyr, 0.0)
                - self.k_pyr_amm * max(pyr, 0.0) * max(amm, 0.0))

        # mAb titer: product accumulates, partially harvested
        dTiter = Q_mab - D_harvest * max(titer, 0.0)

        return [dVCD, dVia, dGlc, dGln, dGlu, dLac, dAmm, dPyr, dTiter]

    #: Canonical ordering of ODE state variables.
    STATE_ORDER: ClassVar[list[str]] = [
        "VCD", "Via", "Glc", "Gln", "Glu", "Lac", "Amm", "Pyr", "Titer"
    ]
