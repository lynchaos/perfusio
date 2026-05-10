"""Predictive constraint alarm system for the digital twin.

Raises alarms when the *forecast* trajectory is predicted to violate process
constraints before the alarm occurs — giving operators lead time to intervene.

Alarm channels (configurable via :class:`~perfusio.config.AlarmConfig`):
- Console / logging (always active)
- Email (SMTP, requires ``EMAIL_*`` environment variables)
- Slack webhook (requires ``SLACK_WEBHOOK_URL``)

References
----------
.. [Gadiyar2026] Gadiyar et al. (2026), §3.4 (constraint monitoring).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AlarmEvent:
    """A single predictive alarm event."""

    day: int
    species: str
    predicted_value: float
    threshold: float
    direction: str  # "above" or "below"
    lead_days: int  # days ahead the violation is predicted


class AlarmNotifier:
    """Predictive alarm dispatcher for the digital twin.

    Parameters
    ----------
    thresholds:
        Dict mapping species name → ``(lo, hi)`` tuple.  If a predicted
        value crosses ``lo`` (below) or ``hi`` (above) within ``lead_days``,
        an alarm is raised.
    lead_days:
        How many days ahead to scan the forecast for violations.
    channels:
        List of channel names: ``"log"``, ``"email"``, ``"slack"``.

    Examples
    --------
    >>> from perfusio.twin.notifications import AlarmNotifier
    >>> notifier = AlarmNotifier(
    ...     thresholds={"Glc": (2.0, 8.0), "Amm": (0.0, 5.0)},
    ...     channels=["log"],
    ... )
    """

    def __init__(
        self,
        thresholds: dict[str, tuple[float, float]],
        lead_days: int = 3,
        channels: list[str] | None = None,
    ) -> None:
        self.thresholds = thresholds
        self.lead_days = lead_days
        self.channels = channels or ["log"]

    def check_forecast(
        self,
        forecast: dict[str, Any],
        species_names: list[str],
        current_day: int,
    ) -> list[AlarmEvent]:
        """Scan the horizon forecast for predicted constraint violations.

        Parameters
        ----------
        forecast:
            Output of :func:`~perfusio.hybrid.forecast.forecast_run` — dict
            with ``"mean"`` key, shape ``(horizon, n_species)``.
        species_names:
            Ordered list of species names matching the model output.
        current_day:
            Current culture day.

        Returns
        -------
        list[AlarmEvent]
            All violations predicted within the lead window.
        """
        mean_traj = forecast["mean"]  # (horizon, n_species)
        events: list[AlarmEvent] = []

        for j, name in enumerate(species_names):
            if name not in self.thresholds:
                continue
            lo, hi = self.thresholds[name]
            for t in range(mean_traj.shape[0]):
                val = float(mean_traj[t, j])
                day_pred = current_day + t + 1
                lead = t + 1
                if val < lo:
                    events.append(AlarmEvent(day_pred, name, val, lo, "below", lead))
                if val > hi:
                    events.append(AlarmEvent(day_pred, name, val, hi, "above", lead))

        for event in events:
            self._dispatch(event)

        return events

    def _dispatch(self, event: AlarmEvent) -> None:
        for channel in self.channels:
            if channel == "log":
                logger.warning(
                    "ALARM [Day %d, +%d days lead]: %s predicted %s threshold "
                    "(value=%.3f, threshold=%.3f)",
                    event.day,
                    event.lead_days,
                    event.species,
                    event.direction,
                    event.predicted_value,
                    event.threshold,
                )
            elif channel == "email":
                self._send_email(event)
            elif channel == "slack":
                self._send_slack(event)
            else:
                logger.error("Unknown alarm channel: %s", channel)

    def _send_email(self, event: AlarmEvent) -> None:
        """Send email alarm via SMTP (requires environment variables)."""
        import smtplib
        from email.message import EmailMessage

        smtp_host = os.environ.get("EMAIL_SMTP_HOST", "")
        smtp_port = int(os.environ.get("EMAIL_SMTP_PORT", "587"))
        sender = os.environ.get("EMAIL_SENDER", "")
        recipient = os.environ.get("EMAIL_RECIPIENT", "")
        password = os.environ.get("EMAIL_PASSWORD", "")

        if not all([smtp_host, sender, recipient]):
            logger.warning("Email alarm suppressed: EMAIL_* env vars not configured.")
            return

        msg = EmailMessage()
        msg["Subject"] = f"perfusio ALARM: {event.species} predicted {event.direction} threshold"
        msg["From"] = sender
        msg["To"] = recipient
        msg.set_content(
            f"Predictive alarm triggered:\n"
            f"  Species:   {event.species}\n"
            f"  Direction: {event.direction}\n"
            f"  Value:     {event.predicted_value:.4f}\n"
            f"  Threshold: {event.threshold:.4f}\n"
            f"  Pred. day: {event.day}\n"
            f"  Lead:      {event.lead_days} days\n"
        )

        try:
            with smtplib.SMTP(smtp_host, smtp_port) as s:
                s.starttls()
                if password:
                    s.login(sender, password)
                s.send_message(msg)
        except Exception:
            logger.exception("Failed to send alarm email.")

    def _send_slack(self, event: AlarmEvent) -> None:
        """Post alarm to Slack webhook (requires SLACK_WEBHOOK_URL)."""
        import json
        import urllib.request

        url = os.environ.get("SLACK_WEBHOOK_URL", "")
        if not url:
            logger.warning("Slack alarm suppressed: SLACK_WEBHOOK_URL not set.")
            return

        payload = json.dumps(
            {
                "text": (
                    f":warning: *perfusio ALARM* — `{event.species}` predicted "
                    f"{event.direction} threshold on day {event.day} "
                    f"(lead: {event.lead_days} d, value: {event.predicted_value:.3f}, "
                    f"threshold: {event.threshold:.3f})"
                )
            }
        ).encode()

        req = urllib.request.Request(  # — webhook URL set by operator
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    logger.warning("Slack alarm: unexpected status %d", resp.status)
        except Exception:
            logger.exception("Failed to post Slack alarm.")
