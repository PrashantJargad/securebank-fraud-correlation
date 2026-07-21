"""
Correlation engine.

Interface (unchanged, so the app never has to change):
    score_transfer(session_id, amount, balance_before,
                   is_new_beneficiary, seconds_since_beneficiary_added)
      -> (score, level, reasons[list], action_taken)

It still does the *correlation* itself: it pulls the whole session from the
event log and combines the security-side signals (how the user got in) with the
transaction-side signals (what they're doing now). What changed is the verdict —
that now comes from the trained model in ml.py (Route 1). If the model can't be
loaded for any reason, it transparently falls back to the original rule engine,
so the app always works.
"""

from models import Event
from features import NEW_BENEFICIARY_WINDOW_MIN
import ml

HIGH_THRESHOLD = 70
MEDIUM_THRESHOLD = 40

_model = None


def _get_model():
    """Load the trained model once and cache it; retry until it's available."""
    global _model
    if _model is None:
        try:
            _model = ml.load_model()
        except Exception:
            _model = None
    return _model


def _session_signals(session_id, amount, balance_before, seconds_since_beneficiary_added):
    """Correlate: read the session's event log and assemble the feature signals."""
    events = Event.query.filter_by(session_id=session_id).all()

    new_device = int(any(e.action == "login_success" and e.is_new_device for e in events))
    unusual_location = int(any(e.is_unusual_location for e in events))
    impossible_travel = int(any(e.impossible_travel for e in events))
    used_vpn = int(any(e.used_vpn for e in events))
    credentials_breached = int(any(e.credentials_breached for e in events))
    failed_logins = sum(1 for e in events if e.action == "login_fail")

    if seconds_since_beneficiary_added is None:
        ben_age_min = 43200.0                      # treat as an old, trusted payee
    else:
        ben_age_min = seconds_since_beneficiary_added / 60.0
    amount_fraction = (amount / balance_before) if balance_before > 0 else 0.0

    return {
        "new_device": new_device,
        "unusual_location": unusual_location,
        "impossible_travel": impossible_travel,
        "used_vpn": used_vpn,
        "credentials_breached": credentials_breached,
        "failed_logins": failed_logins,
        "beneficiary_age_min": ben_age_min,
        "amount": float(amount),
        "amount_fraction": float(amount_fraction),
    }


def _verdict(score):
    if score >= HIGH_THRESHOLD:
        return "HIGH", "blocked"
    if score >= MEDIUM_THRESHOLD:
        return "MEDIUM", "flagged"
    return "LOW", "cleared"


def score_transfer(session_id, amount, balance_before,
                   is_new_beneficiary, seconds_since_beneficiary_added):
    signals = _session_signals(session_id, amount, balance_before,
                               seconds_since_beneficiary_added)
    model = _get_model()

    if model is not None:
        prob = ml.predict_proba(model, signals)
        score = int(round(prob * 100))
        level, action = _verdict(score)
        reasons = ml.explain(model, signals)
        return score, level, reasons, action

    # ---- fallback: original rule engine (only if the model is unavailable) ----
    return _rule_score(signals)


# ---------------------------------------------------------------------------
# Legacy rule engine — kept as a safety net and as the baseline to compare the
# model against in your write-up.
# ---------------------------------------------------------------------------
_W = dict(new_device=35, unusual_location=30, failed=15, new_ben=20,
          rapid=30, large_abs=15, large_rel=25)
RAPID_MIN = 10
LARGE_ABS = 40000
LARGE_REL = 0.5


def _rule_score(s):
    score = 0
    reasons = []
    if s["new_device"]:
        score += _W["new_device"]; reasons.append("Login from a new, unrecognised device")
    if s["unusual_location"]:
        score += _W["unusual_location"]; reasons.append("Login from an unusual location")
    if s.get("impossible_travel"):
        score += 40; reasons.append("Impossible travel between consecutive logins")
    if s.get("used_vpn"):
        score += 15; reasons.append("Login came through a VPN / proxy")
    if s.get("credentials_breached"):
        score += 25; reasons.append("Account credentials appear in a known data breach")
    if s["failed_logins"] >= 1:
        score += _W["failed"]; reasons.append(f"{s['failed_logins']} failed login attempt(s) this session")
    if s["beneficiary_age_min"] <= NEW_BENEFICIARY_WINDOW_MIN:
        score += _W["new_ben"]; reasons.append("Transfer to a recently added beneficiary")
    if s["beneficiary_age_min"] <= RAPID_MIN:
        score += _W["rapid"]; reasons.append(f"Beneficiary added {round(s['beneficiary_age_min'],1)} min before transfer")
    if s["amount"] >= LARGE_ABS:
        score += _W["large_abs"]; reasons.append(f"High amount (₹{s['amount']:,.0f})")
    if s["amount_fraction"] >= LARGE_REL:
        score += _W["large_rel"]; reasons.append(f"Amount is {round(100*s['amount_fraction'])}% of balance")
    level, action = _verdict(score)
    if not reasons:
        reasons = ["No correlated risk signals detected"]
    return score, level, reasons, action
