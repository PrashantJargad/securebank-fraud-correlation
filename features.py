"""
Feature definitions shared by the data generator, the trainer, and the scorer.

Keeping this in one place guarantees the model is trained on exactly the same
features (in the same order) that the live app extracts at scoring time. If these
ever drift apart, the model silently gets garbage — so there is ONE source of
truth here.

Each session is reduced to a fixed row of numbers:

    new_device            0/1  did the session start on an unrecognised device
    unusual_location      0/1  was the login location unusual for this user
    failed_logins         int  failed attempts before success this session
    is_new_beneficiary    0/1  payee added within the last 24h
    beneficiary_age_min   float  minutes between adding the payee and paying it
    amount                float  transfer amount (rupees)
    amount_fraction       float  amount / balance before the transfer
"""

import numpy as np

FEATURES = [
    "new_device",
    "unusual_location",
    "impossible_travel",
    "used_vpn",
    "credentials_breached",
    "failed_logins",
    "beneficiary_age_min",
    "amount",
    "amount_fraction",
]

# The "nothing suspicious" value for each feature. Used for two things:
#  1) as the neutral reference when explaining a prediction (see ml.explain), and
#  2) conceptually, what a totally clean session looks like.
SAFE_BASELINE = {
    "new_device": 0,
    "unusual_location": 0,
    "impossible_travel": 0,
    "used_vpn": 0,
    "credentials_breached": 0,
    "failed_logins": 0,
    "beneficiary_age_min": 43200.0,   # 30 days -> effectively an old, trusted payee
    "amount": 1000.0,
    "amount_fraction": 0.01,
}

NEW_BENEFICIARY_WINDOW_MIN = 1440      # 24h, matches the app's definition of "new"


def signals_to_vector(signals):
    """Turn a dict of raw session signals into the ordered numeric row."""
    return np.array([float(signals[f]) for f in FEATURES], dtype=float)


def safe_vector():
    return np.array([float(SAFE_BASELINE[f]) for f in FEATURES], dtype=float)


def human_reason(feature, signals):
    """A plain-English reason for a feature, using the session's actual values."""
    if feature == "new_device":
        return "Login from a new, unrecognised device"
    if feature == "unusual_location":
        return "Login from an unusual location for this user"
    if feature == "impossible_travel":
        return "Login location changed faster than physically possible (impossible travel)"
    if feature == "used_vpn":
        return "Login came through a VPN / anonymising proxy"
    if feature == "credentials_breached":
        return "Account credentials appear in a known data breach"
    if feature == "failed_logins":
        n = int(signals["failed_logins"])
        return f"{n} failed login attempt(s) earlier this session"
    if feature == "is_new_beneficiary":
        return "Transfer to a recently added beneficiary"
    if feature == "beneficiary_age_min":
        mins = round(float(signals["beneficiary_age_min"]), 1)
        return f"Beneficiary was added only {mins} min before this transfer"
    if feature == "amount":
        return f"High transfer amount (₹{float(signals['amount']):,.0f})"
    if feature == "amount_fraction":
        pct = round(100 * float(signals["amount_fraction"]))
        return f"Amount is {pct}% of the available balance"
    return feature
