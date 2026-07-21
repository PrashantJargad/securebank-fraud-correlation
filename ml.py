"""
The AI (Route 1: supervised learning).

Because a prototype has no real, labelled banking-fraud history, we BOOTSTRAP the
training data: `generate_dataset` produces thousands of synthetic sessions — some
scripted to resemble ordinary customers, some to resemble account takeovers —
each stamped with its true label (0 = legit, 1 = fraud). The distributions
deliberately OVERLAP (some legit users are on new devices; some fraud is
deliberately subtle), so the model has to learn which *combinations* of signals
predict fraud rather than memorising a clean rule.

`train_and_save` then fits a gradient-boosted tree on those examples. In
production you retrain the exact same pipeline on the bank's real labelled cases —
the synthetic data just proves the pipeline end to end.

Explanations come from `explain`: we measure how far the fraud probability drops
when each active signal is reset to its safe baseline. The signals that drop it
most are the reasons — model-agnostic, exact, and needs no extra libraries.
"""

import os
import numpy as np

from features import (FEATURES, signals_to_vector, safe_vector,
                      human_reason, NEW_BENEFICIARY_WINDOW_MIN)

MODEL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                          "fraud_model.joblib")


# ---------------------------------------------------------------------------
# 1) Synthetic, labelled training data
# ---------------------------------------------------------------------------
def _legit_session(rng):
    # Most legit logins are from a known device in a normal place.
    new_device = int(rng.random() < 0.15)          # sometimes a new phone / travel
    unusual_location = int(rng.random() < 0.10)
    impossible_travel = int(rng.random() < 0.02)    # very rare (clock skew etc.)
    used_vpn = int(rng.random() < 0.15)             # plenty of legit users use a VPN
    credentials_breached = int(rng.random() < 0.10)  # some ordinary users are in a breach
    failed_logins = int(rng.choice([0, 1, 2], p=[0.90, 0.08, 0.02]))

    # 60% pay an existing/old payee, 40% a recently added one — and plenty of
    # those legit new payees are paid within minutes (a new merchant, a friend).
    # So "fresh beneficiary" on its own must NOT scream fraud.
    if rng.random() < 0.60:
        ben_age = rng.uniform(NEW_BENEFICIARY_WINDOW_MIN, 43200)   # 1–30 days
    else:
        ben_age = rng.uniform(0.5, NEW_BENEFICIARY_WINDOW_MIN)     # minutes–1 day

    balance = rng.uniform(20000, 200000)
    if rng.random() < 0.80:
        frac = min(abs(rng.normal(0.08, 0.10)), 0.5)               # small, routine
    else:
        frac = rng.uniform(0.30, 0.70)                             # occasional big (rent)
    amount = frac * balance
    return _row(new_device, unusual_location, impossible_travel, used_vpn,
                credentials_breached, failed_logins, ben_age, amount, balance)


def _fraud_session(rng):
    # Takeovers usually arrive on a new device from an odd place, drain to a
    # freshly added mule account, and move a big share of the balance — but a
    # patient attacker may pre-stage an older mule, so age alone can't carry it.
    new_device = int(rng.random() < 0.85)
    unusual_location = int(rng.random() < 0.75)
    impossible_travel = int(rng.random() < 0.45)    # victim here, attacker far away
    used_vpn = int(rng.random() < 0.55)             # attackers often hide behind a VPN
    credentials_breached = int(rng.random() < 0.65)  # takeovers usually start from leaked creds
    failed_logins = int(rng.choice([0, 1, 2, 3], p=[0.40, 0.35, 0.20, 0.05]))

    r = rng.random()
    if r < 0.50:
        ben_age = rng.uniform(0.1, 60)             # added minutes ago
    elif r < 0.80:
        ben_age = rng.uniform(60, 1440)            # hours ago
    else:
        ben_age = rng.uniform(1440, 5760)          # pre-staged 1–4 days ago

    balance = rng.uniform(20000, 200000)
    if rng.random() < 0.80:
        frac = rng.uniform(0.40, 1.0)              # drain
    else:
        frac = rng.uniform(0.05, 0.30)             # ...but some stay subtle
    amount = frac * balance
    return _row(new_device, unusual_location, impossible_travel, used_vpn,
                credentials_breached, failed_logins, ben_age, amount, balance)


def _row(new_device, unusual_location, impossible_travel, used_vpn,
         credentials_breached, failed_logins, ben_age, amount, balance):
    frac = amount / balance if balance > 0 else 0.0
    return {
        "new_device": new_device,
        "unusual_location": unusual_location,
        "impossible_travel": impossible_travel,
        "used_vpn": used_vpn,
        "credentials_breached": credentials_breached,
        "failed_logins": failed_logins,
        "beneficiary_age_min": ben_age,
        "amount": amount,
        "amount_fraction": frac,
    }


def generate_dataset(n_per_class=3000, seed=42):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for _ in range(n_per_class):
        rows.append(signals_to_vector(_legit_session(rng)));  labels.append(0)
        rows.append(signals_to_vector(_fraud_session(rng)));  labels.append(1)
    return np.array(rows), np.array(labels)


# ---------------------------------------------------------------------------
# 2) Train / save / load
# ---------------------------------------------------------------------------
def train_and_save(path=MODEL_PATH, verbose=False):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score
    import joblib

    X, y = generate_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              random_state=42, stratify=y)
    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_tr, y_tr)

    if verbose:
        p = model.predict_proba(X_te)[:, 1]
        print(f"  training samples : {len(X_tr)}")
        print(f"  test accuracy    : {accuracy_score(y_te, p >= 0.5):.3f}")
        print(f"  test ROC AUC     : {roc_auc_score(y_te, p):.3f}")
        print("  feature importance:")
        for f, imp in sorted(zip(FEATURES, model.feature_importances_),
                             key=lambda t: -t[1]):
            print(f"    {f:22s} {imp:.3f}")

    joblib.dump(model, path)
    return model


def load_model(path=MODEL_PATH):
    import joblib
    return joblib.load(path)


def ensure_model(path=MODEL_PATH):
    """Train the model once if it isn't there yet (mirrors the DB bootstrap)."""
    if not os.path.exists(path):
        train_and_save(path)


# ---------------------------------------------------------------------------
# 3) Scoring + explanation
# ---------------------------------------------------------------------------
def predict_proba(model, signals):
    x = signals_to_vector(signals).reshape(1, -1)
    return float(model.predict_proba(x)[0, 1])


def explain(model, signals, min_contribution=0.01, top_k=6):
    """Return reasons ordered by each signal's contribution to the fraud score,
    computed with exact Shapley values.

    Shapley attribution asks, for each signal, how much it raises the score
    *averaged over every combination of the other signals*. That fairly splits
    credit when several signals are individually redundant — so a session with a
    new device AND an odd location AND a fresh payee names all three, instead of
    crediting only whichever one happened to tip it over the edge.

    With only a handful of features we can enumerate all subsets exactly rather
    than approximating.
    """
    import math
    from itertools import combinations

    x = signals_to_vector(signals)
    base = safe_vector()
    n = len(FEATURES)

    # Build every subset of features (those "present" take their real value,
    # the rest sit at the safe baseline) and score them all in one batch.
    subsets = [frozenset(c) for r in range(n + 1) for c in combinations(range(n), r)]
    rows = []
    for S in subsets:
        v = base.copy()
        for i in S:
            v[i] = x[i]
        rows.append(v)
    probs = model.predict_proba(np.array(rows))[:, 1]
    f = {S: p for S, p in zip(subsets, probs)}

    shap = np.zeros(n)
    for i in range(n):
        for S in subsets:
            if i in S:
                continue
            s = len(S)
            weight = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
            shap[i] += weight * (f[S | {i}] - f[S])

    contribs = sorted(((shap[i], FEATURES[i]) for i in range(n)
                       if shap[i] >= min_contribution), reverse=True)
    reasons = [human_reason(feat, signals) for _, feat in contribs[:top_k]]
    if not reasons:
        reasons = ["No significant risk signals detected by the model"]
    return reasons


if __name__ == "__main__":
    print("Training fraud model on synthetic sessions...")
    train_and_save(verbose=True)
    print(f"Saved model -> {MODEL_PATH}")
