"""
Automated breach monitoring.

Real-world flow: a third party (say Zomato) is breached and the leaked records
circulate. A breach-intelligence provider (Have I Been Pwned, SpyCloud, card
networks) ingests them. The bank periodically cross-references its own customers
against that intelligence and flags anyone affected — because leaked credentials
are how credential-stuffing account takeovers begin.

TWO THINGS THIS PROTOTYPE DELIBERATELY GETS RIGHT:

1) We never handle the raw stolen dump. We compare *hashes*, never plaintext
   emails/phones. In production you'd use a provider's k-anonymity API (send only
   a hash prefix, compare candidates locally) so the actual value never leaves
   your system. The hashing here stands in for that.

2) No CVV, no raw card data. PCI-DSS forbids a bank from storing CVV at all, so
   there's nothing to match on there. We match on email and phone (and, in
   production, tokenised/hashed card numbers).

This is automation — a lookup/join — not machine learning. The ML is the fraud
model in ml.py, which then WEIGHS the "credentials_breached" signal this produces.
"""

import hashlib
from datetime import datetime

from models import db, User, BreachIntel


def _hash(value):
    """Stand-in for the provider-side hashed comparison (never compare plaintext)."""
    if not value:
        return None
    return hashlib.sha256(value.strip().lower().encode()).hexdigest()


# Simulated external breach feeds. In production these arrive from a breach-intel
# provider as hashed records; here we hash them on load so the matcher only ever
# sees hashes. The plaintext emails below are fictional demo data.
_RAW_FEEDS = [
    {
        "name": "Zomato",
        "date": "2024-08",
        "leaked_fields": "email, phone, hashed password",
        "emails": [
            "rahul@example.com",          # <- a real bank customer
            "someoneelse@gmail.com",
            "foodie2021@yahoo.com",
            "bulk.user@hotmail.com",
        ],
        "phones": ["9990000001"],
    },
    {
        "name": "QuickCart",
        "date": "2023-11",
        "leaked_fields": "email, name, address",
        "emails": [
            "amit@example.com",           # <- a real bank customer
            "randomshopper@gmail.com",
        ],
        "phones": [],
    },
]

# Precompute hashed feeds once (mimics receiving already-hashed intel).
_FEEDS = [{
    "name": f["name"],
    "date": f["date"],
    "leaked_fields": f["leaked_fields"],
    "email_hashes": {_hash(e) for e in f["emails"]},
    "phone_hashes": {_hash(p) for p in f["phones"]},
} for f in _RAW_FEEDS]


def scan_breaches():
    """Cross-reference every bank user against every breach feed, by hash.

    For each match, upsert a BreachIntel watchlist entry. Idempotent — running it
    again won't create duplicates. Returns a per-breach report for display.
    """
    users = User.query.all()
    report = []

    for feed in _FEEDS:
        matched = []
        for u in users:
            if (_hash(u.email) in feed["email_hashes"]
                    or _hash(u.phone) in feed["phone_hashes"]):
                matched.append(u.username)
                source = f"{feed['name']} breach ({feed['date']}) — leaked: {feed['leaked_fields']}"
                exists = BreachIntel.query.filter_by(
                    identifier=u.username, source=source).first()
                if not exists:
                    db.session.add(BreachIntel(identifier=u.username, source=source))
        report.append({
            "name": feed["name"],
            "date": feed["date"],
            "leaked_fields": feed["leaked_fields"],
            "records_checked": len(feed["email_hashes"]) + len(feed["phone_hashes"]),
            "matched_users": matched,
        })

    db.session.commit()
    return report
