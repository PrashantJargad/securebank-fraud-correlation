"""
Database models.

The heart of this project is the `Event` table. Every action the user takes —
logging in, checking a balance, adding a beneficiary, transferring funds — writes
one row here, tagged with a shared set of join keys (user_id, session_id,
device_id, timestamp). The correlation engine never touches the banking tables
directly; it reads this one event log. Get this schema right and everything
downstream is easy.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    home_city = db.Column(db.String(64), default="Jaipur")
    balance = db.Column(db.Float, default=0.0)
    account_number = db.Column(db.String(32), unique=True)
    email = db.Column(db.String(120))
    phone = db.Column(db.String(20))

    beneficiaries = db.relationship("Beneficiary", backref="owner", lazy=True)


class Beneficiary(db.Model):
    __tablename__ = "beneficiaries"
    id = db.Column(db.Integer, primary_key=True)
    owner_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    name = db.Column(db.String(120), nullable=False)
    account_number = db.Column(db.String(32), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class KnownDevice(db.Model):
    """Devices we've seen a given user log in from before.

    A login from a device_id not in this table (for that user) is a
    'new device' event — a mild but real security signal.
    """
    __tablename__ = "known_devices"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    device_id = db.Column(db.String(64), nullable=False)
    first_seen = db.Column(db.DateTime, default=datetime.utcnow)


class Event(db.Model):
    """The unified telemetry log — security signals and transaction signals
    in one place, sharing join keys so they can be correlated."""
    __tablename__ = "events"
    id = db.Column(db.Integer, primary_key=True)
    ts = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    # ---- shared join keys ----
    user_id = db.Column(db.Integer, index=True)
    username = db.Column(db.String(64))
    session_id = db.Column(db.String(64), index=True)
    device_id = db.Column(db.String(64), index=True)

    # ---- security-side telemetry ----
    action = db.Column(db.String(32))          # login_success, login_fail, balance_check, add_beneficiary, transfer
    is_new_device = db.Column(db.Boolean, default=False)
    ip = db.Column(db.String(64))
    geo = db.Column(db.String(64))             # derived / simulated location
    is_unusual_location = db.Column(db.Boolean, default=False)
    used_vpn = db.Column(db.Boolean, default=False)
    impossible_travel = db.Column(db.Boolean, default=False)
    credentials_breached = db.Column(db.Boolean, default=False)
    user_agent = db.Column(db.String(256))

    # ---- transaction-side telemetry ----
    amount = db.Column(db.Float)
    beneficiary_id = db.Column(db.Integer)
    beneficiary_name = db.Column(db.String(120))
    is_new_beneficiary = db.Column(db.Boolean, default=False)
    seconds_since_beneficiary_added = db.Column(db.Integer)

    # ---- correlation output (filled in by the engine for scored events) ----
    risk_score = db.Column(db.Integer)
    risk_level = db.Column(db.String(16))      # LOW / MEDIUM / HIGH
    risk_reasons = db.Column(db.Text)          # human-readable, pipe-separated
    action_taken = db.Column(db.String(32))    # cleared / flagged / blocked

    def to_dict(self):
        return {
            "id": self.id,
            "ts": self.ts.strftime("%Y-%m-%d %H:%M:%S") if self.ts else None,
            "username": self.username,
            "session_id": self.session_id,
            "device_id": self.device_id,
            "action": self.action,
            "is_new_device": self.is_new_device,
            "ip": self.ip,
            "geo": self.geo,
            "is_unusual_location": self.is_unusual_location,
            "amount": self.amount,
            "beneficiary_name": self.beneficiary_name,
            "is_new_beneficiary": self.is_new_beneficiary,
            "seconds_since_beneficiary_added": self.seconds_since_beneficiary_added,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "risk_reasons": self.risk_reasons,
            "action_taken": self.action_taken,
        }


class BreachIntel(db.Model):
    """The bank's breach-intelligence watchlist.

    In production this is populated from external feeds (Have I Been Pwned, card
    networks, threat-intel providers) and lists identifiers — usernames, account
    numbers, or card numbers — known to have leaked in a data breach. A login by
    someone on this list is a credential-stuffing / takeover risk.
    """
    __tablename__ = "breach_intel"
    id = db.Column(db.Integer, primary_key=True)
    identifier = db.Column(db.String(64), index=True)   # username / account / card
    source = db.Column(db.String(160))                  # where it leaked
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
