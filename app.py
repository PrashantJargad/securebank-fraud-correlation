"""
FINSPARK prototype — AI-Driven Correlation of Cybersecurity Telemetry &
Transactional Behaviour.

The banking app is intentionally thin. Its real job is to GENERATE telemetry:
every action writes one row to the unified event log (see models.Event), and the
correlation engine (correlation.py) reads that log to catch fraud that no single
signal would reveal on its own.
"""

import os
import re
import uuid
from datetime import datetime, timedelta

from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, g)
from werkzeug.security import check_password_hash, generate_password_hash

from models import db, User, Beneficiary, Event, BreachIntel
from telemetry import (is_new_device_for_user, remember_device,
                       resolve_geo, log_event)
from correlation import score_transfer

app = Flask(__name__)
app.config["SECRET_KEY"] = "finspark-prototype-secret-change-me"

# Absolute path so the DB is the SAME file no matter which directory the app or
# init script is launched from. (Relative sqlite:/// paths resolve against the
# instance folder / CWD and can silently diverge.)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "banking.db")
db.init_app(app)

NEW_BENEFICIARY_WINDOW = 86400  # a beneficiary younger than 24h counts as "new"


# ---------------------------------------------------------------------------
# Device cookie handling: a persistent device_id so "new device" is meaningful.
# ---------------------------------------------------------------------------
@app.before_request
def ensure_device_id():
    device_id = request.cookies.get("device_id")
    if not device_id:
        device_id = uuid.uuid4().hex
        g.set_device_cookie = device_id
    g.device_id = device_id


@app.after_request
def persist_device_cookie(response):
    new_cookie = g.pop("set_device_cookie", None)
    if new_cookie:
        response.set_cookie("device_id", new_cookie,
                            max_age=60 * 60 * 24 * 365,
                            httponly=True, samesite="Lax")
    return response


def current_user():
    uid = session.get("user_id")
    return User.query.get(uid) if uid else None


def _digits(s):
    return re.sub(r"\D", "", s or "")


# Approximate coordinates for the demo cities, used to detect "impossible travel"
# between two consecutive logins. Locations without coordinates (e.g. "Unknown
# location", "Local Network") can't be measured, so travel is not flagged for them.
CITY_COORDS = {
    "Jaipur": (26.9124, 75.7873),
    "Mumbai": (19.0760, 72.8777),
    "Delhi":  (28.7041, 77.1025),
}
MAX_PLAUSIBLE_KMH = 900          # faster than a commercial flight => impossible
SAME_CITY_KM = 50                # ignore tiny/zero distances


def _haversine_km(a, b):
    from math import radians, sin, cos, asin, sqrt
    lat1, lon1 = a
    lat2, lon2 = b
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    h = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(h))


def is_impossible_travel(prev_geo, prev_ts, cur_geo, now):
    """True if getting from prev_geo to cur_geo in the elapsed time would require
    travelling faster than a flight."""
    if prev_geo not in CITY_COORDS or cur_geo not in CITY_COORDS:
        return False
    distance = _haversine_km(CITY_COORDS[prev_geo], CITY_COORDS[cur_geo])
    if distance < SAME_CITY_KM:
        return False
    hours = (now - prev_ts).total_seconds() / 3600.0
    if hours <= 0:
        return True
    return (distance / hours) > MAX_PLAUSIBLE_KMH


def find_account(account_number):
    """Find the user whose account matches, comparing digits only so small
    formatting differences (spaces, an 'AC' prefix) still resolve correctly."""
    target = _digits(account_number)
    if not target:
        return None
    for u in User.query.all():
        if _digits(u.account_number) == target:
            return u
    return None


def breach_record(user):
    """Return the breach-intel record if this user's username or account number
    is on the watchlist, else None."""
    acc = _digits(user.account_number)
    for rec in BreachIntel.query.all():
        ident = rec.identifier or ""
        if ident.lower() == user.username.lower() or _digits(ident) == acc:
            return rec
    return None


def require_login():
    if not session.get("user_id"):
        return redirect(url_for("login"))
    return None


# ---------------------------------------------------------------------------
# Login  (security telemetry is captured here — the richest signal source)
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def login():
    # A session id is minted before authentication so that failed attempts
    # correlate with the eventual successful login.
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        simulated_location = request.form.get("location", "").strip()

        user = User.query.filter_by(username=username).first()
        ip = request.remote_addr
        ua = request.headers.get("User-Agent", "")[:255]

        if not user or not check_password_hash(user.password_hash, password):
            log_event(
                user_id=user.id if user else None,
                username=username or "(unknown)",
                session_id=session["sid"],
                device_id=g.device_id,
                action="login_fail",
                ip=ip, user_agent=ua,
            )
            flash("Invalid username or password.", "error")
            return redirect(url_for("login"))

        # Successful login — compute the security signals BEFORE remembering it.
        new_device = is_new_device_for_user(user.id, g.device_id)
        geo, unusual = resolve_geo(ip, simulated_location, user.home_city)
        used_vpn = bool(request.form.get("used_vpn"))

        # Impossible travel: compare against this user's PREVIOUS login. If the
        # distance between the two cities couldn't be covered in the elapsed
        # time (faster than a flight), the two logins can't be the same person.
        prev = (Event.query
                .filter_by(user_id=user.id, action="login_success")
                .order_by(Event.ts.desc()).first())
        impossible = False
        if prev and prev.geo:
            impossible = is_impossible_travel(prev.geo, prev.ts, geo, datetime.utcnow())

        breached = breach_record(user) is not None

        log_event(
            user_id=user.id,
            username=user.username,
            session_id=session["sid"],
            device_id=g.device_id,
            action="login_success",
            is_new_device=new_device,
            ip=ip, geo=geo, is_unusual_location=unusual,
            used_vpn=used_vpn, impossible_travel=impossible,
            credentials_breached=breached,
            user_agent=ua,
        )
        remember_device(user.id, g.device_id)

        session["user_id"] = user.id
        session["username"] = user.username
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Balance check  (logged — attackers recon the balance before draining)
# ---------------------------------------------------------------------------
@app.route("/dashboard")
def dashboard():
    guard = require_login()
    if guard:
        return guard
    user = current_user()

    log_event(
        user_id=user.id, username=user.username,
        session_id=session["sid"], device_id=g.device_id,
        action="balance_check",
    )
    return render_template("dashboard.html", user=user, breach=breach_record(user))


# ---------------------------------------------------------------------------
# Beneficiaries  (add / list)
# ---------------------------------------------------------------------------
@app.route("/beneficiaries", methods=["GET", "POST"])
def beneficiaries():
    guard = require_login()
    if guard:
        return guard
    user = current_user()

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        account_number = request.form.get("account_number", "").strip()
        if not name or not account_number:
            flash("Both name and account number are required.", "error")
            return redirect(url_for("beneficiaries"))

        ben = Beneficiary(owner_id=user.id, name=name,
                          account_number=account_number,
                          created_at=datetime.utcnow())
        db.session.add(ben)
        db.session.commit()

        log_event(
            user_id=user.id, username=user.username,
            session_id=session["sid"], device_id=g.device_id,
            action="add_beneficiary",
            beneficiary_id=ben.id, beneficiary_name=ben.name,
        )
        flash(f"Beneficiary '{name}' added.", "ok")
        return redirect(url_for("beneficiaries"))

    bens = Beneficiary.query.filter_by(owner_id=user.id).order_by(
        Beneficiary.created_at.desc()).all()
    return render_template("beneficiaries.html", user=user, beneficiaries=bens)


# ---------------------------------------------------------------------------
# Fund transfer  (the high-value action; correlation runs here)
# ---------------------------------------------------------------------------
@app.route("/transfer", methods=["GET", "POST"])
def transfer():
    guard = require_login()
    if guard:
        return guard
    user = current_user()
    bens = Beneficiary.query.filter_by(owner_id=user.id).order_by(
        Beneficiary.created_at.desc()).all()

    if request.method == "POST":
        try:
            amount = float(request.form.get("amount", "0"))
        except ValueError:
            amount = 0.0
        ben_id = request.form.get("beneficiary_id", type=int)
        ben = Beneficiary.query.filter_by(id=ben_id, owner_id=user.id).first()

        if not ben or amount <= 0:
            flash("Pick a beneficiary and a valid amount.", "error")
            return redirect(url_for("transfer"))

        # transaction-side signals
        age_seconds = int((datetime.utcnow() - ben.created_at).total_seconds())
        is_new_ben = age_seconds <= NEW_BENEFICIARY_WINDOW
        balance_before = user.balance

        # Which real account (if any) does this payee point at? Matched on the
        # digits only, so "1002", "AC1002" and "1002 " all resolve the same.
        recipient = find_account(ben.account_number)

        # ---- run the correlation engine ----
        score, level, reasons, taken = score_transfer(
            session_id=session["sid"],
            amount=amount,
            balance_before=balance_before,
            is_new_beneficiary=is_new_ben,
            seconds_since_beneficiary_added=age_seconds,
        )
        reasons_text = " | ".join(reasons)

        executed = False
        credited = False
        if taken != "blocked" and amount <= balance_before:
            user.balance = balance_before - amount
            if recipient and recipient.id != user.id:
                recipient.balance += amount
                credited = True
            db.session.commit()
            executed = True
        elif amount > balance_before:
            taken = "blocked"
            reasons_text = "Insufficient balance | " + reasons_text

        log_event(
            user_id=user.id, username=user.username,
            session_id=session["sid"], device_id=g.device_id,
            action="transfer",
            amount=amount,
            beneficiary_id=ben.id, beneficiary_name=ben.name,
            is_new_beneficiary=is_new_ben,
            seconds_since_beneficiary_added=age_seconds,
            risk_score=score, risk_level=level,
            risk_reasons=reasons_text, action_taken=taken,
        )

        return render_template("transfer_result.html", user=user, ben=ben,
                               amount=amount, score=score, level=level,
                               reasons=reasons, taken=taken, executed=executed,
                               recipient=recipient, credited=credited)

    return render_template("transfer.html", user=user, beneficiaries=bens)


# ---------------------------------------------------------------------------
# SOC / analyst view  (read-only window onto the fused telemetry)
# ---------------------------------------------------------------------------
@app.route("/soc")
def soc():
    events = Event.query.order_by(Event.ts.desc()).limit(200).all()
    return render_template("soc.html", events=events)


@app.route("ty-scan")
def breach_scan():
    """Run the automated breach monitor and show which customers were found in
    external breaches (and are now on the watchlist)."""
    from breach_monitor import scan_breaches
    report = scan_breaches()
    watchlist = BreachIntel.query.order_by(BreachIntel.identifier).all()
    return render_template("breach_scan.html", report=report, watchlist=watchlist)


def seed_demo_data():
    """Seed several login accounts, each with its own balance and beneficiaries.

    Account numbers are shared where it makes sense: rahul's 'Landlord' payee
    points at the real 'landlord' account, so paying rent actually moves money —
    log in as landlord afterwards and the balance has gone up.
    """
    if User.query.first() is not None:
        return

    # username, password, account_number, home_city, balance, email, phone
    people = [
        ("rahul",    "password123", "1001", "Jaipur", 100000.0, "rahul@example.com",    "9990000001"),
        ("priya",    "password123", "1002", "Mumbai",  50000.0, "priya@example.com",    "9990000002"),
        ("amit",     "password123", "1003", "Delhi",   75000.0, "amit@example.com",     "9990000003"),
        ("landlord", "password123", "1004", "Jaipur",  20000.0, "landlord@example.com", "9990000004"),
    ]
    users = {}
    for username, pw, acc, city, bal, email, phone in people:
        u = User(username=username, password_hash=generate_password_hash(pw),
                 account_number=acc, home_city=city, balance=bal,
                 email=email, phone=phone)
        db.session.add(u)
        users[username] = u
    db.session.commit()

    # A couple of pre-existing, trusted payees (added days ago => not "new").
    db.session.add(Beneficiary(
        owner_id=users["rahul"].id, name="Landlord",
        account_number=users["landlord"].account_number,
        created_at=datetime.utcnow() - timedelta(days=5)))
    db.session.add(Beneficiary(
        owner_id=users["priya"].id, name="Amit",
        account_number=users["amit"].account_number,
        created_at=datetime.utcnow() - timedelta(days=10)))
    db.session.commit()


def bootstrap():
    """Create tables if missing and seed demo accounts if the DB is empty.

    This makes `python app.py` work on its own — no separate init step required.
    Use init_db.py only when you want to wipe and reset.
    """
    db.create_all()
    seed_demo_data()
    # Run the automated breach monitor so the watchlist reflects current
    # breach intelligence from the moment the app starts.
    from breach_monitor import scan_breaches
    scan_breaches()


with app.app_context():
    bootstrap()
    try:
        import ml
        ml.ensure_model()          # train the fraud model once if it's not there yet
    except Exception as exc:       # never let this crash the app
        print(f"[warn] ML model unavailable ({exc}); using rule-based fallback. "
              f"Run `pip install -r requirements.txt` then `python ml.py`.")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
