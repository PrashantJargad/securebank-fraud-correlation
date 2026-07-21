"""
Telemetry capture.

Everything the correlation engine relies on is captured here, at the moment
the action happens. Some of these signals CANNOT be reconstructed later — if the
login didn't grab the device and IP, that data is simply gone — which is why
logging is wired into each action rather than bolted on afterwards.

For a prototype we capture real signals where we honestly can:
  - device_id: a persistent cookie. First time we see it for a user => new device.
                Clearing cookies / opening incognito => a genuinely new device.
  - user_agent, ip: read straight off the request.
  - geo: real IP geolocation needs an external service and won't work on
         localhost, so we expose a small "simulate location" control on login.
         This is honest for a prototype: we're simulating the *signal*, not
         faking a capability we claim to have.
"""

import uuid
from models import db, Event, KnownDevice


def get_or_create_device_id(request, response_cookies):
    """Return the device_id from the cookie, creating one if absent.

    `response_cookies` is a list we append (name, value) to; app.py sets them
    on the outgoing response.
    """
    device_id = request.cookies.get("device_id")
    if not device_id:
        device_id = uuid.uuid4().hex
        response_cookies.append(("device_id", device_id))
    return device_id


def is_new_device_for_user(user_id, device_id):
    seen = KnownDevice.query.filter_by(user_id=user_id, device_id=device_id).first()
    return seen is None


def remember_device(user_id, device_id):
    if is_new_device_for_user(user_id, device_id):
        db.session.add(KnownDevice(user_id=user_id, device_id=device_id))
        db.session.commit()


def resolve_geo(ip, simulated_location, home_city):
    """Return (geo_label, is_unusual_location).

    In production this is an IP-geolocation lookup. Here we let the login form
    supply a simulated location so the demo can show a 'foreign login'.
    """
    if simulated_location:
        geo = simulated_location
    elif ip in ("127.0.0.1", "localhost", None, "::1"):
        geo = "Local Network"
    else:
        geo = "Unknown"

    unusual = geo not in (home_city, "Local Network")
    return geo, unusual


def log_event(**kwargs):
    """Insert one row into the unified event log and return it."""
    event = Event(**kwargs)
    db.session.add(event)
    db.session.commit()
    return event
