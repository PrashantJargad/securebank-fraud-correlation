"""
Wipe and rebuild the database with the demo accounts.

Run this whenever you want a clean reset:   python init_db.py
(The app also seeds itself automatically on first launch, so this is only for
resetting an existing database.)

Demo logins (all password: password123):
    rahul     — Jaipur, balance 100000, trusts payee "Landlord"
    priya     — Mumbai, balance  50000, trusts payee "Amit"
    amit      — Delhi,  balance  75000
    landlord  — Jaipur, balance  20000   (receives rahul's rent transfers)
"""

from app import app, seed_demo_data
from models import db


def seed():
    with app.app_context():
        db.drop_all()
        db.create_all()
        seed_demo_data()
        print("Database reset. Demo logins (password: password123):")
        print("  rahul, priya, amit, landlord")


if __name__ == "__main__":
    seed()
