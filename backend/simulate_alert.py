#!/usr/bin/env python3
# Simulate suspicious login: check model, send alert, set must_change_backup_pin.
import argparse
import asyncio
from datetime import datetime, timezone
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from anomaly import load_model, check_anomaly

# NOTE: script imports DB/SMTP values from main.py to avoid duplication.
# Try several import strategies to support running from repo root or from /backend
try:
    # preferred when running from repo root: import backend package
    import backend.main as mainmod
except Exception:
    try:
        # when running the script from within the backend folder
        import main as mainmod
    except Exception:
        # fallback: add parent directory to sys.path and try again
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        import backend.main as mainmod

DATABASE_URL = getattr(mainmod, "DATABASE_URL")
SMTP_SERVER = getattr(mainmod, "SMTP_SERVER")
SMTP_PORT = getattr(mainmod, "SMTP_PORT")
SMTP_USERNAME = getattr(mainmod, "SMTP_USERNAME")
SMTP_PASSWORD = getattr(mainmod, "SMTP_PASSWORD")
APP_NAME = getattr(mainmod, "APP_NAME", "NoteVault")

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

async def mark_user_force_stepup(engine, email):
    async with engine.begin() as conn:
        await conn.execute(text(
            "UPDATE users SET must_change_backup_pin = true WHERE email = :email"
        ), {"email": email})
        await conn.commit()

def send_alert_email(to_email):
    msg = MIMEMultipart()
    msg["From"] = f"{APP_NAME} Security <{SMTP_USERNAME}>"
    msg["To"] = to_email
    msg["Subject"] = f"[{APP_NAME}] Suspicious login detected"
    body = "<p>We detected a suspicious login to your account. Please review activity and change your backup PIN.</p>"
    msg.attach(MIMEText(body, "html"))
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USERNAME, SMTP_PASSWORD)
        s.send_message(msg)

async def run(email, hour, success):
    model = load_model()
    if model is None:
        print("No anomaly model found (backend/anomaly_model.pkl). Train one first.")
        return
    dt = datetime.now(timezone.utc).replace(hour=int(hour), minute=0, second=0, microsecond=0)
    suspicious = check_anomaly(model, dt, bool(success))
    print("Anomaly check:", suspicious)
    if suspicious:
        # send email
        try:
            send_alert_email(email)
            print("Alert email sent to", email)
        except Exception as e:
            print("Failed sending email:", e)
        # set DB flag
        engine = create_async_engine(DATABASE_URL, echo=False)
        await mark_user_force_stepup(engine, email)
        await engine.dispose()
        print("User flagged for forced step-up (must_change_backup_pin = true).")
    else:
        print("Not suspicious according to model.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--email", required=True)
    p.add_argument("--hour", type=int, default=3, help="Hour to simulate (0-23)")
    p.add_argument("--success", type=int, default=1, help="1 for success, 0 for failed")
    args = p.parse_args()
    asyncio.run(run(args.email, args.hour, args.success))