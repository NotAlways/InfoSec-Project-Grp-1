from fastapi import FastAPI, HTTPException, Depends, Form, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, select, update, delete
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import LargeBinary
from pydantic import BaseModel
from datetime import datetime
import pyotp
import qrcode
import io
import base64
import re
import uuid
from datetime import datetime, timedelta, timezone
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from passlib.context import CryptContext
from itsdangerous import URLSafeTimedSerializer
from backend.crypto import (
    load_key,
    encrypt_content,
    decrypt_content,
    generate_key,
    save_key,
    key_exists,
    wrap_key,
    unwrap_key,
)
from backend.activity import init_migration, log_activity
from backend.anomaly import check_anomaly, load_model
from pathlib import Path
from fastapi import Cookie
from typing import Optional
import asyncio
import secrets
import random
from typing import Union
import traceback
import time
import hashlib
import math
import json
import numpy as np
from webauthn import (
    generate_registration_options,
    verify_registration_response,
    generate_authentication_options,
    verify_authentication_response,
    options_to_json,
    base64url_to_bytes,
)
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    AuthenticatorAttachment,
    ResidentKeyRequirement,
    UserVerificationRequirement,
    PublicKeyCredentialDescriptor,
    AttestationConveyancePreference,
)
from webauthn.helpers import bytes_to_base64url
import os
import base64
from fastapi.responses import JSONResponse
from urllib.parse import quote
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sqlalchemy.ext.mutable import MutableList
from io import BytesIO
from fastapi.responses import StreamingResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch


limiter = Limiter(key_func=get_remote_address)


# --- IMPORTS FOR EMAIL ---
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "prathipkumarans@gmail.com"
SMTP_PASSWORD = "mnlr xiis amxy lvxh"
APP_NAME = "NoteVault"

# Database setup, change to your own password here. Make sure PostgreSQL is running. To be encrypted in the future.
# password: 1m1f1b1m
DATABASE_URL = "postgresql+asyncpg://postgres:password@localhost/notevault"
Base = declarative_base()

# Load encryption key on startup
encryption_key = None

def get_encryption_key():
    global encryption_key
    if encryption_key is None:
        encryption_key = load_key()
    return encryption_key


# FastAPI app
app = FastAPI(title="NoteVault API")
app.state.limiter = limiter

def redirect_with_error(path: str, msg: str) -> RedirectResponse:
    return RedirectResponse(url=f"{path}?error={quote(msg)}", status_code=303)

@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return redirect_with_error(
        "/login",
        "Too many requests. Please wait and try again"
    )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database session
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

def send_security_email(subject: str, recipient: str, body_html: str):
    """Sends a secure SMTP email for 2FA/Verification"""
    try:
        msg = MIMEMultipart()
        # FIX: Wrapped in quotes to make it a string
        msg['From'] = f"{APP_NAME} Security <{SMTP_USERNAME}>" 
        msg['To'] = recipient
        msg['Subject'] = f"[{APP_NAME}] {subject}"
        msg.attach(MIMEText(body_html, 'html'))

        # This block uses the constants defined above
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Upgrade connection to secure
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"CRITICAL: Email failed to send to {recipient}. Error: {e}")
        return False

def get_sg_time():
    """Helper function to get current time in GMT+8 (Naive for DB compatibility)"""
    # 1. Get time with TZ
    sg_time = datetime.now(timezone(timedelta(hours=8)))
    # 2. Remove the TZ info so SQLAlchemy doesn't crash
    return sg_time.replace(tzinfo=None)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    password_history = Column(JSON, default=list)
    totp_secret = Column(String)
    backup_pin_hash = Column(String)
    backup_pin_history = Column(MutableList.as_mutable(JSON), default=list)
    backup_pin_changed_at = Column(DateTime, default=get_sg_time)
    must_change_backup_pin = Column(Boolean, default=False)
    role = Column(String, default="user")
    status = Column(String, default="active")
    failed_login_attempts = Column(Integer, default=0)
    failed_2fa_attempts = Column(Integer, default=0) # Track 2FA/PIN separately
    lockout_until = Column(DateTime, nullable=True)  # Progressive timer
    current_session_id = Column(String, nullable=True)
    device_key = Column(String, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=get_sg_time)
    passkeys = Column(JSON, default=list)
    last_activity_at = Column(DateTime, nullable=True)  # ✅ tracks last user activity for auto-logout


# Models
class Note(Base):
    __tablename__ = "notes"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content = Column(String)
    wrapped_key = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class NoteSchema(BaseModel):
    title: str
    content: str

class NoteResponse(BaseModel):
    id: int
    title: str
    content: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

async def verify_session(request: Request, db: AsyncSession = Depends(get_db)):
    session_id = request.cookies.get("session_id")
    if not session_id:
        raise HTTPException(status_code=303)

    res = await db.execute(select(User).where(User.current_session_id == session_id))
    user = res.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=303)

    now = get_sg_time()

    # ✅ A) Inactivity Auto Logout (compute inactive_seconds safely)
    if user.last_activity_at:
        inactive_seconds = (now - user.last_activity_at).total_seconds()
    else:
        # If missing, treat as fresh login (or set it now)
        inactive_seconds = 0

    if inactive_seconds > (INACTIVITY_TIMEOUT_MINUTES * 60):
        # invalidate session server-side
        user.current_session_id = None
        user.last_activity_at = None  # prevent immediate re-expire after login
        await db.commit()
        raise HTTPException(status_code=440, detail="SESSION_EXPIRED")

    # ✅ Update last activity (throttled)
    if (not user.last_activity_at) or ((now - user.last_activity_at).total_seconds() > LAST_ACTIVITY_UPDATE_SECONDS):
        user.last_activity_at = now
        await db.commit()

    # ✅ GLOBAL PIN GATE (your existing logic)
    pin_expired = is_backup_pin_expired(user) or bool(getattr(user, "must_change_backup_pin", False))
    path = request.url.path

    allowed_paths = {
        "/change-backup-pin",
        "/logout",
        "/login",
        "/",
    }

    if path.startswith("/static") or path.startswith("/auth/") or path.startswith("/verify-"):
        return user

    if pin_expired and path not in allowed_paths:
        raise HTTPException(status_code=409, detail="PIN_EXPIRED")

    return user

# Routes
@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
        # Migration: Add wrapped_key column if it doesn't exist
        try:
            from sqlalchemy import text
            await conn.execute(text("""
                ALTER TABLE notes 
                ADD COLUMN IF NOT EXISTS wrapped_key VARCHAR
            """))
            print("✓ Migration: Added wrapped_key column to notes table")
        except Exception as e:
            print(f"Migration note: wrapped_key column may already exist or migration skipped: {e}")
        
        # Migration: Add last_activity_at column to users table
        try:
            from sqlalchemy import text
            await conn.execute(text("""
                ALTER TABLE users 
                ADD COLUMN IF NOT EXISTS last_activity_at TIMESTAMP
            """))
            print("✓ Migration: Added last_activity_at column to users table")
        except Exception as e:
            print(f"Migration note: last_activity_at column may already exist or migration skipped: {e}")

    # Ensure user_activity table exists for activity logging
    try:
        from backend.activity import init_migration, log_activity
        await init_migration(engine)
        print("✓ Migration: user_activity table ready")
    except Exception as e:
        print(f"Migration note: user_activity table migration skipped: {e}")

    # Load anomaly detection model if available
    try:
        from backend.anomaly import load_model
        model = load_model()
        app.state.anomaly_model = model
        if model:
            print("✓ Anomaly model loaded")
        else:
            print("! Anomaly model not found; create and save one using backend/anomaly.py")
    except Exception as e:
        print(f"Anomaly model load skipped: {e}")

@app.post("/notes", response_model=NoteResponse)
async def create_note(note: NoteSchema, request: Request, user: User = Depends(verify_session), db: AsyncSession = Depends(get_db)):
    master_key = get_encryption_key()
    # generate a per-note DEK and encrypt the note with it
    dek = generate_key()
    encrypted_content = encrypt_content(note.content, dek)
    # wrap the DEK with the master key
    wrapped = wrap_key(dek, master_key)
    new_note = Note(title=note.title, content=encrypted_content, wrapped_key=wrapped)
    db.add(new_note)
    await db.commit()
    await db.refresh(new_note)
    return new_note

@app.get("/notes", response_model=list[NoteResponse])
async def get_notes(request: Request, user: User = Depends(verify_session), db: AsyncSession = Depends(get_db),):
    from sqlalchemy import select
    master_key = get_encryption_key()
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Note))
        notes = result.scalars().all()
        # Decrypt content for each note
        for note in notes:
            try:
                if note.wrapped_key:
                    dek = unwrap_key(note.wrapped_key, master_key)
                    note.content = decrypt_content(note.content, dek)
                else:
                    # legacy/plaintext
                    note.content = decrypt_content(note.content, master_key)
            except Exception:
                # if decryption fails, leave content as-is or set an error placeholder
                note.content = "[unable to decrypt]"
        return notes

@app.get("/notes/{note_id}", response_model=NoteResponse)
async def get_note(note_id: int, request: Request, user: User = Depends(verify_session), db: AsyncSession = Depends(get_db),):
    from sqlalchemy import select
    master_key = get_encryption_key()
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Note).filter(Note.id == note_id))
        note = result.scalar_one_or_none()
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")
        try:
            if note.wrapped_key:
                dek = unwrap_key(note.wrapped_key, master_key)
                note.content = decrypt_content(note.content, dek)
            else:
                note.content = decrypt_content(note.content, master_key)
        except Exception:
            note.content = "[unable to decrypt]"
        return note

@app.put("/notes/{note_id}", response_model=NoteResponse)
async def update_note(note_id: int, note: NoteSchema, request: Request, user: User = Depends(verify_session), db: AsyncSession = Depends(get_db),):
    from sqlalchemy import select
    master_key = get_encryption_key()
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Note).filter(Note.id == note_id))
        db_note = result.scalar_one_or_none()
        if not db_note:
            raise HTTPException(status_code=404, detail="Note not found")
        # generate a new per-note DEK for the updated content
        dek = generate_key()
        db_note.title = note.title
        db_note.content = encrypt_content(note.content, dek)
        db_note.wrapped_key = wrap_key(dek, master_key)
        db_note.updated_at = datetime.utcnow()
        await session.commit()
        await session.refresh(db_note)
        # Decrypt for response
        try:
            dek = unwrap_key(db_note.wrapped_key, master_key) if db_note.wrapped_key else master_key
            db_note.content = decrypt_content(db_note.content, dek)
        except Exception:
            db_note.content = "[unable to decrypt]"
        return db_note


# @app.delete("/notes/{note_id}")
# async def delete_note(note_id: int):
#     from sqlalchemy import select
#     async with AsyncSessionLocal() as session:
#         result = await session.execute(select(Note).filter(Note.id == note_id))
#         db_note = result.scalar_one_or_none()
#         if not db_note:
#             raise HTTPException(status_code=404, detail="Note not found")
#         # Zero the wrapped key first to ensure cryptographic deletion of DEK material
#         db_note.wrapped_key = None
#         await session.commit()
#         await session.delete(db_note)
#         await session.commit()
#         return {"message": "Note deleted"}


# --- from here onwards Prathip's code ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
EMAIL_SERIALIZER = URLSafeTimedSerializer("EMAIL_TOKEN_SECRET_KEY")
WEAK_PINS = ["123456", "000000", "111111", "222222", "333333", "444444", "555555", "666666", "777777", "888888", "999999", "654321"]

# Path adjustment for your directory structure
from pathlib import Path
_BASE_DIR = Path(__file__).parent.parent
templates = Jinja2Templates(directory=str(_BASE_DIR / "frontend" / "templates"))
app.mount("/static", StaticFiles(directory=str(_BASE_DIR / "frontend")), name="static")

# --- SESSION SECURITY: Inactivity Auto Logout ---
INACTIVITY_TIMEOUT_MINUTES = 15
LAST_ACTIVITY_UPDATE_SECONDS = 60  # don't spam DB updates on every request


# --- NEW: Trash + Audit models ---

class NoteTrash(Base):
    __tablename__ = "notes_trash"
    id = Column(Integer, primary_key=True, index=True)

    original_note_id = Column(Integer, index=True, nullable=False)

    title = Column(String, index=True)
    content = Column(String)

    wrapped_key = Column(String, nullable=True)

    created_at = Column(DateTime)
    updated_at = Column(DateTime)

    deleted_at = Column(DateTime, default=get_sg_time, index=True)

    deleted_by_email = Column(String, index=True, nullable=False)
    deleted_by_username = Column(String, nullable=True)

    integrity_hmac = Column(String, nullable=True)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String, index=True, nullable=False)  # NOTE_TRASHED, NOTE_RESTORED, TRASH_PURGED, etc.
    actor_email = Column(String, index=True, nullable=True)
    actor_role = Column(String, index=True, nullable=True)
    ip = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    note_id = Column(Integer, index=True, nullable=True)
    trash_id = Column(Integer, index=True, nullable=True)
    details = Column(JSON, default=dict)
    created_at = Column(DateTime, default=get_sg_time, index=True)

# --- End: NEW Trash + Audit models ---


BACKUP_PIN_MAX_AGE_DAYS = 180


def is_backup_pin_expired(user: "User") -> bool:
    if not user.backup_pin_changed_at:
        return True
    return (get_sg_time() - user.backup_pin_changed_at) > timedelta(days=BACKUP_PIN_MAX_AGE_DAYS)


def get_backup_pin_history_limit(user: "User") -> int:
    # Policy: users last 3, admins/superadmin last 5
    return 5 if user.role in ["admin", "superadmin"] else 3

def backup_pin_reused(user: "User", new_pin: str) -> bool:
    # Blocks reuse of current pin + last N pin hashes

    # 1) Block current PIN
    try:
        if user.backup_pin_hash and pwd_context.verify(new_pin, user.backup_pin_hash):
            return True
    except Exception:
        # If current hash is malformed, fail safe
        return True

    # 2) Normalize history into a Python list
    history = user.backup_pin_history
    if history is None:
        history = []
    elif isinstance(history, str):
        # If stored/returned as JSON string for any reason
        try:
            history = json.loads(history)
        except Exception:
            history = []

    # 3) Check last N history hashes
    limit = get_backup_pin_history_limit(user)
    for old_hash in history[-limit:]:
        if not old_hash:
            continue
        try:
            if pwd_context.verify(new_pin, old_hash):
                return True
        except Exception:
            # Malformed hash -> fail safe
            return True

    return False

def push_backup_pin_history(user: "User"):
    # Store the previous pin hash, keep only last N
    if not user.backup_pin_hash:
        return

    history = user.backup_pin_history
    if history is None:
        history = []
    elif isinstance(history, str):
        try:
            history = json.loads(history)
        except Exception:
            history = []

    history.append(user.backup_pin_hash)

    limit = get_backup_pin_history_limit(user)
    user.backup_pin_history = history[-limit:]


def ensure_str_challenge(challenge) -> str:
    """
    webauthn may give challenge as bytes or str depending on version.
    DB column is VARCHAR, so we always store base64url STRING.
    """
    if isinstance(challenge, bytes):
        return bytes_to_base64url(challenge)  # from webauthn.helpers
    return str(challenge)


def set_trusted_device_cookie(response: RedirectResponse, user: "User"):
    response.set_cookie(
        key="device_token",
        value=EMAIL_SERIALIZER.dumps(user.email, salt=user.device_key),
        max_age=60 * 60 * 24 * 30,  # 30 days
        httponly=True,
        samesite="lax",
        path="/",
    )


TRUSTED_DEVICE_COOKIE = "device_token"
TRUST_DAYS = 30

def _hash_ua(ua: str) -> str:
    ua = ua or ""
    return hashlib.sha256(ua.encode("utf-8")).hexdigest()

def _parse_device_cookie(val: str) -> tuple[str, str] | tuple[None, None]:
    # format: "<device_id>.<secret>"
    if not val or "." not in val:
        return (None, None)
    device_id, secret = val.split(".", 1)
    if not device_id or not secret:
        return (None, None)
    return (device_id, secret)

def _set_trusted_device_cookie(response: RedirectResponse, device_id: str, secret: str):
    response.set_cookie(
        key=TRUSTED_DEVICE_COOKIE,
        value=f"{device_id}.{secret}",
        max_age=60 * 60 * 24 * TRUST_DAYS,
        httponly=True,
        samesite="lax",
        secure=True,   # IMPORTANT since you're using https://localhost
        path="/",
    )

async def is_trusted_device(request: Request, user: "User", db: AsyncSession) -> bool:
    raw = request.cookies.get(TRUSTED_DEVICE_COOKIE)
    device_id, secret = _parse_device_cookie(raw)
    if not device_id:
        return False

    res = await db.execute(
        select(TrustedDevice).where(
            TrustedDevice.id == device_id,
            TrustedDevice.user_email == user.email,
            TrustedDevice.revoked_at.is_(None),
        )
    )
    td = res.scalar_one_or_none()
    if not td:
        return False

    # Verify cookie secret against server-stored hash
    try:
        if not pwd_context.verify(secret, td.secret_hash):
            return False
    except Exception:
        return False

    # Optional: UA consistency check (helps against cookie theft)
    current_ua_hash = _hash_ua(request.headers.get("user-agent", ""))
    if td.ua_hash and td.ua_hash != current_ua_hash:
        return False

    # Update last seen (best effort)
    td.last_seen_at = get_sg_time()
    await db.commit()

    return True

async def trust_this_device(request: Request, user: "User", db: AsyncSession, response: RedirectResponse):
    """
    Creates or refreshes trust for this browser/profile.
    If cookie already exists and is valid, just refresh last_seen and cookie expiry.
    Otherwise create a new TrustedDevice entry.
    """
    raw = request.cookies.get(TRUSTED_DEVICE_COOKIE)
    device_id, secret = _parse_device_cookie(raw)

    ua_hash = _hash_ua(request.headers.get("user-agent", ""))

    # If an existing cookie is present, try to reuse device_id
    if device_id:
        res = await db.execute(
            select(TrustedDevice).where(
                TrustedDevice.id == device_id,
                TrustedDevice.user_email == user.email,
                TrustedDevice.revoked_at.is_(None),
            )
        )
        td = res.scalar_one_or_none()
        if td:
            # Refresh expiry + last seen
            td.last_seen_at = get_sg_time()
            td.ua_hash = ua_hash  # keep updated
            await db.commit()

            # Also refresh cookie max_age by re-setting it
            # NOTE: we cannot recover the original secret, so only refresh if current cookie is valid
            try:
                if pwd_context.verify(secret or "", td.secret_hash):
                    _set_trusted_device_cookie(response, device_id, secret)
            except Exception:
                pass
            return

    # Otherwise create a new trusted device
    new_device_id = str(uuid.uuid4())
    new_secret = secrets.token_urlsafe(32)

    td = TrustedDevice(
        id=new_device_id,
        user_email=user.email,
        secret_hash=pwd_context.hash(new_secret),
        ua_hash=ua_hash,
        created_at=get_sg_time(),
        last_seen_at=get_sg_time(),
        revoked_at=None,
    )
    db.add(td)
    await db.commit()

    _set_trusted_device_cookie(response, new_device_id, new_secret)

async def revoke_this_device(request: Request, user: "User", db: AsyncSession):
    raw = request.cookies.get(TRUSTED_DEVICE_COOKIE)
    device_id, secret = _parse_device_cookie(raw)
    if not device_id:
        return

    res = await db.execute(
        select(TrustedDevice).where(
            TrustedDevice.id == device_id,
            TrustedDevice.user_email == user.email,
            TrustedDevice.revoked_at.is_(None),
        )
    )
    td = res.scalar_one_or_none()
    if not td:
        return

    # Only revoke if the cookie secret matches (prevents random revokes)
    try:
        if not pwd_context.verify(secret or "", td.secret_hash):
            return
    except Exception:
        return

    td.revoked_at = get_sg_time()
    await db.commit()

# --- WebAuthn / Passkeys (Platform Biometrics for Admins) ---
WEBAUTHN_RP_ID = "localhost"                 # must match your domain
WEBAUTHN_RP_NAME = "NoteVault"
WEBAUTHN_ORIGIN = "https://localhost:8000"    # MUST match your browser URL exactly
PASSKEY_TTL_SECONDS = 120


# -------------------------
# Helpers (keep + add)
# -------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.clip(np.dot(a, b), -1.0, 1.0))

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    n = float(np.linalg.norm(v) + 1e-12)
    return v / n

def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(float(x) for x in vals)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def device_fingerprint_string(request: Request) -> str:
    ua = request.headers.get("user-agent", "") or ""
    ip = request.client.host if request.client else ""
    return f"{ua}|{ip}"


def is_device_match(request: Request, stored_hash: str) -> bool:
    if not stored_hash:
        return False
    try:
        return pwd_context.verify(device_fingerprint_string(request), stored_hash)
    except Exception:
        return False


class PendingAction(Base):
    """Temporary table to track email verification status across tabs"""
    __tablename__ = "pending_actions"
    email = Column(String, primary_key=True)
    action_data = Column(JSON) # Stores username/password temporarily
    action_type = Column(String) # 'signup' or 'login'
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=get_sg_time)


class WebAuthnSession(Base):
    __tablename__ = "webauthn_sessions"
    id = Column(String, primary_key=True, index=True)  # uuid
    email = Column(String, index=True, nullable=False)
    kind = Column(String, nullable=False)              # "register" or "login"
    challenge = Column(String, nullable=False)         # base64url challenge
    created_at = Column(DateTime, default=get_sg_time)
    used = Column(Boolean, default=False)


class TrustedDevice(Base):
    __tablename__ = "trusted_devices"
    id = Column(String, primary_key=True, index=True)          # device_id (uuid string)
    user_email = Column(String, index=True, nullable=False)    # link to User.email
    secret_hash = Column(String, nullable=False)               # hash(secret)
    ua_hash = Column(String, nullable=True)                    # hash(user-agent) (optional but useful)
    created_at = Column(DateTime, default=get_sg_time)
    last_seen_at = Column(DateTime, default=get_sg_time)
    revoked_at = Column(DateTime, nullable=True)


def has_passkey(user: "User") -> bool:
    return bool(user.passkeys and isinstance(user.passkeys, list) and len(user.passkeys) > 0)


@app.exception_handler(HTTPException)
async def auth_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 303:
        return RedirectResponse(url="/login", status_code=303)

    # ✅ Inactivity logout: clear session cookies and redirect
    if exc.status_code == 440 and exc.detail == "SESSION_EXPIRED":
        resp = RedirectResponse(
            url="/login?error=Session%20expired%20due%20to%20inactivity.%20Please%20log%20in%20again",
            status_code=303
        )
        resp.delete_cookie("session_id", path="/")
        resp.delete_cookie("tmp_login", path="/")
        resp.delete_cookie(ADMIN_PASSKEY_COOKIE, path="/")
        # NOTE: We do NOT delete trusted device cookie (by your design)
        return resp

    if exc.status_code == 409 and exc.detail == "PIN_EXPIRED":
        return RedirectResponse(url="/change-backup-pin", status_code=303)

    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


def apply_lockout_policy(user: User):
    """
    Applies Infosec Progressive Lockout.
    Admins: Hard lock after 3 failed attempts (manual unlock only).
    Users: 30m base lockout, increasing by 30m for every failure thereafter.
    """
    if user.role in ["admin", "superadmin"]:
        # Allow admins 3 attempts before triggering the hard 'locked' status
        if user.failed_login_attempts >= 3 or user.failed_2fa_attempts >= 3:
            user.status = "locked"
            user.lockout_until = None
    else:
        # --- DO NOT CHANGE: Normal User Logic ---
        total_fails = max(user.failed_login_attempts, user.failed_2fa_attempts)
        
        if user.failed_login_attempts >= 5 or user.failed_2fa_attempts >= 3:
            user.status = "locked"
            threshold = 5 if user.failed_login_attempts >= 5 else 3
            multiplier = (total_fails - (threshold - 1))
            user.lockout_until = get_sg_time() + timedelta(minutes=30 * multiplier)


# --- NEW: Trash retention policy ---
TRASH_RETENTION_DAYS = 30

AUTO_TRASH_PURGE_INTERVAL_SECONDS = 10  # run once a day

async def purge_expired_trash_system():
    cutoff = get_sg_time() - timedelta(days=TRASH_RETENTION_DAYS)

    async with AsyncSessionLocal() as db:
        # Collect IDs first (so we can log purged_count)
        res = await db.execute(select(NoteTrash.id).where(NoteTrash.deleted_at < cutoff))
        ids = [r[0] for r in res.all()]

        if ids:
            await db.execute(delete(NoteTrash).where(NoteTrash.id.in_(ids)))
            await db.commit()

            # Log as SYSTEM purge (no request, no actor)
            await audit_log(
                db=db,
                request=None,
                event_type="TRASH_PURGED",
                actor=None,
                details={
                    "mode": "expired_auto",
                    "retention_days": TRASH_RETENTION_DAYS,
                    "purged_count": len(ids),
                },
            )


async def auto_purge_expired_trash_loop():
    while True:
        try:
            await purge_expired_trash_system()
        except Exception as e:
            print("AUTO TRASH PURGE FAILED:", repr(e))
        await asyncio.sleep(AUTO_TRASH_PURGE_INTERVAL_SECONDS)

def _trash_hmac_secret() -> bytes:
    # put this in env var in real deployment
    return (os.getenv("TRASH_HMAC_SECRET") or "CHANGE_ME_TRASH_HMAC_SECRET").encode("utf-8")

def compute_trash_hmac(payload: str) -> str:
    return hashlib.sha256(_trash_hmac_secret() + payload.encode("utf-8")).hexdigest()

async def audit_log(
    db: AsyncSession,
    request: Request | None,
    event_type: str,
    actor: "User" = None,
    note_id: int = None,
    trash_id: int = None,
    details: dict = None,
):
    ev = AuditEvent(
        event_type=event_type,
        actor_email=getattr(actor, "email", None) if actor else None,
        actor_role=getattr(actor, "role", None) if actor else None,
        ip=(request.client.host if (request and request.client) else None),
        user_agent=(request.headers.get("user-agent", None) if request else None),
        note_id=note_id,
        trash_id=trash_id,
        details=details or {},
        created_at=get_sg_time(),
    )
    db.add(ev)
    await db.commit()


@app.on_event("startup")
async def startup_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# --- NEW: Trash Routes ---

@app.on_event("startup")
async def start_auto_trash_purge():
    # run once immediately
    try:
        await purge_expired_trash_system()
    except Exception as e:
        print("STARTUP TRASH PURGE FAILED:", repr(e))

    # then run daily
    asyncio.create_task(auto_purge_expired_trash_loop())

def require_admin_passkey(request: Request, user: "User"):
    if user.role not in ["admin", "superadmin"]:
        raise HTTPException(status_code=403, detail="Admins only")
    if request.cookies.get(ADMIN_PASSKEY_COOKIE) != "1":
        raise HTTPException(status_code=403, detail="Passkey verification required")

@app.delete("/notes/{note_id}")
@limiter.limit("10/minute")
async def delete_note(
    note_id: int,
    request: Request,
    user: User = Depends(verify_session),
    db: AsyncSession = Depends(get_db),
):
    # 1) Load note
    res = await db.execute(select(Note).where(Note.id == note_id))
    note = res.scalar_one_or_none()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    # 2) Create trash record (soft-delete)
    deleted_at = get_sg_time()
    payload = f"{note.id}|{note.title}|{note.content}|{note.created_at}|{note.updated_at}|{deleted_at}|{user.email}"
    h = compute_trash_hmac(payload)

    trash = NoteTrash(
        original_note_id=note.id,
        title=note.title,
        content=note.content,           # keep encrypted blob as-is
        wrapped_key=note.wrapped_key,
        created_at=note.created_at,
        updated_at=note.updated_at,
        deleted_at=deleted_at,
        deleted_by_email=user.email,
        deleted_by_username=user.username,
        integrity_hmac=h,
    )
    db.add(trash)

    # 3) Preserve teammate's crypto-delete intent:
    #    wipe key material on the NOTE row before deleting it (only if column exists)
    if hasattr(note, "wrapped_key"):
        note.wrapped_key = None

    # 4) Delete original note row
    await db.delete(note)

    # 5) Commit once (atomic)
    await db.commit()
    await db.refresh(trash)

    # 6) Audit
    await audit_log(
        db=db,
        request=request,
        event_type="NOTE_TRASHED",
        actor=user,
        note_id=note_id,
        trash_id=trash.id,
        details={"retention_days": TRASH_RETENTION_DAYS},
    )

    return {"message": "Note deleted", "trash_id": trash.id}


@app.get("/trash")
@limiter.limit("20/minute")
async def list_my_trash(
    request: Request,
    user: User = Depends(verify_session),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(
        select(NoteTrash)
        .where(NoteTrash.deleted_by_email == user.email)
        .order_by(NoteTrash.deleted_at.desc())
    )
    items = res.scalars().all()

    # return minimal metadata (content stays encrypted; decrypt client-side only if needed)
    return [
        {
            "trash_id": t.id,
            "original_note_id": t.original_note_id,
            "title": t.title,
            "deleted_at": t.deleted_at,
            "deleted_by_email": t.deleted_by_email,
        }
        for t in items
    ]


@app.post("/trash/{trash_id}/restore")
@limiter.limit("10/minute")
async def restore_from_trash(
    trash_id: int,
    request: Request,
    user: User = Depends(verify_session),
    db: AsyncSession = Depends(get_db),
):
    # ✅ Enforce "only the deleter can restore" (privacy-by-default)
    res = await db.execute(
        select(NoteTrash).where(
            NoteTrash.id == trash_id,
            NoteTrash.deleted_by_email == user.email,   # ✅ correct column
        )
    )
    t = res.scalar_one_or_none()
    if not t:
        raise HTTPException(status_code=404, detail="Trash item not found")

    # ✅ Integrity guard (HMAC)
    payload = (
        f"{t.original_note_id}|{t.title}|{t.content}|{t.created_at}|"
        f"{t.updated_at}|{t.deleted_at}|{t.deleted_by_email}"
    )
    expected = compute_trash_hmac(payload)
    if t.integrity_hmac and t.integrity_hmac != expected:
        await audit_log(
            db, request, "TRASH_TAMPER_DETECTED",
            actor=user, trash_id=trash_id,
            details={"reason": "HMAC mismatch"}
        )
        raise HTTPException(status_code=409, detail="Trash item integrity failed (possible tampering)")

    # ✅ Save values BEFORE delete (safer)
    original_note_id = t.original_note_id

    # Restore note (still encrypted content stored)
    restored = Note(
        title=t.title,
        content=t.content,
        wrapped_key=getattr(t, "wrapped_key", None),
        created_at=t.created_at or get_sg_time(),
        updated_at=get_sg_time(),
    )
    db.add(restored)

    await db.delete(t)
    await db.commit()
    await db.refresh(restored)

    await audit_log(
        db, request, "NOTE_RESTORED",
        actor=user,
        note_id=restored.id,
        trash_id=trash_id,
        details={"from_original_note_id": original_note_id},
    )

    return {"message": "Restored", "restored_note_id": restored.id}


@app.delete("/trash/{trash_id}/purge")
@limiter.limit("10/minute")
async def purge_one_trash_item(
    trash_id: int,
    request: Request,
    user: User = Depends(verify_session),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(select(NoteTrash).where(NoteTrash.id == trash_id))
    t = res.scalar_one_or_none()
    if not t:
        raise HTTPException(status_code=404, detail="Trash item not found")

    # ✅ Only the user who deleted it can permanently delete it
    if t.deleted_by_email != user.email:
        raise HTTPException(status_code=403, detail="Not allowed")

    await db.delete(t)
    await db.commit()

    await audit_log(
        db,
        request,
        "TRASH_PURGED",
        actor=user,
        trash_id=trash_id,
        details={"mode": "user_single"},
    )

    return {"message": "Permanently Deleted"}


@app.post("/trash/purge-expired")
@limiter.limit("2/minute")
async def purge_expired_trash(
    request: Request,
    user: User = Depends(verify_session),
    db: AsyncSession = Depends(get_db),
):
    # Admin-only + passkey
    require_admin_passkey(request, user)

    cutoff = get_sg_time() - timedelta(days=TRASH_RETENTION_DAYS)

    # fetch ids for audit count
    res = await db.execute(select(NoteTrash.id).where(NoteTrash.deleted_at < cutoff))
    ids = [r[0] for r in res.all()]

    if ids:
        await db.execute(delete(NoteTrash).where(NoteTrash.id.in_(ids)))
        await db.commit()

    await audit_log(db, request, "TRASH_PURGED", actor=user, details={
        "mode": "expired",
        "retention_days": TRASH_RETENTION_DAYS,
        "purged_count": len(ids),
    })

    return {"message": "Expired trash purged", "purged_count": len(ids), "retention_days": TRASH_RETENTION_DAYS}


@app.get("/notes/{note_id}/copy-text")
@limiter.limit("6/minute")
async def get_copy_text(
    note_id: int,
    request: Request,
    user: User = Depends(verify_session),
):
    key = get_encryption_key()

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Note).where(Note.id == note_id))
        note = result.scalar_one_or_none()
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")

        # decrypt
        master_key = get_encryption_key()

        if note.wrapped_key:
            dek = unwrap_key(note.wrapped_key, master_key)
            content = decrypt_content(note.content, dek)
        else:
            content = decrypt_content(note.content, master_key)

        # stamp
        stamp_time = get_sg_time().strftime("%d-%m-%Y %H:%M:%S (SGT)")
        stamped = (
            f"{note.title}\n\n{content}\n\n"
            f"— Copied from NoteVault by {user.email} on {stamp_time} • note_id={note_id}"
        )

    # Optional audit trail (recommended)
    print(
        f"AUDIT_COPY note_id={note_id} user={user.email} ip={request.client.host} "
        f"ua={request.headers.get('user-agent','')}"
    )

    return {"text": stamped}

@app.get("/notes/{note_id}/export-pdf")
@limiter.limit("2/minute")
async def export_note_pdf(
    note_id: int,
    request: Request,
    user: User = Depends(verify_session),
):
    key = get_encryption_key()

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Note).where(Note.id == note_id))
        note = result.scalar_one_or_none()
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")

        # decrypt content
        master_key = get_encryption_key()

        if note.wrapped_key:
            dek = unwrap_key(note.wrapped_key, master_key)
            content = decrypt_content(note.content, dek)
        else:
            content = decrypt_content(note.content, master_key)

    # --- Build PDF in-memory ---
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # InfoSec watermark stamp (non-repudiation / traceability)
    stamp_time = get_sg_time().strftime("%d-%m-%Y %H:%M:%S (SGT)")
    watermark_text = f"NoteVault • {user.email} • {user.username} • note_id={note_id} • {stamp_time}"

    c.saveState()
    try:
        # If supported, real transparency
        c.setFillAlpha(0.08)
    except Exception:
        # Fallback for older reportlab: just use light gray
        pass

    c.setFont("Helvetica-Bold", 22)
    c.setFillColorRGB(0.2, 0.2, 0.2)
    c.translate(width / 2, height / 2)
    c.rotate(30)
    c.drawCentredString(0, 0, watermark_text)
    c.restoreState()

    # Title + body
    margin_x = 0.75 * inch
    y = height - 1.0 * inch

    c.setFont("Helvetica-Bold", 16)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(margin_x, y, note.title or f"Note {note_id}")

    y -= 0.4 * inch
    c.setFont("Helvetica", 11)

    # Simple word-wrap
    max_width = width - (2 * margin_x)
    words = (content or "").split()
    line = ""

    def flush_line(curr_y, text_line):
        c.drawString(margin_x, curr_y, text_line)

    for w in words:
        test = (line + " " + w).strip()
        if c.stringWidth(test, "Helvetica", 11) <= max_width:
            line = test
        else:
            flush_line(y, line)
            y -= 14
            line = w
            if y < 0.9 * inch:
                c.showPage()
                y = height - 1.0 * inch
                # repeat watermark on new page
                c.saveState()
                try:
                    c.setFillAlpha(0.08)
                except Exception:
                    pass
                c.setFont("Helvetica-Bold", 22)
                c.setFillColorRGB(0.2, 0.2, 0.2)
                c.translate(width / 2, height / 2)
                c.rotate(30)
                c.drawCentredString(0, 0, watermark_text)
                c.restoreState()
                c.setFont("Helvetica", 11)

    if line:
        flush_line(y, line)

    # Footer stamp (explicit attribution)
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColorRGB(0.2, 0.2, 0.2)
    c.drawRightString(width - margin_x, 0.6 * inch, watermark_text)

    c.showPage()
    c.save()

    buffer.seek(0)

    # Audit log (optional but good for InfoSec)
    print(
        f"AUDIT_EXPORT_PDF note_id={note_id} user={user.email} ip={request.client.host} "
        f"ua={request.headers.get('user-agent','')}"
    )

    filename = f"note_{note_id}.pdf"
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

# Dashboard
@app.get("/home", name="dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: User = Depends(verify_session)):
    message = request.query_params.get("message")  # ✅ read from /home?message=...
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": user,
        "message": message or f"Welcome back, {user.username}"
    })

@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request, user: User = Depends(verify_session)):
    if user.role not in ["admin", "superadmin"]:
        return RedirectResponse(url="/home", status_code=303)

    # Must have passkey enrolled
    if not has_passkey(user):
        return RedirectResponse(url="/admin/passkey", status_code=303)

    # ✅ Must have verified passkey this session
    if request.cookies.get(ADMIN_PASSKEY_COOKIE) != "1":
        return RedirectResponse(url="/admin/passkey-verify", status_code=303)

    return templates.TemplateResponse("admin.html", {"request": request})


# --- HANDSHAKE POLLING ---

@app.get("/auth/poll-status/{email}")
async def poll_verification_status(email: str, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(PendingAction).filter(PendingAction.email == email))
    action = res.scalar_one_or_none()

    if not action:
        return {"verified": False}

    TTL_SECONDS = 700

    # NEW: created_at safety
    if not action.created_at:
        # If timestamp is missing, treat as stale and delete
        await db.execute(delete(PendingAction).where(PendingAction.email == email))
        await db.commit()
        return {"verified": False}

    age = (get_sg_time() - action.created_at).total_seconds()
    if age > TTL_SECONDS:
        await db.execute(delete(PendingAction).where(PendingAction.email == email))
        await db.commit()
        return {"verified": False}

    if action.is_verified:
        return {"verified": True, "action": action.action_type}

    return {"verified": False}

# --- SIGNUP FLOW ---

@app.get("/register", response_class=HTMLResponse)
async def render_signup(request: Request):
    # Standard render, no error
    return templates.TemplateResponse("signup.html", {"request": request})

def validate_password_complex(password: str, email: str = None, username: str = None):
    errors = []
    if len(password) < 14:
        errors.append("Password must be at least 14 characters long")
    if not re.search(r"[A-Z]", password):
        errors.append("Password must be at least one uppercase letter")
    if not re.search(r"[a-z]", password):
        errors.append("Password must be at least one lowercase letter")
    if not re.search(r"\d", password):
        errors.append("Password must be at least one number")
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        errors.append("Password must be at least one special character")

    # NEW: Identity Overlap Protection
    if email and email.lower() in password.lower():
        errors.append("Password cannot contain your email address")
    if username and username.lower() in password.lower():
        errors.append("Password cannot contain your username")
    return errors

@app.post("/register")
async def signup_start(
    request: Request, 
    username: str = Form(...), 
    email: str = Form(...), 
    password: str = Form(...), 
    confirm_password: str = Form(...), 
    db: AsyncSession = Depends(get_db)
):
    all_errors = []

    # 1. NEW: Identity Policy Check (Username must not be the same as Email)
    if username.strip().lower() == email.strip().lower():
        all_errors.append("Username and Email must be different")

    # 2. Check if passwords match
    if password != confirm_password:
        all_errors.append("Passwords do not match")

    # 3. Check all password requirements (Including identity overlap check)
    password_errors = validate_password_complex(password, email, username)
    if password_errors:
        all_errors.extend(password_errors)

    # 4. Impersonation Protection & Duplicate Check
    normalized_new_user = re.sub(r'\s+', '', username).lower()
    res = await db.execute(select(User))
    existing_users = res.scalars().all()
    
    for u in existing_users:
        if u.email == email:
            all_errors.append("Email is too similar or already registered")
        
        normalized_existing = re.sub(r'\s+', '', u.username).lower()
        if normalized_existing == normalized_new_user:
            all_errors.append("Username is already taken or too similar to an existing account")

    # 5. Return all errors in "One Shot"
    if all_errors:
        error_message = " | ".join(all_errors) 
        return templates.TemplateResponse("signup.html", {
            "request": request, 
            "error": error_message, 
            "username": username, 
            "email": email
        })

    # 6. Success Flow: Handle PendingAction
    
    signup_nonce = str(uuid.uuid4())
    action_data = {"username": username, "password": password, "signup_nonce": signup_nonce}
    pending = PendingAction(
        email=email,
        action_type="signup",
        action_data=action_data,
        is_verified=False,
        created_at=get_sg_time()
    )
    await db.merge(pending)
    await db.commit()

    # 7. Generate token and send email
    token = EMAIL_SERIALIZER.dumps(email, salt=signup_nonce)
    verify_url = f"https://localhost:8000/verify-registration?token={token}"
    
    html_content = f"""
    <h3>Welcome to {APP_NAME}</h3>
    <p>Please click the button below to verify your account:</p>
    <a href="{verify_url}" style="background:#2c3e50; color:white; padding:10px; text-decoration:none; border-radius:5px;">Verify Email Address</a>
    """
    
    send_security_email("Verify Your Email", email, html_content)
    print(f"DEBUG: Sent registration email to {email} with link: {verify_url}")
    
    return templates.TemplateResponse("2fa_email_code.html", {"request": request, "email": email})

@app.get("/verify-registration", response_class=HTMLResponse)
async def verify_registration(token: str, db: AsyncSession = Depends(get_db)):
    try:
        # 1. Unsafely get the email to find the pending registration data
        raw_email = EMAIL_SERIALIZER.loads_unsafe(token)[1]
        email = raw_email.decode('utf-8') if isinstance(raw_email, bytes) else raw_email
        res = await db.execute(select(PendingAction).filter(PendingAction.email == email))
        pending = res.scalar_one_or_none()
        
        if not pending:
            raise Exception("No pending registration found")

        # 2. Verify the token using the action_data (username/password) as the salt
        # This binds the link to that specific signup attempt
        signup_nonce = (pending.action_data or {}).get("signup_nonce")
        if not signup_nonce:
            raise Exception("Missing signup nonce (stale signup request)")
        
        EMAIL_SERIALIZER.loads(token, salt=signup_nonce, max_age=600)
        
        await db.execute(update(PendingAction).where(PendingAction.email == email).values(is_verified=True))
        await db.commit()
        
        return HTMLResponse("""
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; background-color: #f4f4f4; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
                .container { background: white; padding: 50px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1); text-align: center; max-width: 450px; }
                h1 { color: #5cb85c; margin-bottom: 20px; }
                p { color: #666; line-height: 1.5; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Verification Successful!</h1>
                <p>You can now close this tab and return to your original registration window</p>
            </div>
        </body>
        </html>
        """)
    except Exception as e:
        print(f"DEBUG: Registration Link Error: {e}")
        return HTMLResponse("""
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; background-color: #f4f4f4; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
                .container { background: white; padding: 50px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1); text-align: center; }
                h1 { color: #d9534f; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Link Expired or Invalid</h1>
                <p>Please try registering again. Old registration links are voided if a newer one is requested</p>
            </div>
        </body>
        </html>
        """)

@app.get("/setup-google-auth", response_class=HTMLResponse)
async def render_setup_qr(request: Request, email: str):
    """Called automatically by Tab A when polling detects verification"""
    secret = pyotp.random_base32()
    totp = pyotp.TOTP(secret)
    prov_url = totp.provisioning_uri(name=email, issuer_name="NoteVault")
    
    img = qrcode.make(prov_url)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    qr_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return templates.TemplateResponse("setup_google_auth.html", {
        "request": request, "qr_code": qr_b64, "email": email, "secret": secret
    })

@app.post("/setup-google-auth")
async def process_totp(request: Request, token: str = Form(...), secret: str = Form(...), email: str = Form(...)):
    if pyotp.TOTP(secret).verify(token):
        return templates.TemplateResponse("backup_code.html", {"request": request, "email": email, "secret": secret})
    raise HTTPException(status_code=400, detail="Invalid Google Auth Code")

@app.post("/setup-backup-pin")
async def finalize_signup(
    request: Request, 
    backup_pin: str = Form(...), 
    confirm_pin: str = Form(...), 
    secret: str = Form(...), 
    email: str = Form(...), 
    db: AsyncSession = Depends(get_db)
):
    pin_errors = []

    # 1. Validation
    if backup_pin != confirm_pin:
        pin_errors.append("PINs do not match")
    if backup_pin in WEAK_PINS:
        pin_errors.append("This PIN is too weak. Please choose a more secure 6-digit PIN")

    if pin_errors:
        error_message = " | ".join(pin_errors)
        return templates.TemplateResponse("backup_code.html", {
            "request": request, "email": email, "secret": secret, "error": error_message
        })
    
    # 2. Retrieve original signup data
    res = await db.execute(select(PendingAction).filter(PendingAction.email == email))
    pending = res.scalar_one_or_none()
    
    if not pending:
        return templates.TemplateResponse("signup.html", {
            "request": request, "error": "Session expired. Please restart registration"
        })
    
    # 3. Create the New User
    session_id = str(uuid.uuid4()) # Generate session ID for auto-login
    new_user = User(
        username=pending.action_data['username'],
        email=email, 
        hashed_password=pwd_context.hash(pending.action_data['password']), 
        backup_pin_hash=pwd_context.hash(backup_pin),
        backup_pin_history=[],
        backup_pin_changed_at=get_sg_time(),
        must_change_backup_pin=False,
        totp_secret=secret,
        role="user",
        status="active",
        current_session_id=session_id, # Log them in immediately
        last_activity_at=get_sg_time(),
    )
    
    db.add(new_user)

    # Ensure master encryption key exists; generate and save automatically on first user creation
    try:
        if not key_exists():
            master_key = generate_key()
            save_key(master_key)
            print("DEBUG: Generated master encryption key during user signup")
    except Exception as e:
        # Log error but do not block user creation -- administrators should investigate
        print(f"CRITICAL: Failed to generate/save master encryption key: {e}")

    await db.execute(delete(PendingAction).where(PendingAction.email == email))
    await db.commit()
    
    # 4. AUTO-LOGIN: Define Redirect to /home and attach the session cookie
    response = RedirectResponse(url="/home", status_code=303)
    response.set_cookie(
        key="session_id", 
        value=session_id, 
        httponly=True, 
        samesite="lax",
        path="/",
    )
    
    return response

# --- LOGIN FLOW ---

@app.get("/", response_class=HTMLResponse)
async def render_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def render_login(request: Request):
    error = request.query_params.get("error")
    message = request.query_params.get("message")
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": error,
        "message": message
    })

@app.post("/login")
@limiter.limit("5/minute")
async def login_process(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    # 1. Fetch user
    res = await db.execute(select(User).filter(User.email == email))
    user = res.scalar_one_or_none()

    if not user:
        await asyncio.sleep(0.5)
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Incorrect email or password"}
        )

    # 2. Cleanup old pending actions for this email (prevents cross-flow pollution)
    await db.execute(delete(PendingAction).where(PendingAction.email == email))
    await db.commit()

    # 3. Auto-unlock timed locks (users only)
    if user.status == "locked" and user.lockout_until:
        if get_sg_time() > user.lockout_until:
            user.status = "active"
            user.failed_login_attempts = 0
            user.failed_2fa_attempts = 0
            user.lockout_until = None
            await db.commit()

    # 4. Status checks
    if user.status == "banned":
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Account locked. Contact support."}
        )

    if user.status == "locked":
        # HARD LOCK for admins/superadmins (lockout_until is None)
        if user.role in ["admin", "superadmin"] and not user.lockout_until:
            return templates.TemplateResponse(
                "login.html",
                {"request": request, "error": "Account locked due to multiple failed attempts. Contact support."}
            )

        # TIMED LOCK for users
        if user.lockout_until:
            remaining = int((user.lockout_until - get_sg_time()).total_seconds() / 60)
            display_time = remaining if remaining > 0 else 1
            return templates.TemplateResponse(
                "login.html",
                {"request": request, "error": f"Too many attempts. Try again in {display_time} minutes"}
            )

        # Fallback
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Account locked. Contact support."}
        )

    # 5. Password Verification
    if not pwd_context.verify(password, user.hashed_password):
        user.failed_login_attempts += 1
        apply_lockout_policy(user)
        await db.commit()
        # Log failed login attempt
        try:
            ip = request.client.host if request.client else None
            device = device_fingerprint_string(request)
            await log_activity(db, user.id, ip, device, "login", success=False)
        except Exception as e:
            print(f"Activity logging failed: {e}")
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Incorrect email or password"}
        )
    


    # 6. Device Recognition Check (SERVER-SIDE TRUSTED DEVICES)
    is_recognized_device = await is_trusted_device(request, user, db)

    # 7. Create a PENDING session id (real session comes only after 2FA/PIN)
    pending_session_id = str(uuid.uuid4())

    # 8. Routing Logic
    if not is_recognized_device:
        # Handshake requires email link verification
        pending = PendingAction(
            email=email,
            action_type="login",
            action_data={"pending_session_id": pending_session_id},
            is_verified=False,
            created_at=get_sg_time()
        )
        await db.merge(pending)
        await db.commit()

        token = EMAIL_SERIALIZER.dumps(email, salt=pending_session_id)
        verify_url = f"https://localhost:8000/verify-login?token={token}"
        client_ip = request.client.host if request.client else "Unknown"
        user_agent = request.headers.get("user-agent", "Unknown")
        accept_lang = request.headers.get("accept-language", "Unknown")
        login_time = get_sg_time().strftime("%d %B %Y, %H:%M:%S (SGT)")

        html_body = (
            f"<h2>New Device Login</h2>"
            f"<p>We detected a login attempt from a new device. If this is you, authorize this session:</p>"
            f'<p><b>IP Address:</b> {client_ip}<br>'
            f"<b>Device/Browser (User-Agent):</b> {user_agent}<br>"
            f"<b>Language:</b> {accept_lang}<br>"
            f"<b>Time:</b> {login_time}</p>"
            f"<p>If you do not recognize this activity, do not authorize it and consider changing your password.</p>"
            f'<a href="{verify_url}" style="background:#2c3e50;color:white;padding:10px 20px;'
            f'text-decoration:none;border-radius:5px;">Authorize Login</a>'
        )
        send_security_email("NoteVault Login", email, html_body)
        print(f"DEBUG: Sent login verification email to {email} with link: {verify_url}")

        response = templates.TemplateResponse("2fa_email_code.html", {"request": request, "email": email})

    else:
        # Recognized device: treat as verified for the email-step
        pending = PendingAction(
            email=email,
            action_type="login",
            action_data={"pending_session_id": pending_session_id},
            is_verified=True,
            created_at=get_sg_time()
        )
        await db.merge(pending)
        await db.commit()

        response = RedirectResponse(url=f"/google-auth-2fa?email={email}", status_code=303)

    # tmp cookie only (optional, but useful for debugging)
    response.set_cookie(
        key="tmp_login",
        value=pending_session_id,
        httponly=True,
        samesite="lax",
        path="/"
    )

    return response

@app.get("/admin/list-users")
async def admin_list_users(current: User = Depends(verify_session), db: AsyncSession = Depends(get_db)):
    if current.role not in ["admin", "superadmin"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    res = await db.execute(select(User))
    users = res.scalars().all()
    out = []
    for u in users:
        out.append({
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "role": u.role,
            "status": u.status,
            "last_activity_at": u.last_activity_at.isoformat() if getattr(u, 'last_activity_at', None) else None,
        })
    return out


@app.post("/admin/manage")
async def admin_manage(target: str = None, action: str = None, current: User = Depends(verify_session), db: AsyncSession = Depends(get_db)):
    if current.role not in ["admin", "superadmin"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    if not target or not action:
        raise HTTPException(status_code=400, detail="Missing target or action")
    res = await db.execute(select(User).filter(User.email == target))
    u = res.scalar_one_or_none()
    if not u:
        raise HTTPException(status_code=404, detail="Target user not found")

    if action == "unlock":
        u.status = "active"
        u.failed_login_attempts = 0
        u.failed_2fa_attempts = 0
        u.lockout_until = None
    elif action == "ban":
        u.status = "banned"
    else:
        raise HTTPException(status_code=400, detail="Unknown action")

    await db.commit()
    return {"msg": f"Action '{action}' applied to {target}"}


@app.get("/admin/activity")
async def admin_activity(limit: int = 100, current: User = Depends(verify_session), db: AsyncSession = Depends(get_db)):
    if current.role not in ["admin", "superadmin"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    # Fetch recent activity from user_activity table
    q = """
        SELECT ua.id, ua.user_id, ua.login_time, ua.ip_address, ua.device_info, ua.action, ua.success, u.email
        FROM user_activity ua LEFT JOIN users u ON ua.user_id = u.id
        ORDER BY ua.login_time DESC LIMIT :limit
    """
    res = await db.execute(__import__("sqlalchemy").text(q), {"limit": limit})
    rows = res.fetchall()
    out = []
    for r in rows:
        out.append({
            "id": int(r[0]),
            "user_id": int(r[1]) if r[1] is not None else None,
            "login_time": r[2].isoformat() if r[2] else None,
            "ip_address": r[3],
            "device_info": r[4],
            "action": r[5],
            "success": bool(r[6]) if r[6] is not None else None,
            "email": r[7],
        })
    return out


@app.get("/admin/anomaly")
async def admin_anomaly(current: User = Depends(verify_session)):
    if current.role not in ["admin", "superadmin"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    model_file = Path(__file__).parent / "anomaly_model.pkl"
    exists = model_file.exists()
    info = {"model_exists": exists}
    if exists:
        info["modified_at"] = model_file.stat().st_mtime
    return info


@app.get("/verify-login", response_class=HTMLResponse)
async def verify_login_link(token: str, db: AsyncSession = Depends(get_db)):
    try:
        # 1) Extract email without trusting it yet
        raw_email = EMAIL_SERIALIZER.loads_unsafe(token)[1]
        email = raw_email.decode("utf-8") if isinstance(raw_email, bytes) else raw_email

        # 2) Load the latest pending login handshake (source of truth)
        pending_res = await db.execute(
            select(PendingAction).where(
                PendingAction.email == email,
                PendingAction.action_type == "login"
            )
        )
        pending = pending_res.scalar_one_or_none()

        if not pending or not pending.action_data or not pending.action_data.get("pending_session_id"):
            raise Exception("No active login request")

        latest_pending_sid = pending.action_data["pending_session_id"]

        # 3) Verify token using LATEST pending_session_id
        verified_email = EMAIL_SERIALIZER.loads(token, salt=latest_pending_sid, max_age=300)
        if verified_email != email:
            raise Exception("Email mismatch")

        # 4) Mark verified
        await db.execute(
            update(PendingAction)
            .where(PendingAction.email == email, PendingAction.action_type == "login")
            .values(is_verified=True)
        )
        await db.commit()

        return HTMLResponse("""
        <html><body style="font-family:Arial; text-align:center; padding:50px;">
            <h1 style="color:#5cb85c;">Login Verified!</h1>
            <p>You may close this tab and return to your login window</p>
        </body></html>
        """)

    except Exception as e:
        print("DEBUG: Verification Error:", e)
        return HTMLResponse("""
        <html><body style="font-family:Arial; text-align:center; padding:50px;">
            <h1 style="color:#d9534f;">Link Expired or Invalid</h1>
            <p>A newer login link may have been requested, or the link has expired</p>
        </body></html>
        """)


@app.get("/google-auth-2fa", response_class=HTMLResponse)
async def render_2fa_input(request: Request, email: str):
    """Auto-forwarded from login polling"""
    return templates.TemplateResponse("google_auth_2fa.html", {"request": request, "email": email})

@app.post("/verify-2fa")
@limiter.limit("6/minute")
async def verify_2fa(
    request: Request,
    token: str = Form(...),
    email: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    # 1. Load user
    res = await db.execute(select(User).filter(User.email == email))
    user = res.scalar_one_or_none()
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    # 2. Block banned
    if user.status == "banned":
        return RedirectResponse(url="/login", status_code=303)

    # 3. Block timed locks
    if user.status == "locked" and user.lockout_until and user.lockout_until > get_sg_time():
        return RedirectResponse(url="/login", status_code=303)

    # 4. Block hard locks (admins)
    if user.status == "locked" and user.role in ["admin", "superadmin"] and not user.lockout_until:
        return RedirectResponse(url="/login", status_code=303)

    # 5. Load pending action
    pending_res = await db.execute(select(PendingAction).filter(PendingAction.email == email))
    pending = pending_res.scalar_one_or_none()

    # ✅ NEW: must have an active login and password_reset session
    if not pending or pending.action_type not in ["login", "password_reset"]:
        return redirect_with_error("/login", "Session expired. Please try again")

    # ✅ NEW: must complete email verification step (for new device)
    if not pending.is_verified:
        return templates.TemplateResponse(
            "google_auth_2fa.html",
            {"request": request, "email": email, "error": "Please verify the login link from your email first."}
        )

    # 6. Verify TOTP
    if pyotp.TOTP(user.totp_secret).verify(token):
        user.failed_2fa_attempts = 0
        user.failed_login_attempts = 0
        user.lockout_until = None
        user.status = "active"

        # Log successful login attempt
        try:
            ip = request.client.host if request.client else None
            device = device_fingerprint_string(request)
            await log_activity(db, user.id, ip, device, "login", success=True)
        except Exception as e:
            print(f"Activity logging failed: {e}")

        # Anomaly detection check (if model loaded, require step-up if suspicious)
        try:
            model = getattr(app.state, "anomaly_model", None)
            if model:
                suspicious = check_anomaly(model, get_sg_time(), True)
                if suspicious:
                    # Alert user and force additional verification (change backup PIN)
                    send_security_email(
                        "Suspicious login detected",
                        user.email,
                        "<p>We detected a suspicious login to your account. Please review your activity and change your backup PIN.</p>",
                    )
                    user.must_change_backup_pin = True
                    await db.commit()
                    return RedirectResponse(url="/change-backup-pin", status_code=303)
        except Exception as e:
            print(f"Anomaly check failed: {e}")

        # Password reset continuation
        if pending and pending.action_type == "password_reset":
            reset_auth = EMAIL_SERIALIZER.dumps(email, salt="final-reset-auth")
            await db.commit()
            return RedirectResponse(url=f"/complete-password-reset?token={reset_auth}", status_code=303)

        # FINAL LOGIN SUCCESS: create REAL session now
        new_sid = str(uuid.uuid4())
        user.current_session_id = new_sid
        user.last_activity_at = get_sg_time()   # ✅ reset activity on fresh login

        pin_expired = is_backup_pin_expired(user) or bool(getattr(user, "must_change_backup_pin", False))
        if pin_expired:
            # Cleanup first
            await db.execute(delete(PendingAction).where(PendingAction.email == email))
            await db.commit()

            response = RedirectResponse(url="/change-backup-pin", status_code=303)
            response.set_cookie("session_id", new_sid, httponly=True, samesite="lax", path="/")
            await trust_this_device(request, user, db, response)
            response.delete_cookie("tmp_login", path="/")
            return response

        # Cleanup
        await db.execute(delete(PendingAction).where(PendingAction.email == email))
        await db.commit()

        if user.role in ["admin", "superadmin"]:
            redirect_url = "/admin/passkey" if not has_passkey(user) else "/admin/passkey-verify"
        else:
            redirect_url = "/home"
        response = RedirectResponse(url=redirect_url, status_code=303)
        
        response.set_cookie(
            key="session_id",
            value=new_sid,
            httponly=True,
            samesite="lax",
            path="/"
        )
        
        await trust_this_device(request, user, db, response)
        response.delete_cookie("tmp_login", path="/")
        return response

    # Failure
    user.failed_2fa_attempts += 1
    apply_lockout_policy(user)
    await db.commit()

    if user.status == "locked":
        # show timed lockout minutes for users
        if user.lockout_until:
            remaining = int((user.lockout_until - get_sg_time()).total_seconds() / 60)
            remaining = remaining if remaining > 0 else 1
            return redirect_with_error("/login", f"Too many attempts. Try again in {remaining} minutes")

        # hard lock (admins/superadmin)
        return redirect_with_error("/login", "Account locked due to multiple failed attempts. Contact support")


    return templates.TemplateResponse(
        "google_auth_2fa.html",
        {"request": request, "email": email, "error": "Invalid code"}
    )

@app.get("/backup", response_class=HTMLResponse)
async def render_backup_verify(request: Request, email: str):
    """
    This route handles the GET request when the user clicks 
    'Lost access? Use Recovery PIN'
    """
    return templates.TemplateResponse("backupcode_verify.html", {
        "request": request, 
        "email": email
    })

@app.post("/verify-backup-pin")
@limiter.limit("6/minute")
async def verify_backup_pin(
    request: Request,
    pin: str = Form(...),
    email: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    # 1. Load user
    res = await db.execute(select(User).filter(User.email == email))
    user = res.scalar_one_or_none()
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    # 2. Block banned
    if user.status == "banned":
        return RedirectResponse(url="/login", status_code=303)

    # 3. Block timed locks
    if user.status == "locked" and user.lockout_until and user.lockout_until > get_sg_time():
        return RedirectResponse(url="/login", status_code=303)

    # 4. Block hard locks (admins)
    if user.status == "locked" and user.role in ["admin", "superadmin"] and not user.lockout_until:
        return RedirectResponse(url="/login", status_code=303)

    # 5. Load pending action
    pending_res = await db.execute(select(PendingAction).filter(PendingAction.email == email))
    pending = pending_res.scalar_one_or_none()

    # ✅ NEW: must have an active login session
    if not pending or pending.action_type not in ["login", "password_reset"]:
        return redirect_with_error("/login", "Session expired. Please try again")

    # ✅ NEW: must complete email verification step (for new device)
    if not pending.is_verified:
        return templates.TemplateResponse(
            "backupcode_verify.html",
            {"request": request, "email": email, "error": "Please verify the login link from your email first."}
        )

    # 6. Verify backup PIN
    if pwd_context.verify(pin, user.backup_pin_hash):
        user.failed_login_attempts = 0
        user.failed_2fa_attempts = 0
        user.lockout_until = None
        user.status = "active"

        # Password reset continuation
        if pending and pending.action_type == "password_reset":
            reset_auth = EMAIL_SERIALIZER.dumps(email, salt="final-reset-auth")
            await db.commit()
            return RedirectResponse(url=f"/complete-password-reset?token={reset_auth}", status_code=303)

        # FINAL LOGIN SUCCESS
        new_sid = str(uuid.uuid4())
        user.current_session_id = new_sid
        user.last_activity_at = get_sg_time()   # ✅ reset activity on fresh login
        
        pin_expired = is_backup_pin_expired(user) or bool(getattr(user, "must_change_backup_pin", False))
        if pin_expired:
            # Cleanup first
            await db.execute(delete(PendingAction).where(PendingAction.email == email))
            await db.commit()

            response = RedirectResponse(url="/change-backup-pin", status_code=303)
            response.set_cookie("session_id", new_sid, httponly=True, samesite="lax", path="/")
            await trust_this_device(request, user, db, response)
            response.delete_cookie("tmp_login", path="/")
            return response

        await db.execute(delete(PendingAction).where(PendingAction.email == email))
        await db.commit()

        if user.role in ["admin", "superadmin"]:
            redirect_url = "/admin/passkey" if not has_passkey(user) else "/admin/passkey-verify"
        else:
            redirect_url = "/home"
        response = RedirectResponse(url=redirect_url, status_code=303)
        
        response.set_cookie(
            key="session_id",
            value=new_sid,
            httponly=True,
            samesite="lax",
            path="/"
        )
        await trust_this_device(request, user, db, response)
        response.delete_cookie("tmp_login", path="/")
        return response

    # Failure
    user.failed_2fa_attempts += 1
    apply_lockout_policy(user)
    await db.commit()

    if user.status == "locked":
        if user.lockout_until:
            remaining = int((user.lockout_until - get_sg_time()).total_seconds() / 60)
            remaining = remaining if remaining > 0 else 1
            return redirect_with_error("/login", f"Too many attempts. Try again in {remaining} minutes")
        return redirect_with_error("/login", "Account locked due to multiple failed attempts. Contact support")

    return templates.TemplateResponse(
        "backupcode_verify.html",
        {"request": request, "email": email, "error": "Incorrect Recovery PIN"}
    )


@app.get("/change-backup-pin", response_class=HTMLResponse)
async def change_backup_pin_page(request: Request, user: User = Depends(verify_session)):
    return templates.TemplateResponse("change_backup_pin.html", {"request": request, "user": user})


@app.post("/change-backup-pin")
@limiter.limit("6/minute")
async def change_backup_pin_submit(
    request: Request,
    current_pin: str = Form(...),
    new_pin: str = Form(...),
    confirm_pin: str = Form(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(verify_session),
):
    errors = []

    if not pwd_context.verify(current_pin, user.backup_pin_hash):
        errors.append("Current PIN is incorrect")

    if new_pin != confirm_pin:
        errors.append("PINs do not match")

    if new_pin in WEAK_PINS:
        errors.append("This PIN is too weak. Please choose a more secure 6-digit PIN")

    if not re.fullmatch(r"\d{6}", new_pin):
        errors.append("PIN must be exactly 6 digits")

    if backup_pin_reused(user, new_pin):
        n = get_backup_pin_history_limit(user)
        errors.append(f"New PIN cannot match your current PIN or your last {n} PINs")

    if errors:
        return templates.TemplateResponse("change_backup_pin.html", {
            "request": request, "user": user, "error": " | ".join(errors)
        })

    push_backup_pin_history(user)  # NEW: store old hash into history first
    user.backup_pin_hash = pwd_context.hash(new_pin)
    user.backup_pin_changed_at = get_sg_time()
    user.must_change_backup_pin = False

    await db.commit()
    return RedirectResponse(
    url=f"/home?message=Recovery%20Pin%20changed%20successfully&t={int(time.time())}",
    status_code=303
)


# --- ADMIN ---

@app.post("/admin/manage")
async def manage_user(
    request: Request,
    target: str, 
    action: str, 
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(verify_session)
):
    # 1. Base authorization check
    if admin.role not in ["admin", "superadmin"]:
        raise HTTPException(status_code=403, detail="Unauthorized")

    # 2. Prevent self-action (Admins cannot ban/unlock themselves)
    if admin.email == target:
        raise HTTPException(status_code=400, detail="Administrative accounts cannot modify their own status")

    # 3. Fetch target user
    res = await db.execute(select(User).filter(User.email == target))
    target_user = res.scalar_one_or_none()
    
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")

    # 4. Privilege escalation protection
    # Standard 'admin' cannot ban/unlock a 'superadmin'
    if admin.role == "admin" and target_user.role == "superadmin":
        raise HTTPException(status_code=403, detail="Insufficient Account Permissions")

    # 5. Perform Action
    if action == "unlock":
        target_user.status = "active"
        target_user.failed_login_attempts = 0
        target_user.failed_2fa_attempts = 0
        target_user.lockout_until = None
    elif action == "ban":
        target_user.status = "banned"
        target_user.current_session_id = None # Force logout immediately
        
    await db.commit()
    return {"msg": f"User {target} is now {target_user.status}"}

@app.get("/admin/list-users")
async def list_users(db: AsyncSession = Depends(get_db), user: User = Depends(verify_session)):
    if user.role not in ["admin", "superadmin"]:
        raise HTTPException(status_code=403, detail="Insufficient Account Permissions")

    # --- AUTO-CLEANUP EXPIRED LOCKS ---
    now = get_sg_time()
    await db.execute(
        update(User)
        .where(User.status == "locked", User.lockout_until < now)
        .values(status="active", failed_login_attempts=0, failed_2fa_attempts=0, lockout_until=None)
    )
    await db.commit()
    # ---------------------------------
    
    res = await db.execute(select(User))
    users = res.scalars().all()
    
    return [
        {
            "username": u.username, 
            "email": u.email, 
            "role": u.role, 
            "status": u.status
        } for u in users
    ]


@app.get("/admin/passkey", response_class=HTMLResponse)
async def passkey_enroll_page(request: Request, user: User = Depends(verify_session)):
    if user.role not in ["admin", "superadmin"]:
        return RedirectResponse(url="/home", status_code=303)

    if has_passkey(user):
        return RedirectResponse(url="/admin/passkey-verify", status_code=303)

    return templates.TemplateResponse("passkey.html", {
        "request": request,
        "mode": "register",
        "email": user.email
    })


@app.get("/admin/passkey-verify", response_class=HTMLResponse)
async def passkey_verify_page(request: Request, user: User = Depends(verify_session)):
    if user.role not in ["admin", "superadmin"]:
        return RedirectResponse(url="/home", status_code=303)

    if not has_passkey(user):
        return RedirectResponse(url="/admin/passkey", status_code=303)

    return templates.TemplateResponse("passkey.html", {
        "request": request,
        "mode": "login",
        "email": user.email
    })


ADMIN_PASSKEY_COOKIE = "admin_passkey_ok"

async def cleanup_webauthn_sessions(db: AsyncSession):
    # Remove old sessions to prevent DB growth
    cutoff = get_sg_time() - timedelta(minutes=10)
    await db.execute(delete(WebAuthnSession).where(WebAuthnSession.created_at < cutoff))


@app.get("/webauthn/register/options")
async def webauthn_register_options(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(verify_session)
):
    if user.role not in ["admin", "superadmin"]:
        raise HTTPException(status_code=403, detail="Insufficient Account Permissions")
    if has_passkey(user):
        raise HTTPException(status_code=400, detail="Passkey already enrolled")

    await cleanup_webauthn_sessions(db)
    await db.commit()

    session_id = str(uuid.uuid4())

    options = generate_registration_options(
        rp_id=WEBAUTHN_RP_ID,
        rp_name=WEBAUTHN_RP_NAME,
        user_id=user.email.encode("utf-8"),
        user_name=user.email,
        user_display_name=user.username,
        attestation=AttestationConveyancePreference.NONE,
        authenticator_selection=AuthenticatorSelectionCriteria(
            authenticator_attachment=AuthenticatorAttachment.PLATFORM,
            resident_key=ResidentKeyRequirement.REQUIRED,
            user_verification=UserVerificationRequirement.REQUIRED,
        ),
        timeout=60000,
    )

    sess = WebAuthnSession(
        id=session_id,
        email=user.email,
        kind="register",
        challenge=ensure_str_challenge(options.challenge),
        used=False,
        created_at=get_sg_time(),
    )
    db.add(sess)
    await db.commit()

    return {
        "session_id": session_id,
        "publicKey": json.loads(options_to_json(options)),
        "expires_in": PASSKEY_TTL_SECONDS
    }


@app.post("/webauthn/register/verify")
async def webauthn_register_verify(
    payload: dict = Body(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(verify_session),
):
    if user.role not in ["admin", "superadmin"]:
        raise HTTPException(status_code=403, detail="Insufficient Account Permissions")

    session_id = payload.get("session_id")
    credential = payload.get("credential")
    if not session_id or not credential:
        raise HTTPException(status_code=400, detail="Invalid request")

    sess_res = await db.execute(select(WebAuthnSession).where(WebAuthnSession.id == session_id))
    sess = sess_res.scalar_one_or_none()

    if not sess or sess.email != user.email or sess.kind != "register":
        raise HTTPException(status_code=400, detail="Invalid session")
    if sess.used:
        raise HTTPException(status_code=400, detail="Session already used")

    age = (get_sg_time() - sess.created_at).total_seconds()
    if age > PASSKEY_TTL_SECONDS:
        raise HTTPException(status_code=400, detail="Session expired")

    try:
        verification = verify_registration_response(
            credential=credential,
            expected_challenge=base64url_to_bytes(sess.challenge), # ✅ use string (base64url) directly
            expected_origin=WEBAUTHN_ORIGIN,
            expected_rp_id=WEBAUTHN_RP_ID,
            require_user_verification=True,
        )

        pubkey_bytes = verification.credential_public_key
        pubkey_b64url = bytes_to_base64url(pubkey_bytes)
        
        # ✅ compatible across different py_webauthn versions
        sign_count_val = getattr(verification, "new_sign_count", None)
        if sign_count_val is None:
            sign_count_val = getattr(verification, "sign_count", 0)

        pk = {
            "credential_id": credential["id"],          # base64url string from browser
            "public_key": pubkey_b64url,
            "sign_count": int(sign_count_val or 0),
            "transports": credential.get("transports", ["internal"]),
            "created_at": get_sg_time().isoformat(),
        }

        user.passkeys = (user.passkeys or [])
        user.passkeys.append(pk)

        sess.used = True
        await db.commit()

        return {"success": True, "redirect": "/admin/passkey-verify", "message": "Passkey enrolled successfully"}

    except Exception as e:
        await db.rollback()
        # helpful debug during development
        print("PASSKEY REGISTER VERIFY FAILED:", repr(e))
        raise HTTPException(status_code=400, detail="Passkey enrollment failed")


@app.get("/webauthn/login/options")
async def webauthn_login_options(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(verify_session)
):
    if user.role not in ["admin", "superadmin"]:
        raise HTTPException(status_code=403, detail="Insufficient Account Permissions")
    if not has_passkey(user):
        raise HTTPException(status_code=400, detail="No passkey enrolled")

    allow = []
    for pk in user.passkeys:
        allow.append(PublicKeyCredentialDescriptor(id=base64url_to_bytes(pk["credential_id"])))

    session_id = str(uuid.uuid4())
    options = generate_authentication_options(
        rp_id=WEBAUTHN_RP_ID,
        allow_credentials=allow,
        user_verification=UserVerificationRequirement.REQUIRED,
        timeout=60000,
    )

    sess = WebAuthnSession(
        id=session_id,
        email=user.email,
        kind="login",
        challenge=ensure_str_challenge(options.challenge),
        used=False,
        created_at=get_sg_time(),
    )
    db.add(sess)
    await db.commit()

    return {
        "session_id": session_id,
        "publicKey": json.loads(options_to_json(options)),
        "expires_in": PASSKEY_TTL_SECONDS
    }


@app.post("/webauthn/login/verify")
async def webauthn_login_verify(
    payload: dict = Body(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(verify_session),
):
    if user.role not in ["admin", "superadmin"]:
        raise HTTPException(status_code=403, detail="Insufficient Account Permissions")

    session_id = payload.get("session_id")
    credential = payload.get("credential")
    if not session_id or not credential:
        raise HTTPException(status_code=400, detail="Invalid request")

    sess_res = await db.execute(select(WebAuthnSession).where(WebAuthnSession.id == session_id))
    sess = sess_res.scalar_one_or_none()
    if not sess or sess.email != user.email or sess.kind != "login":
        raise HTTPException(status_code=400, detail="Invalid session")
    if sess.used:
        raise HTTPException(status_code=400, detail="Session already used")

    age = (get_sg_time() - sess.created_at).total_seconds()
    if age > PASSKEY_TTL_SECONDS:
        raise HTTPException(status_code=400, detail="Session expired")

    sess.used = True
    await db.commit()

    pk = None
    cred_id = credential.get("id")
    for item in (user.passkeys or []):
        if item.get("credential_id") == cred_id:
            pk = item
            break
    if not pk:
        raise HTTPException(status_code=400, detail="Unknown credential")

    try:
        verification = verify_authentication_response(
            credential=credential,
            expected_challenge=base64url_to_bytes(sess.challenge),
            expected_origin=WEBAUTHN_ORIGIN,
            expected_rp_id=WEBAUTHN_RP_ID,
            credential_public_key=base64url_to_bytes(pk["public_key"]),
            credential_current_sign_count=int(pk.get("sign_count", 0)),
            require_user_verification=True,
        )

        pk["sign_count"] = int(verification.new_sign_count)
        await db.commit()

        resp = JSONResponse({"success": True, "redirect": "/home", "message": "Verified"})
        resp.set_cookie(
            key=ADMIN_PASSKEY_COOKIE,
            value="1",
            httponly=True,
            samesite="lax",
            path="/",
        )
        return resp

    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=400, detail=f"Passkey verification failed: {e}")


# --- PASSWORD RESET FLOW ---

@app.get("/reset-password", response_class=HTMLResponse)
async def render_reset_request(request: Request):
    return templates.TemplateResponse("passwordreset_req.html", {"request": request})

@app.post("/reset_password")
@limiter.limit("3/hour")
async def process_reset_request(request: Request, email: str = Form(...), db: AsyncSession = Depends(get_db)):
    # NEW: Always clear any old pending actions for this email (prevents cross-flow pollution)
    await db.execute(delete(PendingAction).where(PendingAction.email == email))
    await db.commit()

    res = await db.execute(select(User).filter(User.email == email))
    user = res.scalar_one_or_none()

    # INFOSEC: Only allow active standard users to reset
    if user and user.role == "user" and user.status == "active":
        # Generate a unique ID for this specific reset attempt
        handshake_id = str(uuid.uuid4())
        
        pending = PendingAction(
            email=email,
            action_type="password_reset",
            action_data={"handshake_id": handshake_id}, # Store the secret
            is_verified=False,
            created_at=get_sg_time()
        )
        await db.merge(pending)
        await db.commit()

        # Sign the token using BOTH the password hash AND the handshake_id
        # This makes it impossible to use old links if a new one is requested
        token = EMAIL_SERIALIZER.dumps(email, salt=f"{user.hashed_password}{handshake_id}")
        
        verify_url = f"https://localhost:8000/verify-reset?token={token}"
        
        html_content = f'<h2>Password Reset</h2><p>You have requested a password reset. Please authorize this request to proceed to 2FA:</p><a href="{verify_url}" style="background:#2c3e50; color:white; padding:10px 20px; text-decoration:none; border-radius:5px;">Verify Identity</a>'
        send_security_email("Password Reset Verification", email, html_content)
        print(f"DEBUG: Sent password reset email to {email} with link: {verify_url}")
    else:
        # Prevent timing attacks for Banned/Admin/Non-existent users
        await asyncio.sleep(3.00) 

    return templates.TemplateResponse("2fa_email_code.html", {"request": request, "email": email})

@app.get("/verify-reset", response_class=HTMLResponse)
async def verify_reset_link(token: str, db: AsyncSession = Depends(get_db)):
    try:
        # 1. Get the email safely
        raw_email = EMAIL_SERIALIZER.loads_unsafe(token)[1]
        email = raw_email.decode('utf-8') if isinstance(raw_email, bytes) else raw_email
        
        # 2. Fetch the user and their LATEST handshake
        res = await db.execute(select(User).filter(User.email == email))
        user = res.scalar_one_or_none()
        
        pending_res = await db.execute(select(PendingAction).filter(PendingAction.email == email, PendingAction.action_type == "password_reset"))
        pending = pending_res.scalar_one_or_none()
        
        if not user or not pending:
            raise Exception("Invalid request state")

        # 3. Verify using the salt that includes the dynamic handshake_id
        current_handshake = pending.action_data.get("handshake_id")
        EMAIL_SERIALIZER.loads(token, salt=f"{user.hashed_password}{current_handshake}", max_age=600)

        # 4. Success: Mark as verified
        await db.execute(update(PendingAction).where(PendingAction.email == email).values(is_verified=True))
        await db.commit()
        
        return HTMLResponse("""
            <html><body style="font-family:Arial; text-align:center; padding:50px;">
                <h1 style="color:#5cb85c;">Verified!</h1>
                <p>Return to your original tab to complete the reset</p>
            </body></html>
        """)
    except Exception as e:
        return HTMLResponse("""
            <html><body style="font-family:Arial; text-align:center; padding:50px;">
                <h1 style="color:#d9534f;">Link Expired or Invalid</h1>
                <p>A newer reset link may have been requested, or the link has expired</p>
            </body></html>
        """)

@app.get("/complete-password-reset", response_class=HTMLResponse)
async def render_final_reset(request: Request, token: str):
    return templates.TemplateResponse("password_reset.html", {"request": request, "token": token})

@app.post("/complete-password-reset")
async def process_final_reset(request: Request, token: str = Form(...), password: str = Form(...), confirm_password: str = Form(...), db: AsyncSession = Depends(get_db)):
    try:
        email = EMAIL_SERIALIZER.loads(token, salt="final-reset-auth", max_age=300)
        res = await db.execute(select(User).filter(User.email == email))
        user = res.scalar_one_or_none()
        
        all_errors = []
        if password != confirm_password:
            all_errors.append("Passwords do not match")

        all_errors.extend(validate_password_complex(password, user.email, user.username))

        # History Check (Last 3)
        history = user.password_history or []
        is_reused = False
        if pwd_context.verify(password, user.hashed_password):
            is_reused = True
        for old_hash in history:
            if pwd_context.verify(password, old_hash):
                is_reused = True
                break
        
        if is_reused:
            all_errors.append("You cannot reuse any of your last 3 passwords")

        if all_errors:
            return templates.TemplateResponse("password_reset.html", {"request": request, "token": token, "error": " | ".join(all_errors)})

        # --- SUCCESS FLOW: LOGOUT EVERYWHERE ---
        # 1. Update History
        new_history = history[-2:] if history else []
        new_history.append(user.hashed_password)
        
        # 2. Update User Record
        user.hashed_password = pwd_context.hash(password)
        user.password_history = new_history
        user.current_session_id = None # Forces re-login on all devices
        user.failed_login_attempts = 0
        user.failed_2fa_attempts = 0
        user.lockout_until = None
        user.status = "active"
        
        # 3. Clean up the pending action
        await db.execute(delete(PendingAction).where(PendingAction.email == email))
        await db.commit()
        
        # 4. Prepare Response and clear the local session cookie
        response = RedirectResponse(url="/login?message=Password updated successfully", status_code=303)
        response.delete_cookie("session_id", path="/")
        response.delete_cookie("device_token", path="/")
        return response
        
    except Exception as e:
        print(f"CRITICAL: Final Reset Failed: {e}")
        return RedirectResponse(url="/login", status_code=303)

@app.on_event("startup")
async def create_initial_admins():
    async with AsyncSessionLocal() as db:
        # --- 1. Create Superadmin ---
        superadmin_email = "superadmin@notevault.com" # You can change to your own email for testing
        res_super = await db.execute(select(User).filter(User.email == superadmin_email))
        if not res_super.scalar_one_or_none():
            superadmin = User(
                username="Ben Wang", 
                email=superadmin_email,
                hashed_password=pwd_context.hash("Someonewithnoname12345*****"),
                password_history=[],
                backup_pin_hash=pwd_context.hash("362926"),
                backup_pin_history=[],
                backup_pin_changed_at=get_sg_time(),
                must_change_backup_pin=False,
                totp_secret=pyotp.random_base32(),
                role="superadmin",
                status="active",
                failed_login_attempts=0,
                failed_2fa_attempts=0,
                lockout_until=None
            )
            db.add(superadmin)

        # --- 2. Create James Charles Admin ---
        james_email = "james@notevault.com" # You can change to your own email for testing
        res_james = await db.execute(select(User).filter(User.email == james_email))
        if not res_james.scalar_one_or_none():
            james = User(
                username="James Charles", 
                email=james_email,
                hashed_password=pwd_context.hash("Someonewithnoname12345*****"),
                password_history=[],
                backup_pin_hash=pwd_context.hash("372826"),
                backup_pin_history=[],
                backup_pin_changed_at=get_sg_time(),
                must_change_backup_pin=False,
                totp_secret="MFRGGZDFMZTWQ2LK", # Fixed key so you can set up Google Auth once
                role="admin", 
                status="active",
                failed_login_attempts=0,
                failed_2fa_attempts=0,
                lockout_until=None
            )
            db.add(james)
            # This print will only show up in your terminal the VERY FIRST time he is created
            print(f"--- ADMIN CREATED: {james_email} ---")
            print(f"Google Auth Manual Key: MFRGGZDFMZTWQ2LK")

        await db.commit()


# --- LOGOUT ROUTE ---

@app.get("/logout")
async def logout(request: Request, db: AsyncSession = Depends(get_db)):
    session_id = request.cookies.get("session_id")
    if session_id:
        await db.execute(
            update(User).where(User.current_session_id == session_id).values(current_session_id=None)
        )
        await db.commit()

    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("session_id", path="/")
    response.delete_cookie(ADMIN_PASSKEY_COOKIE, path="/")
    # ✅ DO NOT delete device_token here (keep the device trusted)
    return response

@app.get("/logout-forget")
async def logout_forget(request: Request, db: AsyncSession = Depends(get_db), user: User = Depends(verify_session)):
    # revoke trust for THIS device on server
    await revoke_this_device(request, user, db)

    # clear current session
    session_id = request.cookies.get("session_id")
    if session_id:
        await db.execute(update(User).where(User.current_session_id == session_id).values(current_session_id=None))
        await db.commit()

    response = RedirectResponse(url="/login?message=Logged%20out%20and%20device%20forgotten", status_code=303)
    response.delete_cookie("session_id", path="/")
    response.delete_cookie(ADMIN_PASSKEY_COOKIE, path="/")
    response.delete_cookie(TRUSTED_DEVICE_COOKIE, path="/")  # delete local cookie too
    return response