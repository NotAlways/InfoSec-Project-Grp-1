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
from .crypto import load_key, encrypt_content, decrypt_content
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
DATABASE_URL = "postgresql+asyncpg://postgres:1m1f1b1m@localhost/notevault"
Base = declarative_base()

# Load encryption key on startup
encryption_key = None

def get_encryption_key():
    global encryption_key
    if encryption_key is None:
        encryption_key = load_key()
    return encryption_key

# Models
class Note(Base):
    __tablename__ = "notes"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content = Column(String)
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

# FastAPI app
app = FastAPI(title="NoteVault API")
app.state.limiter = limiter
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

# Routes
@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.post("/notes", response_model=NoteResponse)
async def create_note(note: NoteSchema, db: AsyncSession = Depends(get_db)):
    key = get_encryption_key()
    encrypted_content = encrypt_content(note.content, key)
    new_note = Note(title=note.title, content=encrypted_content)
    db.add(new_note)
    await db.commit()
    await db.refresh(new_note)
    return new_note

@app.get("/notes", response_model=list[NoteResponse])
async def get_notes():
    from sqlalchemy import select
    key = get_encryption_key()
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Note))
        notes = result.scalars().all()
        # Decrypt content for each note
        for note in notes:
            note.content = decrypt_content(note.content, key)
        return notes

@app.get("/notes/{note_id}", response_model=NoteResponse)
async def get_note(note_id: int):
    from sqlalchemy import select
    key = get_encryption_key()
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Note).filter(Note.id == note_id))
        note = result.scalar_one_or_none()
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")
        note.content = decrypt_content(note.content, key)
        return note

@app.put("/notes/{note_id}", response_model=NoteResponse)
async def update_note(note_id: int, note: NoteSchema):
    from sqlalchemy import select
    key = get_encryption_key()
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Note).filter(Note.id == note_id))
        db_note = result.scalar_one_or_none()
        if not db_note:
            raise HTTPException(status_code=404, detail="Note not found")
        db_note.title = note.title
        db_note.content = encrypt_content(note.content, key)
        db_note.updated_at = datetime.utcnow()
        await session.commit()
        await session.refresh(db_note)
        # Decrypt for response
        db_note.content = decrypt_content(db_note.content, key)
        return db_note

@app.delete("/notes/{note_id}")
async def delete_note(note_id: int):
    from sqlalchemy import select
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Note).filter(Note.id == note_id))
        db_note = result.scalar_one_or_none()
        if not db_note:
            raise HTTPException(status_code=404, detail="Note not found")
        await session.delete(db_note)
        await session.commit()
        return {"message": "Note deleted"}



# --- from here onwards Prathip's code ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
EMAIL_SERIALIZER = URLSafeTimedSerializer("EMAIL_TOKEN_SECRET_KEY")
WEAK_PINS = ["123456", "000000", "111111", "222222", "333333", "444444", "555555", "666666", "777777", "888888", "999999", "654321"]

# Path adjustment for your directory structure
templates = Jinja2Templates(directory="frontend/templates")
app.mount("/static", StaticFiles(directory="frontend"), name="static")

def get_sg_time():
    """Helper function to get current time in GMT+8 (Naive for DB compatibility)"""
    # 1. Get time with TZ
    sg_time = datetime.now(timezone(timedelta(hours=8)))
    # 2. Remove the TZ info so SQLAlchemy doesn't crash
    return sg_time.replace(tzinfo=None)


def ensure_str_challenge(challenge) -> str:
    """
    webauthn may give challenge as bytes or str depending on version.
    DB column is VARCHAR, so we always store base64url STRING.
    """
    if isinstance(challenge, bytes):
        return bytes_to_base64url(challenge)  # from webauthn.helpers
    return str(challenge)


def redirect_with_error(path: str, msg: str) -> RedirectResponse:
    return RedirectResponse(url=f"{path}?error={quote(msg)}", status_code=303)


# --- WebAuthn / Passkeys (Platform Biometrics for Admins) ---
WEBAUTHN_RP_ID = "localhost"                 # must match your domain
WEBAUTHN_RP_NAME = "NoteVault"
WEBAUTHN_ORIGIN = "http://localhost:8000"    # MUST match your browser URL exactly
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


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    password_history = Column(JSON, default=list)
    totp_secret = Column(String)
    backup_pin_hash = Column(String)
    role = Column(String, default="user")
    status = Column(String, default="active")
    failed_login_attempts = Column(Integer, default=0)
    failed_2fa_attempts = Column(Integer, default=0) # Track 2FA/PIN separately
    lockout_until = Column(DateTime, nullable=True)  # Progressive timer
    current_session_id = Column(String, nullable=True)
    device_key = Column(String, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=get_sg_time)
    passkeys = Column(JSON, default=list)


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


def has_passkey(user: "User") -> bool:
    return bool(user.passkeys and isinstance(user.passkeys, list) and len(user.passkeys) > 0)


async def verify_session(request: Request, db: AsyncSession = Depends(get_db)):
    session_id = request.cookies.get("session_id")
    if not session_id:
        raise HTTPException(status_code=303)

    # Verify ID exists in DB
    res = await db.execute(select(User).filter(User.current_session_id == session_id))
    user = res.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=303)
    
    return user

@app.exception_handler(HTTPException)
async def auth_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 303:
        return RedirectResponse(url="/login")
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

@app.on_event("startup")
async def startup_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Dashboard
@app.get("/home", name="dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: User = Depends(verify_session)):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": user, 
        "message": f"Welcome back, {user.username}"
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
    verify_url = f"http://localhost:8000/verify-registration?token={token}"
    
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
        totp_secret=secret,
        role="user",
        status="active",
        current_session_id=session_id # Log them in immediately
    )
    
    db.add(new_user)
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
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Incorrect email or password"}
        )

    # 6. Device Recognition Check
    device_token = request.cookies.get("device_token")
    is_recognized_device = False

    if device_token:
        try:
            token_email = EMAIL_SERIALIZER.loads(
                device_token,
                salt=user.device_key,
                max_age=60 * 60 * 24 * 30
            )
            if token_email == email:
                is_recognized_device = True
        except:
            is_recognized_device = False

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
        verify_url = f"http://localhost:8000/verify-login?token={token}"
        html_body = (
            f"<h2>New Device Login</h2>"
            f"<p>We detected a login from a new device. If this is you, authorize this session:</p>"
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
            <p>You may close this tab and return to your login window.</p>
        </body></html>
        """)

    except Exception as e:
        print("DEBUG: Verification Error:", e)
        return HTMLResponse("""
        <html><body style="font-family:Arial; text-align:center; padding:50px;">
            <h1 style="color:#d9534f;">Link Expired or Invalid</h1>
            <p>A newer login link may have been requested, or the link has expired.</p>
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

    # For normal login, we REQUIRE a pending record and it must be verified (email step)
    if pending and pending.action_type == "login":
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

        # Password reset continuation
        if pending and pending.action_type == "password_reset":
            reset_auth = EMAIL_SERIALIZER.dumps(email, salt="final-reset-auth")
            await db.commit()
            return RedirectResponse(url=f"/complete-password-reset?token={reset_auth}", status_code=303)

        # FINAL LOGIN SUCCESS: create REAL session now
        new_sid = str(uuid.uuid4())
        user.current_session_id = new_sid

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

    # For normal login, REQUIRE pending login + verified
    if pending and pending.action_type == "login":
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

# --- ADMIN ---

@app.post("/admin/manage")
async def manage_user(
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
        
        verify_url = f"http://localhost:8000/verify-reset?token={token}"
        
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
        await db.execute(update(User).where(User.current_session_id == session_id).values(current_session_id=None))
        await db.commit()

        response = RedirectResponse(url="/login", status_code=303)
        response.delete_cookie("session_id", path="/")
        response.delete_cookie(ADMIN_PASSKEY_COOKIE, path="/")  # ✅ important
        return response
