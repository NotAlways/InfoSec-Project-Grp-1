"""
AES-256-GCM encryption utility for NoteVault.
Keys are stored locally outside the project directory and never hardcoded.
"""

import os
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import base64
import secrets

# Key storage location - outside project directory
KEY_DIR = Path.home() / ".notevault"
KEY_FILE = KEY_DIR / "encryption.key"

def ensure_key_dir():
    """Ensure the key directory exists with restricted permissions."""
    KEY_DIR.mkdir(mode=0o700, exist_ok=True)

def generate_key() -> bytes:
    """
    Generate a new AES-256 key (32 bytes).
    Returns the raw key bytes.
    """
    return AESGCM.generate_key(bit_length=256)

def save_key(key: bytes) -> None:
    """
    Save the encryption key to a file outside the project directory.
    Uses restricted file permissions (0o600) for security.
    """
    ensure_key_dir()
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    # Restrict file permissions
    os.chmod(KEY_FILE, 0o600)
    print(f"✓ Encryption key saved to: {KEY_FILE}")

def load_key() -> bytes:
    """
    Load the encryption key from file.
    Raises FileNotFoundError if key doesn't exist.
    """
    if not KEY_FILE.exists():
        raise FileNotFoundError(
            f"Encryption key not found at {KEY_FILE}. "
            "Run 'python backend/init_crypto.py' to generate one."
        )
    with open(KEY_FILE, "rb") as f:
        return f.read()

def encrypt_content(plaintext: str, key: bytes) -> str:
    """
    Encrypt content using AES-256-GCM.
    
    Args:
        plaintext: The content to encrypt
        key: The encryption key (32 bytes for AES-256)
    
    Returns:
        Base64-encoded ciphertext in format: nonce_b64:ciphertext_b64:tag_b64
    """
    # Generate a random 12-byte nonce
    nonce = secrets.token_bytes(12)
    
    # Create cipher
    cipher = AESGCM(key)
    
    # Encrypt and authenticate
    ciphertext = cipher.encrypt(nonce, plaintext.encode('utf-8'), None)
    
    # ciphertext from AESGCM includes the 16-byte authentication tag at the end
    # Split the tag from ciphertext for cleaner storage
    actual_ciphertext = ciphertext[:-16]
    tag = ciphertext[-16:]
    
    # Encode all components as base64 and combine
    nonce_b64 = base64.b64encode(nonce).decode('utf-8')
    ciphertext_b64 = base64.b64encode(actual_ciphertext).decode('utf-8')
    tag_b64 = base64.b64encode(tag).decode('utf-8')
    
    return f"{nonce_b64}:{ciphertext_b64}:{tag_b64}"

def decrypt_content(encrypted: str, key: bytes) -> str:
    """
    Decrypt AES-256-GCM encrypted content.
    
    Args:
        encrypted: The encrypted content in format: nonce_b64:ciphertext_b64:tag_b64
        key: The encryption key (32 bytes for AES-256)
    
    Returns:
        Decrypted plaintext string
    """
    # If the stored value does not match the expected encrypted format
    # (nonce:ciphertext:tag), assume it's plaintext and return it unchanged.
    if not isinstance(encrypted, str) or ":" not in encrypted:
        return encrypted

    try:
        # Split the encrypted string
        nonce_b64, ciphertext_b64, tag_b64 = encrypted.split(":")

        # Decode from base64
        nonce = base64.b64decode(nonce_b64)
        actual_ciphertext = base64.b64decode(ciphertext_b64)
        tag = base64.b64decode(tag_b64)

        # Reconstruct full ciphertext (ciphertext + tag for AESGCM)
        full_ciphertext = actual_ciphertext + tag

        # Create cipher and decrypt
        cipher = AESGCM(key)
        plaintext = cipher.decrypt(nonce, full_ciphertext, None)

        return plaintext.decode('utf-8')
    except Exception as e:
        # If decryption fails, raise a clear error so callers can handle it.
        raise ValueError(f"Failed to decrypt content: {e}")

def key_exists() -> bool:
    """Check if encryption key exists."""
    return KEY_FILE.exists()


def wrap_key(dek: bytes, master_key: bytes) -> str:
    """
    Wrap (encrypt) a per-item DEK using the master key with AES-256-GCM.
    Returns a base64 `nonce:ciphertext:tag` string.
    """
    nonce = secrets.token_bytes(12)
    cipher = AESGCM(master_key)
    ct = cipher.encrypt(nonce, dek, None)
    actual_ct = ct[:-16]
    tag = ct[-16:]
    nonce_b64 = base64.b64encode(nonce).decode('utf-8')
    ct_b64 = base64.b64encode(actual_ct).decode('utf-8')
    tag_b64 = base64.b64encode(tag).decode('utf-8')
    return f"{nonce_b64}:{ct_b64}:{tag_b64}"


def unwrap_key(wrapped: str, master_key: bytes) -> bytes:
    """
    Unwrap (decrypt) a wrapped DEK string using the master key and return raw DEK bytes.
    """
    if not isinstance(wrapped, str) or ":" not in wrapped:
        raise ValueError("Wrapped key is not in expected format")
    try:
        nonce_b64, ct_b64, tag_b64 = wrapped.split(":")
        nonce = base64.b64decode(nonce_b64)
        actual_ct = base64.b64decode(ct_b64)
        tag = base64.b64decode(tag_b64)
        full_ct = actual_ct + tag
        cipher = AESGCM(master_key)
        dek = cipher.decrypt(nonce, full_ct, None)
        return dek
    except Exception as e:
        raise ValueError(f"Failed to unwrap key: {e}")
