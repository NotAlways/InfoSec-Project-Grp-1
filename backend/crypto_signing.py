import base64
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

def generate_signing_keypair():
    private = ed25519.Ed25519PrivateKey.generate()
    public = private.public_key()

    private_bytes = private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    )

    public_bytes = public.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )

    return private_bytes, public_bytes

def sign_data(private_key_bytes: bytes, data: bytes) -> bytes:
    private = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    return private.sign(data)

def verify_signature(public_key_bytes: bytes, signature: bytes, data: bytes):
    public = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)
    public.verify(signature, data)

def b64e(data: bytes) -> str:
    return base64.b64encode(data).decode()

def b64d(data: str) -> bytes:
    return base64.b64decode(data.encode())
