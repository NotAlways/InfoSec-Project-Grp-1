"""
file_security.py - NoteVault Secure File Ingest Module
=======================================================
Responsibilities:
  1. MIME-type validation via magic bytes (not just extension spoofing)
  2. File-size enforcement
  3. Malware pattern scan (known bad byte signatures)
  4. Optional ClamAV integration (graceful fallback when daemon absent)
  5. SHA-256 integrity hash generation

Integrates with: main.py /files/upload route
"""

import hashlib
import os
from typing import Tuple

# ---------------------------------------------------------------------------
# Policy constants
# ---------------------------------------------------------------------------

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB hard limit (Increase the first value if necessary for file submission)

# Allowed MIME types (whitelist – reject everything else)
ALLOWED_MIME_TYPES: set[str] = {
    "application/pdf",
    "text/plain",
    "text/csv",
    # MS-Office Open XML
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    # Images
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
}

# Allowed file extensions (secondary check – must agree with MIME)
ALLOWED_EXTENSIONS: set[str] = {
    ".pdf", ".txt", ".csv",
    ".docx", ".xlsx", ".pptx",
    ".jpg", ".jpeg", ".png", ".gif", ".webp",
}

# ---------------------------------------------------------------------------
# Known-malware byte signatures (magic-byte patterns)
# Checked against first 512 bytes of the file.
# ---------------------------------------------------------------------------
_MALWARE_PATTERNS: list[tuple[bytes, str]] = [
    (b"MZ",             "Windows PE executable (EXE/DLL)"),
    (b"\x7fELF",        "Linux ELF executable"),
    (b"#!/",            "Shell script shebang"),
    (b"#!python",       "Python script"),
    (b"<?php",          "PHP code injection"),
    (b"<script",        "Embedded JavaScript"),
    (b"EICAR-STANDARD", "EICAR antivirus test signature"),
    (b"X5O!P%@AP",      "EICAR test virus string"),
    (b"\xd4\xc3\xb2\xa1", "PCAP network capture"),
    (b"PK\x03\x04",     None),  # ZIP – allowed only when MIME is .docx/.xlsx/.pptx
]

# MIME types that are legitimately ZIP-based (Office Open XML)
_ZIP_BASED_ALLOWED_MIMES: set[str] = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


# ---------------------------------------------------------------------------
# Helper – detect MIME via python-magic (libmagic)
# ---------------------------------------------------------------------------

def _detect_mime(file_bytes: bytes) -> str:
    """Return MIME type detected from actual file bytes (first 4 KB)."""
    try:
        import magic  # python-magic package
        return magic.from_buffer(file_bytes[:4096], mime=True)
    except ImportError:
        # python-magic not installed – fall back to simple header heuristics
        return _heuristic_mime(file_bytes)
    except Exception:
        return "application/octet-stream"


def _heuristic_mime(data: bytes) -> str:
    """Minimal byte-header MIME heuristic when python-magic is unavailable."""
    h = data[:8]
    if h[:4] == b"%PDF":
        return "application/pdf"
    if h[:2] in (b"\xff\xd8", b"\xff\xe0", b"\xff\xe1"):
        return "image/jpeg"
    if h[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if h[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if h[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if h[:4] == b"PK\x03\x04":
        return "application/zip"
    return "application/octet-stream"


# ---------------------------------------------------------------------------
# Individual validation functions
# ---------------------------------------------------------------------------

def validate_file_size(size_bytes: int) -> Tuple[bool, str]:
    """Enforce max file size and reject empty files."""
    if size_bytes == 0:
        return False, "File is empty"
    if size_bytes > MAX_FILE_SIZE_BYTES:
        mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
        return False, f"File exceeds the {mb} MB limit ({size_bytes:,} bytes received)"
    return True, "OK"


def validate_extension(filename: str) -> Tuple[bool, str]:
    """Check file extension against the whitelist."""
    _, ext = os.path.splitext(filename.lower())
    if not ext:
        return False, "File has no extension"
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Extension '{ext}' is not permitted. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
    return True, ext


def validate_mime_type(file_bytes: bytes, filename: str) -> Tuple[bool, str]:
    """
    Dual validation:
      a) Detect actual MIME via magic bytes (prevents extension spoofing)
      b) Ensure detected MIME is in the whitelist
    """
    detected = _detect_mime(file_bytes)
    if detected not in ALLOWED_MIME_TYPES:
        return False, f"Detected file type '{detected}' is not permitted"

    # Cross-check extension vs detected MIME (catch renamed executables)
    _, ext = os.path.splitext(filename.lower())
    _ext_to_mime = {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".csv": "text/csv",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    expected = _ext_to_mime.get(ext)
    if expected and detected != expected:
        return False, (
            f"Extension/content mismatch: extension '{ext}' implies {expected} "
            f"but file content is {detected}"
        )

    return True, detected


def scan_malware_patterns(file_bytes: bytes, detected_mime: str = "") -> Tuple[bool, str]:
    """
    Check the first 512 bytes against known malicious byte signatures.
    ZIP (PK\x03\x04) is allowed only for Office Open XML MIME types.
    """
    header = file_bytes[:512]
    for pattern, label in _MALWARE_PATTERNS:
        if pattern not in header:
            continue
        # ZIP header is legitimate for Office Open XML
        if pattern == b"PK\x03\x04" and detected_mime in _ZIP_BASED_ALLOWED_MIMES:
            continue
        reason = label or f"Disallowed binary signature: {pattern.hex()}"
        return False, f"Malware pattern detected – {reason}"

    # Scan for script injections inside text files
    if detected_mime in ("text/plain", "text/csv"):
        lower = file_bytes[:2048].lower()
        for injection in (b"<script", b"<?php", b"javascript:", b"vbscript:"):
            if injection in lower:
                return False, f"Script injection pattern detected in text content"

    return True, "Pattern scan clean"


def _try_clamav_scan(file_bytes: bytes) -> Tuple[bool | None, str]:
    """
    Optional ClamAV scan via clamd Unix socket.
    Returns:
      (True,  reason)  – clean
      (False, reason)  – threat found
      (None,  reason)  – ClamAV unavailable (soft failure)
    """
    try:
        import clamd
        from io import BytesIO
        cd = clamd.ClamdUnixSocket()
        result = cd.instream(BytesIO(file_bytes))
        stream_result = result.get("stream", ("OK", ""))
        status, detail = stream_result[0], stream_result[1] if len(stream_result) > 1 else ""
        if status == "OK":
            return True, "ClamAV: clean"
        return False, f"ClamAV threat: {detail}"
    except Exception as exc:
        return None, f"ClamAV unavailable ({type(exc).__name__}) – relying on pattern scan"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def full_security_scan(file_bytes: bytes, filename: str) -> Tuple[bool, str]:
    """
    Run all security checks in order.  Short-circuits on first failure.

    Pipeline:
      1. Size validation
      2. Extension whitelist check
      3. MIME-type detection & whitelist check
      4. Malware byte-pattern scan
      5. ClamAV scan (optional, graceful fallback)

    Returns:
      (True,  "All security checks passed")   – safe to store
      (False, "<reason>")                     – reject file
    """
    # 1 – size
    ok, msg = validate_file_size(len(file_bytes))
    if not ok:
        return False, msg

    # 2 – extension
    ok, ext_or_msg = validate_extension(filename)
    if not ok:
        return False, ext_or_msg

    # 3 – MIME
    ok, mime_or_msg = validate_mime_type(file_bytes, filename)
    if not ok:
        return False, mime_or_msg
    detected_mime = mime_or_msg  # validated MIME string

    # 4 – malware patterns
    ok, msg = scan_malware_patterns(file_bytes, detected_mime)
    if not ok:
        return False, msg

    # 5 – ClamAV (non-blocking if unavailable)
    av_ok, av_msg = _try_clamav_scan(file_bytes)
    if av_ok is False:          # explicitly flagged as threat (None = unavailable)
        return False, av_msg
    if av_ok is None:
        print(f"[file_security] {av_msg}")  # log soft failure

    return True, "All security checks passed"


def compute_sha256(data: bytes) -> str:
    """Return hex-encoded SHA-256 digest of the given bytes."""
    return hashlib.sha256(data).hexdigest()
