import base64
import hashlib
import hmac
import json
import os
import time

_AUDIENCE = "exam-ia"
_LEEWAY = 30


def service_secret() -> str:
    return os.environ.get("SERVICE_JWT_SECRET", "")


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def rejection_reason(token: str, secret: str):
    if not token:
        return "no_token"
    try:
        header_segment, payload_segment, signature_segment = token.split(".")
    except ValueError:
        return "malformed"

    try:
        header = json.loads(_b64url_decode(header_segment))
    except Exception:
        return "bad_header"
    if header.get("alg") != "HS256":
        return "bad_alg"

    key = hashlib.sha256(secret.encode("utf-8")).digest()
    signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
    expected = hmac.new(key, signing_input, hashlib.sha256).digest()
    try:
        if not hmac.compare_digest(expected, _b64url_decode(signature_segment)):
            return "bad_signature"
    except Exception:
        return "bad_signature"

    try:
        payload = json.loads(_b64url_decode(payload_segment))
    except Exception:
        return "bad_payload"

    audience = payload.get("aud")
    audiences = audience if isinstance(audience, list) else [audience]
    if _AUDIENCE not in audiences:
        return "bad_audience"

    now = int(time.time())
    exp = payload.get("exp")
    if exp is None or now > int(exp) + _LEEWAY:
        return "expired"

    nbf = payload.get("nbf")
    if nbf is not None and now + _LEEWAY < int(nbf):
        return "not_yet_valid"

    return None


def is_valid_service_token(token: str, secret: str) -> bool:
    return rejection_reason(token, secret) is None
