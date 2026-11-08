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


def is_valid_service_token(token: str, secret: str) -> bool:
    try:
        header_segment, payload_segment, signature_segment = token.split(".")
        header = json.loads(_b64url_decode(header_segment))
        if header.get("alg") != "HS256":
            return False

        key = hashlib.sha256(secret.encode("utf-8")).digest()
        signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
        expected = hmac.new(key, signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64url_decode(signature_segment)):
            return False

        payload = json.loads(_b64url_decode(payload_segment))
        audience = payload.get("aud")
        audiences = audience if isinstance(audience, list) else [audience]
        if _AUDIENCE not in audiences:
            return False

        now = int(time.time())
        exp = payload.get("exp")
        if exp is None or now > int(exp) + _LEEWAY:
            return False

        nbf = payload.get("nbf")
        if nbf is not None and now + _LEEWAY < int(nbf):
            return False

        return True
    except Exception:
        return False
