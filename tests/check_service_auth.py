import base64
import hashlib
import hmac
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service_auth import is_valid_service_token

SECRET = "a-shared-secret-of-any-length-123"


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def mint(secret: str, *, aud="exam-ia", exp_delta=180, nbf_delta=0) -> str:
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"iss": "platx-backend", "aud": aud, "jti": "abc123",
               "iat": now, "nbf": now + nbf_delta, "exp": now + exp_delta}
    header_segment = _b64url(json.dumps(header, separators=(",", ":")).encode())
    payload_segment = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    key = hashlib.sha256(secret.encode("utf-8")).digest()
    signature = hmac.new(key, f"{header_segment}.{payload_segment}".encode("ascii"), hashlib.sha256).digest()
    return f"{header_segment}.{payload_segment}.{_b64url(signature)}"


def main() -> int:
    cases = [
        ("valid token", is_valid_service_token(mint(SECRET), SECRET) is True),
        ("expired token", is_valid_service_token(mint(SECRET, exp_delta=-120), SECRET) is False),
        ("not-yet-valid token", is_valid_service_token(mint(SECRET, nbf_delta=600), SECRET) is False),
        ("wrong audience", is_valid_service_token(mint(SECRET, aud="someone-else"), SECRET) is False),
        ("wrong secret", is_valid_service_token(mint(SECRET), "different-secret") is False),
        ("tampered signature", is_valid_service_token(mint(SECRET)[:-4] + "AAAA", SECRET) is False),
        ("garbage token", is_valid_service_token("not.a.jwt", SECRET) is False),
        ("empty token", is_valid_service_token("", SECRET) is False),
    ]
    ok = True
    for name, passed in cases:
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
        ok = ok and passed
    print("ALL GREEN" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
