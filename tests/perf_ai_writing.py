import os
import sys
import time

os.environ.setdefault("DEEPSEEK_API_KEY", "test")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from fastapi.testclient import TestClient

import ai_writing
from test_ai_writing import FakeClient, HEADERS, KEY, body

ai_writing._SERVICE_KEY = KEY
ai_writing._client = FakeClient()

app = FastAPI()
app.include_router(ai_writing.router)
http = TestClient(app)

MAX_MS_PER_REQUEST = 20.0
MAX_MS_PER_PROMPT = 0.2
payload = body(text="a" * ai_writing.MAX_TEXT_CHARS, instructions="b" * ai_writing.MAX_INSTRUCTIONS_CHARS)


def timed(count, call):
    started = time.perf_counter()
    for _ in range(count):
        call()
    return (time.perf_counter() - started) * 1000 / count


def test_prompt_building_is_negligible():
    profile = ai_writing.TARGETS["course-description"]
    context = {"title": "Algebra", "shortDescription": "Short"}
    per_prompt = timed(5000, lambda: ai_writing._build_message(profile, "improve", "a" * 3000, context, "b" * 300))
    print(f"prompt build: {per_prompt:.4f} ms")
    assert per_prompt < MAX_MS_PER_PROMPT


def test_a_full_request_adds_little_overhead_over_the_model_call():
    http.post("/ai-writing", json=payload, headers=HEADERS)
    per_request = timed(500, lambda: http.post("/ai-writing", json=payload, headers=HEADERS))
    print(f"full request (model mocked): {per_request:.3f} ms")
    assert per_request < MAX_MS_PER_REQUEST


def test_rejected_requests_are_cheaper_than_accepted_ones():
    bad = body(action="nope")
    before = len(ai_writing._client.chat.completions.calls)
    per_rejection = timed(500, lambda: http.post("/ai-writing", json=bad, headers=HEADERS))
    per_unauthorised = timed(500, lambda: http.post("/ai-writing", json=payload))
    print(f"rejection: {per_rejection:.3f} ms, unauthorised: {per_unauthorised:.3f} ms")
    assert per_rejection < MAX_MS_PER_REQUEST and per_unauthorised < MAX_MS_PER_REQUEST
    assert len(ai_writing._client.chat.completions.calls) == before


def test_concurrent_requests_complete_quickly():
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=16) as pool:
        statuses = list(pool.map(lambda _: http.post("/ai-writing", json=payload, headers=HEADERS).status_code, range(200)))
    elapsed = time.perf_counter() - started
    print(f"200 concurrent requests: {elapsed:.2f} s")
    assert set(statuses) == {200}
    assert elapsed < 10
