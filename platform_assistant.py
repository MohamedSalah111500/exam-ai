import json
import os
import re
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from openai import OpenAI
from pydantic import BaseModel

from config import DEEPSEEK_MODEL

router = APIRouter()

_API_KEY = os.environ.get('ASSISTANT_API_KEY') or os.environ.get('DEEPSEEK_API_KEY')
_client = OpenAI(api_key=_API_KEY, base_url=os.environ.get('ASSISTANT_BASE_URL', 'https://api.deepseek.com')) if _API_KEY else None

MAX_ANSWER_TOKENS = 350
TOUR_ID_PATTERN = re.compile(r'^[a-z0-9-]{1,64}$')

SYSTEM_PROMPT = (
    "You are the built-in PlatX helper: a friendly support agent that sits inside the PlatX teaching "
    "platform and walks teachers (platform owners) and their assistants (staff) through the screens.\n"
    "\n"
    "RULES:\n"
    "- Answer ONLY from the KNOWLEDGE block. If it does not cover the question, say you do not know yet "
    "and suggest contacting PlatX support. Never invent buttons, pages or features.\n"
    "- Respect the user's role. 'owner' can do everything. 'assistant' (staff) cannot create groups or add "
    "staff; tell them the owner must do it and what to ask the owner for.\n"
    "- If the user is in the middle of a tour, use the CURRENT TOUR line to answer about where they are, "
    "including how to go back and fix a step they forgot.\n"
    "- When one of the listed tours shows the user exactly how to do what they asked, put its id in "
    "\"tour\" so the app can walk them through it; otherwise leave it empty. Never put a tour the KNOWLEDGE "
    "marks as owner-only when the role is 'assistant'.\n"
    "- Reply in the user's language. For Arabic use simple, clear Modern Standard Arabic. Quote button "
    "names exactly as written in the KNOWLEDGE.\n"
    "- Be short: at most 4 short lines or numbered steps. Plain text, no markdown symbols.\n"
    "\n"
    "Return JSON only: {\"answer\": \"...\", \"tour\": \"<tour id or empty>\"}"
)


class AssistantMessage(BaseModel):
    role: str
    content: str


class AssistantTour(BaseModel):
    id: str
    step: int
    total: int
    stepText: Optional[str] = None


class AssistantRequest(BaseModel):
    knowledge: str
    role: str
    question: str
    page: Optional[str] = None
    tour: Optional[AssistantTour] = None
    history: Optional[List[AssistantMessage]] = None


def _context_line(req: AssistantRequest) -> str:
    parts = [f"ROLE: {req.role}"]
    if req.page:
        parts.append(f"PAGE: {req.page}")
    if req.tour:
        step_text = f" — \"{req.tour.stepText}\"" if req.tour.stepText else ""
        parts.append(f"CURRENT TOUR: {req.tour.id}, step {req.tour.step} of {req.tour.total}{step_text}")
    return "\n".join(parts)


@router.post("/platform-assistant")
def platform_assistant(req: AssistantRequest):
    if _client is None:
        raise HTTPException(status_code=503, detail="ASSISTANT_NOT_CONFIGURED")
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="EMPTY_QUESTION")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"=== KNOWLEDGE ===\n{req.knowledge.strip()}\n=== END ==="},
    ]
    for m in (req.history or [])[-6:]:
        if m.role in ("user", "assistant") and m.content.strip():
            messages.append({"role": m.role, "content": m.content.strip()[:1000]})
    messages.append({"role": "user", "content": f"{_context_line(req)}\nQUESTION: {question[:500]}"})

    try:
        response = _client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=MAX_ANSWER_TOKENS,
        )
        data = json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assistant Error: {str(e)}")

    answer = data.get("answer") if isinstance(data, dict) else None
    tour = data.get("tour") if isinstance(data, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        raise HTTPException(status_code=500, detail="EMPTY_ANSWER")
    return {
        "answer": answer.strip(),
        "tour": tour.strip() if isinstance(tour, str) and TOUR_ID_PATTERN.match(tour.strip()) else "",
    }
