import html
import json
import re
import unicodedata
from collections import Counter
from typing import List, Optional

MAX_QUESTIONS = 10
MAX_SOURCE_CHARS = 60000
MAX_ANSWER_TOKENS = 8000
CHOICE_WINDOW = 800

SYSTEM_PROMPT = """You build multiple-choice exam questions for a teacher from the SOURCE they provide.

PRIORITY
1. The TEACHER INSTRUCTIONS are the primary instruction. They override every default below: the count, the language, the level, the question style and whether to copy or generate.
2. The defaults only fill what the teacher instructions do not cover and never contradict them.
3. Never return more than 10 questions, whatever is asked.

MODES
- "copy": the teacher asks to copy, extract, rewrite exactly, or keep the source's questions as they are (in any language, for example "copy the questions", "exactly as given", "انسخ الأسئلة", "زي ما هي"). Then:
  - Do not generate, rephrase, translate, reorder, merge or add anything.
  - Take the source's questions in their original order from the first one, up to the count. If the source has fewer, return fewer. Never fill the gap with new questions.
  - questionHead: the full passage or context exactly as written, then a blank line, then the question stem exactly as written. Keep blanks such as "______".
  - answers: the choices exactly as written, in their original order, without their letter labels (A., B., ...).
  - correctAnswer: the index of the correct choice, from the source's answer key when it has one, otherwise your best judgement.
- "generate": in every other case, write new questions from the source content that follow the teacher instructions and the defaults.

IGNORE in the source: page headers and footers, page numbers, names, phone numbers, branding, section titles, and tips, rules, reminders or notes that are not questions. A question always has its own stem and its own choices.

DEFAULTS (only where the teacher instructions say nothing)
- Count: {count}.
- Language: {language}. Level: {level}.
- Four choices per question with exactly one correct answer.

Return JSON only: {"mode": "copy" or "generate", "questions": [{"id": "1", "questionHead": "...", "answers": ["...", "..."], "correctAnswer": 0}]}"""

_CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "­": None, "�": None,
})
_LABEL = re.compile(r"^\s*[A-Ha-h][\.\)]\s+")


def build_messages(source: str, notes: Optional[str], count: int, language: str, level: str) -> List[dict]:
    rules = (SYSTEM_PROMPT
             .replace("{count}", str(max(1, min(count, MAX_QUESTIONS))))
             .replace("{language}", language or "the source language")
             .replace("{level}", level or "medium"))
    teacher = (notes or "").strip() or "None. Use the defaults."
    return [
        {"role": "system", "content": rules},
        {"role": "system", "content": f"=== TEACHER INSTRUCTIONS (highest priority) ===\n{teacher}\n=== END TEACHER INSTRUCTIONS ==="},
        {"role": "user", "content": f"=== SOURCE ===\n{source[:MAX_SOURCE_CHARS]}\n=== END SOURCE ==="},
    ]


def _page_line_key(line: str) -> str:
    key = re.sub(r"\s+", " ", line).strip().lower()
    return re.sub(r"\d+", "#", key) if len(key) <= 40 else key


def join_pages(pages: List[str]) -> str:
    if len(pages) < 3:
        return "\n".join(pages)
    edges = Counter()
    for page in pages:
        lines = [l for l in page.splitlines() if l.strip()]
        edges.update({_page_line_key(l) for l in lines[:2] + lines[-2:]})
    repeated = {key for key, seen in edges.items() if key and seen >= len(pages) * 0.6}
    kept = []
    for page in pages:
        kept.append("\n".join(l for l in page.splitlines() if _page_line_key(l) not in repeated))
    return "\n".join(kept)


def squash(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").translate(_CHAR_MAP)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"[\s\-_‐-―]+", "", text)


def copy_mismatches(questions: list, source: str) -> List[str]:
    flat = squash(source)
    problems = []
    previous = -1
    for number, question in enumerate(questions, 1):
        head = squash(str(question.get("questionHead", "")))
        position = flat.find(head, previous + 1) if head else -1
        if position < 0:
            problems.append(f"Question {number}: the passage or question text is not exactly as in the SOURCE, or it is not the next question in the SOURCE order.")
        else:
            previous = position
        cursor = position + len(head) if position >= 0 else 0
        for choice in question.get("answers") or []:
            needle = squash(str(choice))
            limit = cursor + CHOICE_WINDOW if position >= 0 else len(flat)
            found = flat.find(needle, cursor, limit) if needle else -1
            if found < 0:
                problems.append(f'Question {number}: the choice "{choice}" is not exactly as in the SOURCE or not in its original order.')
            elif position >= 0:
                cursor = found + len(needle)
    return problems


def _to_html(text: str) -> str:
    paragraphs = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text or "")]
    return "".join(f"<p>{html.escape(p, quote=False)}</p>" for p in paragraphs if p)


def _clean_question(question: dict, number: int) -> Optional[dict]:
    head = str(question.get("questionHead", "")).strip()
    answers = [_LABEL.sub("", str(a)).strip() for a in (question.get("answers") or []) if str(a).strip()]
    if not head or len(answers) < 2:
        return None
    correct = question.get("correctAnswer")
    correct = correct if isinstance(correct, int) and 0 <= correct < len(answers) else None
    return {"id": str(number), "questionHead": _to_html(head), "answers": answers, "correctAnswer": correct}


def _ask(client, model: str, messages: List[dict], temperature: float) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=temperature,
        max_tokens=MAX_ANSWER_TOKENS,
    )
    data = json.loads((response.choices[0].message.content or "").strip())
    if not isinstance(data, dict):
        raise ValueError("EMPTY_ANSWER")
    data["questions"] = [q for q in (data.get("questions") or []) if isinstance(q, dict)][:MAX_QUESTIONS]
    return data


def generate_questions(client, model: str, source: str, notes: Optional[str], count: int, language: str, level: str) -> dict:
    messages = build_messages(source, notes, count, language, level)
    temperature = 0.2 if (notes or "").strip() else 0.3
    data = _ask(client, model, messages, temperature)
    mismatches: List[str] = []
    if data.get("mode") == "copy":
        mismatches = copy_mismatches(data["questions"], source)
        if mismatches:
            retry = messages + [
                {"role": "assistant", "content": json.dumps(data, ensure_ascii=False)},
                {"role": "user", "content": "These items were not copied exactly from the SOURCE:\n- " + "\n- ".join(mismatches[:20])
                 + "\nReturn the complete JSON again. Copy every passage, question stem and choice character for character from the SOURCE, in the original order, and do not add questions."},
            ]
            second = _ask(client, model, retry, 0.0)
            second_mismatches = copy_mismatches(second["questions"], source)
            if len(second_mismatches) < len(mismatches):
                data, mismatches = second, second_mismatches
    questions = [q for q in (_clean_question(q, i) for i, q in enumerate(data["questions"], 1)) if q]
    return {"mode": data.get("mode") or "generate", "questions": questions, "unverified": mismatches}
