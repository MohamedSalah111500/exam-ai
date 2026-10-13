from fastapi import FastAPI, UploadFile, Form, HTTPException, Request
from starlette.datastructures import UploadFile as StarletteUploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import pymupdf
import docx2txt
from openai import OpenAI
import os
import io
import re
import json
import base64
import requests
from datetime import datetime
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from pydantic import BaseModel

# إعداد العميل لـ DeepSeek
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

# Shared vision defaults (DeepSeek's multimodal model); HOMEWORK_VISION_* below can override them.
VISION_API_KEY = os.environ.get('VISION_API_KEY') or os.environ.get('DEEPSEEK_API_KEY')
VISION_BASE_URL = os.environ.get('VISION_BASE_URL', 'https://api.deepseek.com')
VISION_MODEL = os.environ.get('VISION_MODEL', 'deepseek-flash')

# Speech-to-text (OpenAI-compatible /audio/transcriptions). DeepSeek has no STT, so a
# Whisper-compatible provider is used: Groq (default base URL) or api.openai.com.
STT_API_KEY = os.environ.get('STT_API_KEY')
STT_BASE_URL = os.environ.get('STT_BASE_URL', 'https://api.groq.com/openai/v1')
STT_MODEL = os.environ.get('STT_MODEL', 'whisper-large-v3-turbo')
stt_client = OpenAI(api_key=STT_API_KEY, base_url=STT_BASE_URL) if STT_API_KEY else None

# Exam generation and homework grading each have their own key, endpoint and model, so they can be billed,
# rate-limited and tuned separately. Every setting falls back to the shared DeepSeek / vision settings above.
EXAM_API_KEY = os.environ.get('EXAM_API_KEY') or os.environ.get('DEEPSEEK_API_KEY')
EXAM_MODEL = os.environ.get('EXAM_MODEL', 'deepseek-chat')
exam_client = OpenAI(api_key=EXAM_API_KEY, base_url=os.environ.get('EXAM_BASE_URL', 'https://api.deepseek.com')) if EXAM_API_KEY else None

HOMEWORK_API_KEY = os.environ.get('HOMEWORK_API_KEY') or os.environ.get('DEEPSEEK_API_KEY')
HOMEWORK_TEXT_MODEL = os.environ.get('HOMEWORK_TEXT_MODEL', 'deepseek-chat')
homework_text_client = OpenAI(api_key=HOMEWORK_API_KEY, base_url=os.environ.get('HOMEWORK_BASE_URL', 'https://api.deepseek.com')) if HOMEWORK_API_KEY else None
HOMEWORK_VISION_API_KEY = os.environ.get('HOMEWORK_VISION_API_KEY') or os.environ.get('HOMEWORK_API_KEY') or VISION_API_KEY
HOMEWORK_VISION_MODEL = os.environ.get('HOMEWORK_VISION_MODEL', VISION_MODEL)
homework_vision_client = OpenAI(
    api_key=HOMEWORK_VISION_API_KEY,
    base_url=os.environ.get('HOMEWORK_VISION_BASE_URL') or VISION_BASE_URL,
) if HOMEWORK_VISION_API_KEY else None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _detect_upload_kind(data: bytes, filename: str, content_type: str) -> Optional[str]:
    """Return "pdf" / "docx" / None. Content bytes win over the filename, because
    non-ASCII names (e.g. Arabic) can arrive RFC 2047-encoded ("=?utf-8?B?...?=")
    or as "blob", which would hide the real extension."""
    head = data[:1024].lstrip()
    if head.startswith(b"%PDF"):
        return "pdf"
    if data.startswith(b"PK") and b"word/" in data:
        return "docx"
    name = (filename or "").lower()
    ctype = (content_type or "").lower()
    if name.endswith(".pdf") or "pdf" in ctype:
        return "pdf"
    if name.endswith(".docx") or "wordprocessingml" in ctype:
        return "docx"
    return None


@app.post("/generate-exam")
async def generate_exam(
    request: Request,
    pdf_file: Optional[UploadFile] = None,
    text_content: Optional[str] = Form(None),
    language: str = Form(...),
    level: str = Form(...),
    question_count: int = Form(...),
    notes: Optional[str] = Form(None)
):
    # Accept the file under any form field name (file, pdf, document, ...).
    if pdf_file is None:
        form = await request.form()
        pdf_file = next((v for v in form.values() if isinstance(v, StarletteUploadFile)), None)

    # 1. Extract Text
    source_text = ""
    if pdf_file:
        contents = await pdf_file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="الملف المرفوع فارغ.")
        kind = _detect_upload_kind(contents, pdf_file.filename, pdf_file.content_type)
        if kind is None:
            raise HTTPException(status_code=400, detail="نوع الملف غير مدعوم. يرجى رفع ملف PDF أو Word (.docx).")
        try:
            if kind == "pdf":
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
                source_text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
            else:
                source_text = docx2txt.process(io.BytesIO(contents)) or ""
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

        if not source_text.strip() and not (text_content and text_content.strip()):
            raise HTTPException(
                status_code=400,
                detail="تعذر استخراج نص من الملف (قد يكون الملف صورًا ممسوحة ضوئيًا). يرجى رفع ملف يحتوي على نص قابل للتحديد.",
            )

    if text_content and text_content.strip():
        source_text += "\n" + text_content.strip()

    if not source_text.strip():
        raise HTTPException(status_code=400, detail="لم يتم العثور على محتوى لإنشاء الأسئلة.")

    # 2. Refined Prompt for better mapping
    instruction_notes = f"User Notes: {notes}" if notes else ""
    
    prompt = (
        f"Generate exactly {question_count} MCQs based on this text: {source_text[:2000]}\n"
        f"Language: {language}, Level: {level}. {instruction_notes}\n"
        f"Return ONLY a JSON object with this structure:\n"
        f"{{\n"
        f"  \"questions\": [\n"
        f"    {{\n"
        f"      \"id\": \"1\",\n"
        f"      \"questionHead\": \"Question text here\",\n"
        f"      \"answers\": [\"A\", \"B\", \"C\", \"D\"],\n"
        f"      \"correctAnswer\": 0\n"
        f"    }}\n"
        f"  ]\n"
        f"}}"
    )

    try:
        # 3. AI Request
        response = exam_client.chat.completions.create(
            model=EXAM_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional teacher. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={'type': 'json_object'},
            temperature=0.3
        )

        response_content = response.choices[0].message.content.strip()
        ai_data = json.loads(response_content)
        
        # Ensure we are passing the list of questions, not the whole object, 
        # to match your frontend's 'questions' property
        raw_questions = ai_data.get("questions", [])

        # 4. Final Structure for Frontend Mapping
        # This matches the "Exam" object structure your frontend likely expects
        return {
            "id": 0,
            "name": f"اختبار {level} - {language}",
            "isAutoCorrect": True,
            "createdBy": 1,
            "creationTime": datetime.now().isoformat(),
            "questions": raw_questions,  # This list must contain objects with questionHead, answers, etc.
            "isShowCorrectAnswers": True
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mapping Error: {str(e)}")


# ----------------------------------------------------------------------
# Homework AI grading
# Called server-to-server by PlatX (.NET) after a student submits homework when the
# grading mode is "AI suggestion" or "AI only". PlatX authenticates the user, enforces
# the tenant's AI-credit limit, and sends the rubric + the student's work here.
# Text files (PDF / Word) are extracted and graded by DeepSeek; image files are read by
# DeepSeek's multimodal model (see VISION_* env vars to use another provider).
# ----------------------------------------------------------------------


class GradeHomeworkRequest(BaseModel):
    homework_name: str
    total_score: float
    # Rubric + the student's answers, already flattened to text by PlatX.
    rubric: str
    # URLs of the student's uploaded files (PlatX-hosted).
    file_urls: List[str] = []
    language: str = "auto"


IMAGE_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
MAX_GRADED_IMAGES = 6
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def _url_ext(url: str) -> str:
    return os.path.splitext(urlparse(url).path.lower())[1]


def _image_data_url(url: str) -> Optional[str]:
    """Download a student image and inline it as a data URL, so the model never has to reach PlatX storage."""
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.content
        if not data or len(data) > MAX_IMAGE_BYTES:
            return None
        ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        mime = ctype if ctype in IMAGE_MIME_BY_EXT.values() else IMAGE_MIME_BY_EXT.get(_url_ext(url))
        if not mime:
            return None
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    except Exception:
        return None


def _parse_grade(raw: str, total_score: float) -> dict:
    match = re.search(r"\{.*\}", raw or "", re.S)
    data = json.loads(match.group(0) if match else raw)
    score = float(data.get("score", 0) or 0)
    return {"score": max(0.0, min(score, float(total_score))), "feedback": data.get("feedback", "")}


PDF_PAGE_DPI = 150


def _read_document(url: str, max_page_images: int = 0):
    """Download a PDF/Word file and return (text, page images as data URLs).

    PDF pages are also rendered as images, annotations included, so handwriting, drawings and
    marks added on top of the printed sheet reach the vision model; text extraction alone misses them.
    """
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.content
    except Exception:
        return "", []

    ctype = (resp.headers.get("Content-Type") or "").lower()
    ext = _url_ext(url)
    if data[:4] == b"%PDF" or ext == ".pdf" or "pdf" in ctype:
        text = ""
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            text = "".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            pass
        images = []
        if max_page_images > 0:
            try:
                with pymupdf.open(stream=data, filetype="pdf") as doc:
                    for page in doc:
                        if len(images) >= max_page_images:
                            break
                        pix = page.get_pixmap(dpi=PDF_PAGE_DPI, annots=True)
                        images.append("data:image/jpeg;base64," + base64.b64encode(pix.tobytes("jpeg", jpg_quality=85)).decode("ascii"))
            except Exception:
                images = []
        return text, images
    if ext == ".docx" or data[:2] == b"PK":
        try:
            return docx2txt.process(io.BytesIO(data)) or "", []
        except Exception:
            return "", []
    return "", []


def _extract_file_text(url: str) -> str:
    """Download a PDF/Word file and return its text (best-effort)."""
    return _read_document(url)[0]


@app.post("/grade-homework")
async def grade_homework(req: GradeHomeworkRequest):
    if not req.rubric or not req.rubric.strip():
        raise HTTPException(status_code=400, detail="Missing rubric / student work to grade.")

    image_urls = [u for u in (req.file_urls or []) if _url_ext(u) in IMAGE_MIME_BY_EXT]
    doc_urls = [u for u in (req.file_urls or []) if _url_ext(u) not in IMAGE_MIME_BY_EXT]

    # 1) Collect any text from uploaded documents.
    extracted = ""
    for u in doc_urls:
        text = _extract_file_text(u)
        if text.strip():
            extracted += f"\n\n[Attached document]\n{text[:4000]}"

    base_instruction = (
        f"You are grading a homework titled \"{req.homework_name}\" worth {req.total_score} points.\n"
        f"Grade the student's work against the rubric. Be fair and encouraging.\n"
        f"Return ONLY a JSON object: {{\"score\": <number 0..{req.total_score}>, \"feedback\": \"<short feedback>\"}}.\n\n"
        f"=== RUBRIC & STUDENT WORK ===\n{req.rubric}\n{extracted}\n=== END ==="
    )

    images = []
    if homework_vision_client is not None:
        images = [d for d in (_image_data_url(u) for u in image_urls[:MAX_GRADED_IMAGES]) if d]

    try:
        if images:
            image_hint = "\n\nThe student's handwritten or photographed work is in the attached images; read it carefully."
            content = [{"type": "text", "text": base_instruction + image_hint}]
            content += [{"type": "image_url", "image_url": {"url": d}} for d in images]
            response = homework_vision_client.chat.completions.create(
                model=HOMEWORK_VISION_MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional teacher. Return valid JSON only."},
                    {"role": "user", "content": content},
                ],
                temperature=0.2,
            )
        else:
            note = ""
            if image_urls:
                note = "\n\nNote: the student uploaded image files that could not be read automatically; grade what is available and mention that images need teacher review."
            response = homework_text_client.chat.completions.create(
                model=HOMEWORK_TEXT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional teacher. Return valid JSON only."},
                    {"role": "user", "content": base_instruction + note},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )

        return _parse_grade(response.choices[0].message.content.strip(), req.total_score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grading Error: {str(e)}")


class HomeworkOption(BaseModel):
    number: int
    letter_ar: str = ""
    letter_en: str = ""
    text: str = ""


class HomeworkQuestion(BaseModel):
    id: int
    number: int
    # single | multi | truefalse | essay
    type: str
    body: str = ""
    max_score: float = 1
    options: List[HomeworkOption] = []
    read_from_images: bool = False
    student_text: Optional[str] = None
    answer_by_upload: bool = False
    model_answer: Optional[str] = None
    model_answer_image_url: Optional[str] = None
    # Files the student attached to this question only.
    file_urls: List[str] = []


class GradeHomeworkV2Request(BaseModel):
    homework_name: str
    total_score: float
    instructions: Optional[str] = None
    questions: List[HomeworkQuestion] = []
    file_urls: List[str] = []
    # Pages (PDF pages or photos) the academy's AI credit pays for; set by PlatX.
    max_pages: int = 50


CHOICE_TYPES = {"single", "multi", "truefalse"}
CONFIDENCE_LEVELS = {"high", "medium", "low"}
CONFIDENCE_RANK = {"low": 1, "medium": 2, "high": 3}
MAX_MODEL_ANSWER_IMAGES = 4
# Every page of the student's work is read; pages are sent in small batches so the model looks at each one carefully.
MAX_STUDENT_PAGES = int(os.environ.get("HOMEWORK_MAX_PAGES", "50"))
PAGES_PER_BATCH = int(os.environ.get("HOMEWORK_PAGES_PER_BATCH", "6"))
BATCH_WORKERS = int(os.environ.get("HOMEWORK_BATCH_WORKERS", "6"))
MAX_PAGE_TEXT = 2500

HOMEWORK_V2_RULES = (
    "You are a careful teacher grading one student's homework.\n"
    "The platform questions are listed below with their id. The student's paper may use its OWN numbering or sections "
    "(numbering can restart per section, and questions may be in a different order). Identify each question by its TEXT "
    "and OPTIONS, never by its number alone, and always report it with the platform id.\n"
    "Options are labelled with Arabic letters (أ ب ج د ...) or English letters (A B C D ...); both labels refer to the same option number.\n\n"
    "- single / multi / truefalse: report which option(s) the student marked (circled, ticked, underlined, highlighted, shaded, "
    "struck through next to a letter, or a written letter such as '3- ب' / an answer table). Return them as option numbers "
    "in selected_options. Do NOT judge whether the choice is correct and set score to null.\n"
    "- essay: grade the student's answer (typed text and/or what is written or drawn) from 0 to max_score. Use the model "
    "answer when one is given. For drawings judge the drawing itself: correct shape, labels, measurements and steps. "
    "Give partial credit for partially correct work.\n\n"
    "answer_found: true when you can actually see the student's answer to that question, false when the question is there "
    "but the student's answer is not.\n"
    "confidence: 'high' only when the answer is clearly visible and unambiguous; 'medium' when probable; 'low' when guessing.\n"
    "Never invent an answer that is not visible.\n"
    "feedback: one short sentence per question, in the language of the questions.\n\n"
    "Return ONLY a JSON object: "
    "{\"questions\": [{\"id\": <platform question id>, \"answer_found\": true|false, \"selected_options\": [<option numbers>], "
    "\"confidence\": \"high|medium|low\", \"score\": <number or null>, \"feedback\": \"...\"}], \"feedback\": \"<short overall feedback>\"}"
)

HOMEWORK_PAGE_RULES = HOMEWORK_V2_RULES + (
    "\n\nThe attached images are some of the student's pages (photos or PDF pages). Look at EVERY attached page from top to "
    "bottom; a page may hold several questions, and an answer table may hold the answers of many questions. Report only the "
    "questions that appear on THESE pages (their text, or the student's answer to them, e.g. in an answer table); leave out "
    "questions that do not appear on them."
)

HOMEWORK_TYPED_RULES = HOMEWORK_V2_RULES + (
    "\n\nThere are no images: grade EVERY listed question from the student's typed answer and report all of them."
)

HOMEWORK_UPLOAD_ONLY_RULES = (
    "You are a careful teacher grading one student's homework that was submitted as uploaded pages.\n"
    "Go through EVERY attached page from top to bottom and list EVERY question or exercise you find, without skipping any.\n"
    "List a question only on the page where its question text starts; if a question started on a page that is not attached, "
    "do not list it.\n"
    "For each item: label = how it is identified on the paper (include the section if numbering restarts, e.g. 'Type 2 - 3'); "
    "page = the page number given in the page caption; student_answer = what the student marked or wrote (e.g. 'B' or a short "
    "summary); correct = true / false after you solve the question yourself, or null if the student left it unanswered or you "
    "cannot decide; confidence = 'high' only when both the student's mark and the correct answer are clear, otherwise 'medium' or 'low'.\n"
    "Follow the teacher's instructions when they give an answer key or marking rules.\n"
    "Return ONLY a JSON object: {\"items\": [{\"label\": \"...\", \"page\": <number>, \"student_answer\": \"...\", "
    "\"correct\": true|false|null, \"confidence\": \"high|medium|low\"}], \"feedback\": \"<short feedback in the language of the work>\"}"
)


def _student_pages(url: str, limit: int):
    """Download one student file and return (pages, document text). Each page is {"image": data URL, "text": printed text}.

    PDF pages are rendered with their annotations, so handwriting, highlights and drawings added on top of a
    printed or digital sheet reach the vision model; the printed text of each page is kept next to its image.
    """
    if _url_ext(url) in IMAGE_MIME_BY_EXT:
        d = _image_data_url(url) if limit > 0 else None
        return [{"image": d, "text": ""}], ""

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.content
    except Exception:
        return [], ""

    ctype = (resp.headers.get("Content-Type") or "").lower()
    if data[:4] == b"%PDF" or _url_ext(url) == ".pdf" or "pdf" in ctype:
        pages = []
        try:
            with pymupdf.open(stream=data, filetype="pdf") as doc:
                for page in doc:
                    text = (page.get_text() or "").strip()[:MAX_PAGE_TEXT]
                    image = None
                    if len(pages) < limit:
                        pix = page.get_pixmap(dpi=PDF_PAGE_DPI, annots=True)
                        image = "data:image/jpeg;base64," + base64.b64encode(pix.tobytes("jpeg", jpg_quality=80)).decode("ascii")
                    pages.append({"image": image, "text": text})
        except Exception:
            return [], ""
        return pages, ""
    if _url_ext(url) == ".docx" or data[:2] == b"PK":
        try:
            return [], (docx2txt.process(io.BytesIO(data)) or "")[:12000]
        except Exception:
            return [], ""
    mime = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if mime in IMAGE_MIME_BY_EXT.values() and limit > 0 and len(data) <= MAX_IMAGE_BYTES:
        return [{"image": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}", "text": ""}], ""
    return [], ""


def _describe_homework_question(q: HomeworkQuestion, has_model_image: bool = False) -> str:
    lines = [f"Question id={q.id} (platform number {q.number}, type={q.type}, max_score={q.max_score}): {q.body}"]
    for o in q.options:
        lines.append(f"  option {o.number} [{o.letter_ar} / {o.letter_en}]: {o.text}")
    if q.student_text:
        lines.append(f"  Student typed answer: {q.student_text}")
    if q.model_answer:
        lines.append(f"  Model answer: {q.model_answer}")
    if has_model_image:
        lines.append("  A model answer image for this question is attached and labelled.")
    return "\n".join(lines)


def _clean_question_result(item: dict, q: HomeworkQuestion) -> dict:
    valid_numbers = {o.number for o in q.options}
    selected = []
    for n in item.get("selected_options") or []:
        try:
            n = int(n)
        except (TypeError, ValueError):
            continue
        if n in valid_numbers and n not in selected:
            selected.append(n)
    confidence = str(item.get("confidence") or "low").strip().lower()
    if confidence not in CONFIDENCE_LEVELS:
        confidence = "low"
    score = None
    if q.type not in CHOICE_TYPES:
        try:
            score = max(0.0, min(float(item.get("score")), float(q.max_score)))
        except (TypeError, ValueError):
            score, confidence = 0.0, "low"
    found = item.get("answer_found")
    if not isinstance(found, bool):
        found = bool(selected) if q.type in CHOICE_TYPES else bool(score)
    return {
        "id": q.id,
        "answer_found": found,
        "selected_options": selected if q.type in CHOICE_TYPES else [],
        "confidence": confidence,
        "score": score,
        "feedback": str(item.get("feedback") or "")[:500],
    }


def _merge_question_results(reports, questions) -> list:
    """Each question may be reported by several batches; keep the most confident reading and flag disagreements."""
    by_id = {q.id: q for q in questions}
    found = {}
    for data in reports:
        for item in data.get("questions") or []:
            try:
                q = by_id.get(int(item.get("id")))
            except (TypeError, ValueError):
                q = None
            if q is not None:
                found.setdefault(q.id, []).append(_clean_question_result(item, q))

    merged = []
    for q in questions:
        results = found.get(q.id)
        if not results:
            merged.append({"id": q.id, "selected_options": [], "confidence": "low",
                           "score": None if q.type in CHOICE_TYPES else 0.0, "feedback": ""})
            continue
        answered = [r for r in results if r["answer_found"]]
        pool = answered or results
        best = dict(max(pool, key=lambda r: CONFIDENCE_RANK[r["confidence"]]))
        if q.type in CHOICE_TYPES:
            readings = {tuple(sorted(r["selected_options"])) for r in answered}
        else:
            readings = {r["score"] for r in answered}
            if answered:
                best["score"] = max(r["score"] or 0.0 for r in answered)
        if len(readings) > 1:
            best["confidence"] = "medium" if best["confidence"] == "high" else best["confidence"]
        best.pop("answer_found", None)
        merged.append(best)
    return merged


def _ask_model(rules: str, prompt: str, images) -> dict:
    """images: list of (caption, data URL). Uses the vision model when there are images, otherwise DeepSeek chat."""
    if images:
        content = [{"type": "text", "text": prompt}]
        for caption, d in images:
            content.append({"type": "text", "text": caption})
            content.append({"type": "image_url", "image_url": {"url": d}})
        response = homework_vision_client.chat.completions.create(
            model=HOMEWORK_VISION_MODEL,
            messages=[{"role": "system", "content": rules}, {"role": "user", "content": content}],
            temperature=0.1,
        )
    else:
        response = homework_text_client.chat.completions.create(
            model=HOMEWORK_TEXT_MODEL,
            messages=[{"role": "system", "content": rules}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
    raw = response.choices[0].message.content or ""
    match = re.search(r"\{.*\}", raw, re.S)
    return json.loads(match.group(0) if match else raw)


def _run_all(jobs):
    """jobs: list of (rules, prompt, images). Runs them in parallel; returns (reports, failed job count).
    A failed batch only leaves its questions for the teacher; if every batch fails the grading fails."""
    def run(job):
        try:
            return _ask_model(*job)
        except Exception:
            return None

    if len(jobs) == 1:
        results = [run(jobs[0])]
    else:
        with ThreadPoolExecutor(max_workers=max(1, BATCH_WORKERS)) as pool:
            results = list(pool.map(run, jobs))
    reports = [r for r in results if isinstance(r, dict)]
    if jobs and not reports:
        raise RuntimeError("every grading request to the AI model failed")
    return reports, len(results) - len(reports)


def _page_batches(pages):
    return [pages[i:i + PAGES_PER_BATCH] for i in range(0, len(pages), PAGES_PER_BATCH)]


def _page_caption(page) -> str:
    caption = f"Student page {page['number']} of {page['total']}"
    if page.get("question"):
        caption = f"Answer for platform question number {page['question']} (attached by the student to that question)"
    if page["text"]:
        caption += f". Printed text on this page: {page['text']}"
    return caption + ":"


def _collect_pages(req: GradeHomeworkV2Request):
    """All student pages in order (whole-homework files, then per-question files), the documents' text and skipped pages."""
    sheet, question_pages, documents, skipped = [], [], "", 0
    budget = max(0, min(MAX_STUDENT_PAGES, req.max_pages)) if homework_vision_client is not None else 0

    def take(url, question_number=None):
        nonlocal documents, skipped
        room = budget - len(sheet) - len(question_pages)
        pages, text = _student_pages(url, max(0, room))
        if text.strip():
            label = f"question {question_number}" if question_number else "the homework"
            documents += f"\n\n[Student document for {label}]\n{text}"
        for p in pages:
            if p["image"] is None:
                skipped += 1
            elif question_number:
                question_pages.append({**p, "question": question_number})
            else:
                sheet.append(p)

    for u in req.file_urls or []:
        take(u)
    for q in req.questions:
        for u in q.file_urls:
            take(u, q.number)

    for i, p in enumerate(sheet, start=1):
        p.update(number=i, total=len(sheet))
    for p in question_pages:
        p.update(number=0, total=0)
    unreadable = bool(req.file_urls or any(q.file_urls for q in req.questions)) and not (sheet or question_pages or documents)
    return sheet, question_pages, documents, skipped, unreadable


def _grade_questions(req: GradeHomeworkV2Request, sheet, question_pages, documents, skipped, unreadable) -> dict:
    model_images = []
    if homework_vision_client is not None:
        for q in req.questions:
            if q.model_answer_image_url and len(model_images) < MAX_MODEL_ANSWER_IMAGES:
                d = _image_data_url(q.model_answer_image_url)
                if d:
                    model_images.append((q.number, d))
    with_model_image = {number for number, _ in model_images}
    model_captions = [(f"Model answer image for platform question number {n} (teacher's reference, not the student's work):", d)
                      for n, d in model_images]

    header = (f"Homework: {req.homework_name}\nTotal score: {req.total_score}\n"
              f"Teacher instructions: {req.instructions or '-'}\n")

    def listing(questions):
        return "\n\n".join(_describe_homework_question(q, q.number in with_model_image) for q in questions)

    reads_paper = [q for q in req.questions if q.read_from_images]
    typed_only = [q for q in req.questions if not q.read_from_images]

    jobs = []
    if typed_only:
        jobs.append((HOMEWORK_TYPED_RULES,
                     header + "\n=== QUESTIONS ===\n" + listing(typed_only) + "\n=== END QUESTIONS ===", []))

    pages = sheet + question_pages
    if reads_paper and pages:
        batches = _page_batches(pages)
        for index, batch in enumerate(batches, start=1):
            note = (f"\nThese are batch {index} of {len(batches)} of the student's pages. Report only the questions "
                    f"whose answers appear on THESE pages.\n")
            prompt = header + note + "\n=== QUESTIONS ===\n" + listing(reads_paper) + "\n=== END QUESTIONS ===" + documents
            jobs.append((HOMEWORK_PAGE_RULES, prompt, [(_page_caption(p), p["image"]) for p in batch] + model_captions))
    elif reads_paper:
        note = ("\nNote: the student's files could not be read; use confidence 'low' for answers that would be in them.\n"
                if unreadable else "\n")
        jobs.append((HOMEWORK_PAGE_RULES if model_captions else HOMEWORK_TYPED_RULES,
                     header + note + "\n=== QUESTIONS ===\n" + listing(reads_paper) + "\n=== END QUESTIONS ===" + documents,
                     model_captions))

    reports, failed = _run_all(jobs) if jobs else ([], 0)
    feedback = " ".join(dict.fromkeys(str(r.get("feedback") or "").strip() for r in reports if r.get("feedback")))
    if failed:
        feedback = f"⚠ {failed} batch(es) could not be read by the AI. " + feedback
    if skipped:
        feedback = f"⏸ {skipped} page(s) not read. " + feedback
    return {
        "questions": _merge_question_results(reports, req.questions),
        "score": None,
        "feedback": feedback[:1000],
        "needs_review": skipped > 0 or unreadable or failed > 0,
        "pages_read": len(pages),
        "pages_skipped": skipped,
    }


def _grade_upload_only(req: GradeHomeworkV2Request, sheet, question_pages, documents, skipped, unreadable) -> dict:
    header = (f"Homework: {req.homework_name}\nTotal score: {req.total_score}\n"
              f"Teacher instructions: {req.instructions or '-'}\n")
    pages = sheet + question_pages
    if pages:
        batches = _page_batches(pages)
        jobs = [(HOMEWORK_UPLOAD_ONLY_RULES,
                 header + f"\nThese are batch {i} of {len(batches)} of the student's pages; list every question on them.\n" + documents,
                 [(_page_caption(p), p["image"]) for p in batch])
                for i, batch in enumerate(batches, start=1)]
    else:
        jobs = [(HOMEWORK_UPLOAD_ONLY_RULES, header + "\nThe student's work as text:\n" + (documents or "(nothing readable)"), [])]

    reports, failed = _run_all(jobs)
    items, seen = [], set()
    for data in reports:
        for item in data.get("items") or []:
            label = str(item.get("label") or "").strip()[:80]
            page = item.get("page")
            key = (label.lower(), str(page))
            if not label or (page is not None and key in seen):
                continue
            seen.add(key)
            confidence = str(item.get("confidence") or "low").strip().lower()
            correct = item.get("correct")
            items.append({
                "label": label,
                "page": item.get("page"),
                "student_answer": str(item.get("student_answer") or "")[:200],
                "correct": correct if isinstance(correct, bool) else None,
                "confidence": confidence if confidence in CONFIDENCE_LEVELS else "low",
            })

    right = sum(1 for i in items if i["correct"] is True)
    complete = items and not skipped and not failed
    score = round(float(req.total_score) * right / len(items), 2) if complete else None
    wrong = [i["label"] for i in items if i["correct"] is False]
    unsure = [i["label"] for i in items if i["correct"] is None or i["confidence"] != "high"]
    summary = f"✔ {right}/{len(items)}" if items else ""
    if wrong:
        summary += " · ✘ " + ", ".join(wrong[:40])
    if unsure:
        summary += " · ? " + ", ".join(unsure[:40])
    if skipped:
        summary += f" · ⏸ {skipped} page(s) not read"
    if failed:
        summary += f" · ⚠ {failed} batch(es) failed"
    model_feedback = " ".join(dict.fromkeys(str(r.get("feedback") or "").strip() for r in reports if r.get("feedback")))
    return {
        "questions": [],
        "items": items,
        "score": score,
        "feedback": (summary + ("\n" if summary and model_feedback else "") + model_feedback)[:1500],
        "needs_review": not items or bool(unsure) or skipped > 0 or unreadable or failed > 0,
        "pages_read": len(pages),
        "pages_skipped": skipped,
    }


@app.post("/grade-homework-v2")
def grade_homework_v2(req: GradeHomeworkV2Request):
    # Plain def: FastAPI runs it in its threadpool, so the slow downloads and model calls never block other endpoints.
    try:
        sheet, question_pages, documents, skipped, unreadable = _collect_pages(req)
        if req.questions:
            return _grade_questions(req, sheet, question_pages, documents, skipped, unreadable)
        return _grade_upload_only(req, sheet, question_pages, documents, skipped, unreadable)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grading Error: {str(e)}")


# ----------------------------------------------------------------------
# Student chatbot
# This service is called server-to-server by PlatX (.NET), NOT by the
# browser. PlatX authenticates the student, resolves the tenant from the
# JWT, builds the tenant's knowledge base, and sends it here as `context`.
# We only ask DeepSeek to answer from that context.
# The context is sent on every request, ordered stable-first (system rules
# + context, then question) so DeepSeek auto-caches it.
# ----------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str          # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    context: str = ""                      # tenant knowledge base, built and sent by PlatX
    history: Optional[List[ChatMessage]] = None


CHATBOT_SYSTEM_PROMPT = (
    "أنت مساعد ذكي لمنصة تعليمية. مهمتك الإجابة على أسئلة الطلاب بالاعتماد *فقط* "
    "على المعلومات الموجودة في قسم (معلومات المنصة) أدناه.\n"
    "قواعد مهمة:\n"
    "1. لا تخترع أي معلومة غير موجودة في (معلومات المنصة).\n"
    "2. إذا كان السؤال عن شيء غير موجود في المعلومات (مثل كورس أو معلم غير مذكور)، "
    "قل بوضوح أن هذه المعلومة غير متوفرة حالياً وانصح الطالب بالتواصل مع الإدارة عبر "
    "أرقام التواصل الرسمية.\n"
    "3. لا تعطِ أي أرقام هواتف أو بيانات تواصل شخصية للمعلمين؛ استخدم فقط أرقام التواصل الرسمية للمنصة.\n"
    "4. أجب بنفس لغة الطالب، وبأسلوب ودود ومختصر.\n"
    "5. لا تذكر أنك تقرأ من (معلومات المنصة)؛ فقط أجب بشكل طبيعي."
)


@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="السؤال فارغ.")

    knowledge = (req.context or "").strip() or "لا توجد معلومات متاحة حالياً."

    context_block = (
        "=== معلومات المنصة ===\n"
        f"{knowledge}\n"
        "=== نهاية المعلومات ==="
    )

    # stable-first ordering so DeepSeek caches the system rules + knowledge
    messages = [
        {"role": "system", "content": CHATBOT_SYSTEM_PROMPT},
        {"role": "system", "content": context_block},
    ]

    # include recent history so follow-up questions keep context
    if req.history:
        for m in req.history[-10:]:
            if m.role in ("user", "assistant") and m.content.strip():
                messages.append({"role": m.role, "content": m.content})

    messages.append({"role": "user", "content": req.question.strip()})

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat Error: {str(e)}")


# ----------------------------------------------------------------------
# CRM insights
# Called server-to-server by PlatX with a text snapshot of the sales CRM (exact
# totals plus the most recent leads and their logged activities/comments). The
# model answers management questions strictly from that snapshot.
# ----------------------------------------------------------------------

class CrmInsightsRequest(BaseModel):
    question: str
    data: str
    history: Optional[List[ChatMessage]] = None


CRM_INSIGHTS_SYSTEM_PROMPT = (
    "You are a senior sales analyst for PlatX, an Egyptian SaaS company that sells online teaching "
    "platforms to teachers, tutors and academies. You analyse the company's sales CRM and answer the "
    "management team's questions.\n"
    "\n"
    "DATA YOU RECEIVE:\n"
    "- 'EXACT TOTALS' are computed by the database over every lead the user can see. Always use them "
    "for counts and percentages when they answer the question.\n"
    "- 'LEAD DETAILS' lists the most recently updated leads (it may not include every lead; the last line "
    "says how many are included). Each lead line has its status, source, priority, dates, agent, notes and "
    "lost reason, followed by its most recent activities (Note, Call, WhatsApp, Meeting, Email, "
    "StatusChange, Assignment, FollowUp) with what the team wrote.\n"
    "- Lead statuses in pipeline order: New, Contacted, Interested, DemoScheduled, Negotiation, Won, Lost.\n"
    "\n"
    "HOW TO ANSWER:\n"
    "- Base every statement only on the data. Never invent leads, numbers, reasons or quotes.\n"
    "- For 'why' questions (why leads don't convert, what annoys customers, objections), read the lost "
    "reasons, notes and activity comments, group similar reasons into clear themes, and rank them by how "
    "many leads mention each theme. Give the count and percentage for each theme and cite 2-3 example lead "
    "numbers (like #123).\n"
    "- For rates (e.g. meeting no-shows), explain exactly what you counted and the denominator. When the "
    "CRM has no dedicated field for something, infer it from the comments, say that it is inferred, and "
    "base the percentage only on the leads whose comments mention it.\n"
    "- When a percentage comes from the lead details rather than the exact totals, say it is based on the N "
    "leads analysed.\n"
    "- If the data cannot answer the question, say so plainly and suggest what the team should start "
    "recording in the CRM to answer it next time.\n"
    "- End with 2-4 short, concrete recommendations when the question is about a problem.\n"
    "\n"
    "LANGUAGE AND FORMAT:\n"
    "- Reply in the language and dialect of the question (Egyptian Arabic if asked in Egyptian Arabic).\n"
    "- Plain text only: no markdown symbols like #, *, ** or tables. Use short lines, a short title line "
    "when helpful, and '- ' at the start of list items.\n"
    "- Be concise and specific; lead with the direct answer."
)


@app.post("/crm-insights")
def crm_insights(req: CrmInsightsRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="EMPTY_QUESTION")

    data_block = (
        "=== CRM SNAPSHOT ===\n"
        f"{(req.data or '').strip() or 'No CRM data.'}\n"
        "=== END OF CRM SNAPSHOT ==="
    )

    messages = [
        {"role": "system", "content": CRM_INSIGHTS_SYSTEM_PROMPT},
        {"role": "system", "content": data_block},
    ]
    if req.history:
        for m in req.history[-10:]:
            if m.role in ("user", "assistant") and m.content.strip():
                messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": req.question.strip()})

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.2,
        )
        return {"answer": response.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CRM Insights Error: {str(e)}")


# ----------------------------------------------------------------------
# Voice student registration
# Called server-to-server by PlatX after a student records a voice note on the
# register screen. Audio is transcribed (STT), then DeepSeek merges the transcript
# into the signup form (name / email / password) and asks for what is still missing.
# ----------------------------------------------------------------------

STUDENT_FIELD_KEYS = ("firstName", "lastName", "email", "password")
PASSWORD_PATTERN = re.compile(r"^[A-Za-z0-9]{8,}$")

VOICE_STUDENT_SYSTEM_PROMPT = (
    "You are a smart voice form-filling assistant for a student sign-up form on an educational platform. "
    "The student speaks in Arabic, English, or a mix. You receive the transcript of their latest recording "
    "plus the fields already collected in earlier recordings, and you return the updated form.\n"
    "\n"
    "FORM FIELDS: firstName, lastName, email, password.\n"
    "\n"
    "MERGING & UPDATES:\n"
    "- Start from the previously collected fields and apply what the new transcript says.\n"
    "- The student may correct or update anything (\"change my last name to Ali\", \"غيّر الإيميل\", \"no, it's ...\", "
    "\"my name is actually ...\"): the newest statement wins.\n"
    "- The student may remove a field (\"remove the email\", \"امسح الباسورد\"): set it to null.\n"
    "- Fields not mentioned in the transcript stay exactly as previously collected.\n"
    "- Never invent a value. If something is unclear or was not said, leave it null.\n"
    "\n"
    "LANGUAGE:\n"
    "- Default is English: write firstName and lastName in Latin letters (transliterate Arabic names, e.g. "
    "محمد -> Mohamed, أحمد -> Ahmed, فاطمة -> Fatma) and write followUpQuestion in English.\n"
    "- Only if the student explicitly asks for Arabic (\"بالعربي\", \"اكتب اسمي بالعربي\", \"in Arabic\") keep the names in "
    "Arabic script and write followUpQuestion in Arabic. Keep that choice for the rest of the conversation "
    "(if previously collected names are in Arabic script, the student already chose Arabic).\n"
    "\n"
    "NORMALIZATION:\n"
    "- Spelled-out letters (\"m o h a m e d\", \"ميم واو حاء\") are joined into one word.\n"
    "- Numbers spoken as words (\"one two three\", \"واحد اثنين ثلاثة\", \"خمسة\") become digits.\n"
    "- email: lower-case; \"at\"/\"آت\" -> \"@\", \"dot\"/\"نقطة\" -> \".\", \"underscore\"/\"شرطة سفلية\" -> \"_\", "
    "\"dash\"/\"شرطة\" -> \"-\"; remove spaces; if only a well-known provider is said (gmail, yahoo, hotmail, "
    "outlook, icloud) without a TLD, append \".com\".\n"
    "- password: the value the student says after \"password\"/\"كلمة السر\"/\"الباسورد\"/\"كلمة المرور\"; join spoken "
    "letters and digits, remove spaces, keep the letter case the student states (\"capital A\" -> \"A\"; default lower-case). "
    "The password must be 8+ characters using ONLY English letters and digits. If what was said violates that "
    "(too short, spaces, symbols, Arabic letters), set password to null and explain the rule in followUpQuestion.\n"
    "- Capitalize the first letter of Latin names.\n"
    "\n"
    "OUTPUT: return ONLY a JSON object:\n"
    "{\"fields\": {\"firstName\": ..., \"lastName\": ..., \"email\": ..., \"password\": ...}, "
    "\"missingFields\": [...], \"followUpQuestion\": \"...\"}\n"
    "- missingFields: which of the four fields are still null.\n"
    "- followUpQuestion: one short friendly sentence. If fields are missing, ask for all of them in one sentence. "
    "If a value looks doubtful (unusual email, odd spelling) ask the student to confirm it. If everything is complete, "
    "briefly summarize name and email (never repeat the password) and ask them to review and submit."
)


def _transcribe_audio(upload: UploadFile, language: str) -> str:
    if stt_client is None:
        raise HTTPException(status_code=503, detail="STT_NOT_CONFIGURED")
    raw = upload.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="EMPTY_AUDIO")
    filename = upload.filename or "voice.webm"
    kwargs = {"model": STT_MODEL, "file": (filename, raw, upload.content_type or "application/octet-stream")}
    if language in ("ar", "en"):
        kwargs["language"] = language
    try:
        result = stt_client.audio.transcriptions.create(**kwargs)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"STT Error: {str(e)}")
    return (getattr(result, "text", None) or "").strip()


def _parse_json_arg(value: Optional[str], default):
    if not value:
        return default
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return default
    return parsed if isinstance(parsed, type(default)) else default


def _normalize_student_fields(raw_fields) -> dict:
    fields = {key: None for key in STUDENT_FIELD_KEYS}
    if not isinstance(raw_fields, dict):
        return fields
    for key in STUDENT_FIELD_KEYS:
        value = raw_fields.get(key)
        if isinstance(value, (int, float)):
            value = str(value)
        if not isinstance(value, str):
            continue
        value = value.strip()
        if key == "email":
            value = value.replace(" ", "").lower()
        elif key == "password":
            value = value.replace(" ", "")
            if not PASSWORD_PATTERN.match(value):
                value = ""
        fields[key] = value or None
    return fields


@app.post("/voice-student")
async def voice_student(
    audio: Optional[UploadFile] = None,
    transcript: Optional[str] = Form(None),
    language: str = Form("auto"),
    current_fields: Optional[str] = Form(None),
):
    language = (language or "auto").lower()
    spoken_text = (transcript or "").strip()
    if audio is not None:
        spoken_text = " ".join(part for part in (spoken_text, _transcribe_audio(audio, language)) if part).strip()
    if not spoken_text:
        raise HTTPException(status_code=400, detail="NO_SPEECH")

    known_fields = {
        key: value for key, value in _parse_json_arg(current_fields, {}).items()
        if key in STUDENT_FIELD_KEYS and isinstance(value, str) and value.strip()
    }

    user_prompt = (
        f"Previously collected fields:\n{json.dumps(known_fields, ensure_ascii=False)}\n"
        f"Transcript of the new recording:\n{spoken_text[:4000]}"
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": VOICE_STUDENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        ai_data = json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice Student Error: {str(e)}")

    fields = _normalize_student_fields(ai_data.get("fields"))
    missing = [key for key in STUDENT_FIELD_KEYS if not fields.get(key)]
    follow_up = ai_data.get("followUpQuestion")
    return {
        "transcript": spoken_text,
        "fields": fields,
        "missingFields": missing,
        "followUpQuestion": follow_up.strip() if isinstance(follow_up, str) else "",
        "isComplete": len(missing) == 0,
    }
