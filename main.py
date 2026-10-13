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
from urllib.parse import urlparse
from pydantic import BaseModel

# إعداد العميل لـ DeepSeek
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

# Image uploads are read by DeepSeek's multimodal model with the same DeepSeek key by default.
# VISION_* env vars can point to any other OpenAI-compatible vision endpoint instead.
VISION_API_KEY = os.environ.get('VISION_API_KEY') or os.environ.get('DEEPSEEK_API_KEY')
VISION_BASE_URL = os.environ.get('VISION_BASE_URL', 'https://api.deepseek.com')
VISION_MODEL = os.environ.get('VISION_MODEL', 'deepseek-flash')
vision_client = OpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL) if VISION_API_KEY else None

# Speech-to-text (OpenAI-compatible /audio/transcriptions). DeepSeek has no STT, so a
# Whisper-compatible provider is used: Groq (default base URL) or api.openai.com.
STT_API_KEY = os.environ.get('STT_API_KEY')
STT_BASE_URL = os.environ.get('STT_BASE_URL', 'https://api.groq.com/openai/v1')
STT_MODEL = os.environ.get('STT_MODEL', 'whisper-large-v3-turbo')
stt_client = OpenAI(api_key=STT_API_KEY, base_url=STT_BASE_URL) if STT_API_KEY else None

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
        response = client.chat.completions.create(
            model="deepseek-chat",
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
    if vision_client is not None:
        images = [d for d in (_image_data_url(u) for u in image_urls[:MAX_GRADED_IMAGES]) if d]

    try:
        if images:
            image_hint = "\n\nThe student's handwritten or photographed work is in the attached images; read it carefully."
            content = [{"type": "text", "text": base_instruction + image_hint}]
            content += [{"type": "image_url", "image_url": {"url": d}} for d in images]
            response = vision_client.chat.completions.create(
                model=VISION_MODEL,
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
            response = client.chat.completions.create(
                model="deepseek-chat",
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


CHOICE_TYPES = {"single", "multi", "truefalse"}
CONFIDENCE_LEVELS = {"high", "medium", "low"}
MAX_MODEL_ANSWER_IMAGES = 4
MAX_QUESTION_IMAGES = 10

HOMEWORK_V2_RULES = (
    "You are a careful teacher grading one student's homework.\n"
    "The student may answer in the platform (typed text), or print the homework, solve it on paper and upload photos, or both.\n"
    "Questions are numbered exactly as on the printed sheet. Options are labelled with Arabic letters (أ ب ج د ...) "
    "or English letters (A B C D ...); both labels refer to the same option number.\n\n"
    "For each question listed below:\n"
    "- single / multi / truefalse: find in the student's images or documents which option(s) the student marked for that "
    "question number (circled, ticked, underlined, shaded, or a written letter such as '3- ب' or an answer table). "
    "Return them as option numbers in selected_options. Do NOT judge whether the choice is correct and set score to null. "
    "If the student marked nothing for that question, return an empty list.\n"
    "- essay: grade the student's answer (typed text and/or what is written or drawn in the images) from 0 to max_score. "
    "Use the model answer when one is given. For drawings (triangles, graphs, diagrams) judge the drawing itself: correct shape, "
    "labels, measurements and steps. Give partial credit for partially correct work.\n\n"
    "confidence: 'high' only when the student's answer is clearly legible and unambiguous; 'medium' when it is probably right; "
    "'low' when you are guessing or could not find the answer. Never invent an answer that is not visible.\n"
    "feedback: one short sentence per question, written in the language of the questions.\n\n"
    "Return ONLY a JSON object: "
    "{\"questions\": [{\"id\": <question id>, \"selected_options\": [<option numbers>], \"confidence\": \"high|medium|low\", "
    "\"score\": <number or null>, \"feedback\": \"...\"}], \"score\": <overall score or null>, \"feedback\": \"<short overall feedback>\"}"
)

HOMEWORK_UPLOAD_ONLY_RULES = (
    "You are a careful teacher grading one student's homework that was submitted as uploaded files.\n"
    "Grade the work against the teacher's instructions from 0 to the total score. Give partial credit where earned.\n"
    "Return ONLY a JSON object: {\"questions\": [], \"score\": <number>, \"feedback\": \"<short feedback in the language of the instructions>\"}"
)


def _describe_homework_question(q: HomeworkQuestion, has_model_image: bool, own_files: int = 0,
                                skipped_files: bool = False, has_sheet: bool = False) -> str:
    lines = [f"Question {q.number} (id={q.id}, type={q.type}, max_score={q.max_score}): {q.body}"]
    for o in q.options:
        lines.append(f"  option {o.number} [{o.letter_ar} / {o.letter_en}]: {o.text}")
    if q.student_text:
        lines.append(f"  Student typed answer: {q.student_text}")
    if own_files:
        lines.append(f"  The student attached {own_files} file(s) for this question only, labelled 'Answer for question {q.number}'. Read the answer from them first.")
    elif q.answer_by_upload and has_sheet:
        lines.append("  The student may have answered this question on paper; look for it in the student's images.")
    if skipped_files:
        lines.append("  Some of the student's files for this question could not be attached; use confidence 'low' unless the answer is clearly visible.")
    elif q.read_from_images and q.type in CHOICE_TYPES:
        lines.append("  Not answered in the platform; read the marked option from the student's images.")
    if q.model_answer:
        lines.append(f"  Model answer: {q.model_answer}")
    if has_model_image:
        lines.append("  A model answer image for this question is attached and labelled.")
    return "\n".join(lines)


def _normalize_homework_result(data: dict, req: GradeHomeworkV2Request) -> dict:
    by_id = {q.id: q for q in req.questions}
    results = {}
    for item in data.get("questions") or []:
        try:
            qid = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        q = by_id.get(qid)
        if q is None or qid in results:
            continue
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
        results[qid] = {
            "id": qid,
            "selected_options": selected if q.type in CHOICE_TYPES else [],
            "confidence": confidence,
            "score": score,
            "feedback": str(item.get("feedback") or "")[:500],
        }

    for q in req.questions:
        results.setdefault(q.id, {"id": q.id, "selected_options": [], "confidence": "low",
                                  "score": None if q.type in CHOICE_TYPES else 0.0, "feedback": ""})

    overall = None
    if not req.questions:
        try:
            overall = max(0.0, min(float(data.get("score")), float(req.total_score)))
        except (TypeError, ValueError):
            overall = None

    return {"questions": list(results.values()), "score": overall, "feedback": str(data.get("feedback") or "")[:1000]}


@app.post("/grade-homework-v2")
def grade_homework_v2(req: GradeHomeworkV2Request):
    # Plain def: FastAPI runs it in its threadpool, so the slow downloads and model call never block other endpoints.
    image_urls = [u for u in (req.file_urls or []) if _url_ext(u) in IMAGE_MIME_BY_EXT]
    doc_urls = [u for u in (req.file_urls or []) if _url_ext(u) not in IMAGE_MIME_BY_EXT]

    question_images = []  # (question number, data url)
    question_texts = ""
    own_file_count = {}
    skipped = set()
    for q in req.questions:
        for u in q.file_urls:
            room = MAX_QUESTION_IMAGES - len(question_images) if vision_client is not None else 0
            if _url_ext(u) in IMAGE_MIME_BY_EXT:
                d = _image_data_url(u) if room > 0 else None
                if d:
                    question_images.append((q.number, d))
                    own_file_count[q.number] = own_file_count.get(q.number, 0) + 1
                else:
                    skipped.add(q.number)
                continue
            text, pages = _read_document(u, max(0, room))
            question_images.extend((q.number, page) for page in pages)
            if pages or text.strip():
                own_file_count[q.number] = own_file_count.get(q.number, 0) + max(1, len(pages))
            else:
                skipped.add(q.number)
            if text.strip():
                question_texts += f"\n\n[Answer for question {q.number} (document text)]\n{text[:3000]}"

    student_images = []
    if vision_client is not None:
        student_images = [d for d in (_image_data_url(u) for u in image_urls[:MAX_GRADED_IMAGES]) if d]

    documents = ""
    document_pages = []
    for u in doc_urls:
        room = MAX_GRADED_IMAGES - len(student_images) - len(document_pages) if vision_client is not None else 0
        text, pages = _read_document(u, max(0, room))
        document_pages.extend(pages)
        if text.strip():
            documents += f"\n\n[Student document text]\n{text[:6000]}"

    model_images = []
    if vision_client is not None:
        for q in req.questions:
            if q.model_answer_image_url and len(model_images) < MAX_MODEL_ANSWER_IMAGES:
                d = _image_data_url(q.model_answer_image_url)
                if d:
                    model_images.append((q.number, d))

    if req.questions:
        rules = HOMEWORK_V2_RULES
        with_model_image = {number for number, _ in model_images}
        has_sheet = bool(student_images or document_pages or documents)
        listing = "\n\n".join(
            _describe_homework_question(q, q.number in with_model_image, own_file_count.get(q.number, 0),
                                        q.number in skipped, has_sheet) for q in req.questions)
    else:
        rules = HOMEWORK_UPLOAD_ONLY_RULES
        listing = "(no questions; grade the uploaded work as a whole)"

    prompt = (
        f"Homework: {req.homework_name}\nTotal score: {req.total_score}\n"
        f"Teacher instructions: {req.instructions or '-'}\n\n"
        f"=== QUESTIONS ===\n{listing}\n=== END QUESTIONS ==={question_texts}{documents}"
    )
    has_question_files = any(q.file_urls for q in req.questions)
    if (image_urls or doc_urls or has_question_files) and not student_images and not document_pages and not documents \
            and not question_images and not question_texts:
        prompt += "\n\nNote: the student's files could not be read; use confidence 'low' for answers that would be in them."

    try:
        if student_images or document_pages or model_images or question_images:
            content = [{"type": "text", "text": prompt}]
            for number, d in question_images:
                content.append({"type": "text", "text": f"Answer for question {number} (attached by the student to this question):"})
                content.append({"type": "image_url", "image_url": {"url": d}})
            for i, d in enumerate(student_images, start=1):
                content.append({"type": "text", "text": f"Student image {i}:"})
                content.append({"type": "image_url", "image_url": {"url": d}})
            for i, d in enumerate(document_pages, start=1):
                content.append({"type": "text", "text": f"Student PDF page {i} (look for handwriting, drawings and marks on it):"})
                content.append({"type": "image_url", "image_url": {"url": d}})
            for number, d in model_images:
                content.append({"type": "text", "text": f"Model answer image for question {number} (teacher's reference, not the student's work):"})
                content.append({"type": "image_url", "image_url": {"url": d}})
            response = vision_client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{"role": "system", "content": rules}, {"role": "user", "content": content}],
                temperature=0.1,
            )
        else:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": rules}, {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )

        raw = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(match.group(0) if match else raw)
        return _normalize_homework_result(data, req)
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
