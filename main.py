from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import docx2txt
from openai import OpenAI
import os
import io
import re
import json
import requests
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel

# إعداد العميل لـ DeepSeek
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

# Optional vision-capable client (OpenAI-compatible). Set these env vars to enable
# grading of image uploads. Works with GPT-4o (api.openai.com), or any OpenAI-compatible
# VLM endpoint. When unset, image uploads fall back to teacher review.
VISION_API_KEY = os.environ.get('VISION_API_KEY')
VISION_BASE_URL = os.environ.get('VISION_BASE_URL', 'https://api.openai.com/v1')
VISION_MODEL = os.environ.get('VISION_MODEL', 'gpt-4o')
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

@app.post("/generate-exam")
async def generate_exam(
    pdf_file: Optional[UploadFile] = None,
    text_content: Optional[str] = Form(None),
    language: str = Form(...),
    level: str = Form(...),
    question_count: int = Form(...),
    notes: Optional[str] = Form(None)
):
    # 1. Extract Text
    source_text = ""
    if pdf_file:
        contents = await pdf_file.read()
        pdf_file.file.seek(0)
        try:
            if pdf_file.filename.lower().endswith(".pdf"):
                pdf_reader = PyPDF2.PdfReader(pdf_file.file)
                source_text = "".join(page.extract_text() for page in pdf_reader.pages)
            elif pdf_file.filename.lower().endswith(".docx"):
                source_text = docx2txt.process(pdf_file.file)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

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
# Text files (PDF / Word) are extracted and graded by DeepSeek; image files are graded
# by a vision model when one is configured (see VISION_* env vars).
# ----------------------------------------------------------------------


class GradeHomeworkRequest(BaseModel):
    homework_name: str
    total_score: float
    # Rubric + the student's answers, already flattened to text by PlatX.
    rubric: str
    # URLs of the student's uploaded files (PlatX-hosted).
    file_urls: List[str] = []
    language: str = "auto"


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")


def _extract_file_text(url: str) -> str:
    """Download a PDF/Word file and return its text (best-effort)."""
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.content
        lower = url.lower()
        if lower.endswith(".pdf"):
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            return "".join(page.extract_text() or "" for page in reader.pages)
        if lower.endswith(".docx"):
            return docx2txt.process(io.BytesIO(data)) or ""
    except Exception:
        return ""
    return ""


@app.post("/grade-homework")
async def grade_homework(req: GradeHomeworkRequest):
    if not req.rubric or not req.rubric.strip():
        raise HTTPException(status_code=400, detail="Missing rubric / student work to grade.")

    image_urls = [u for u in (req.file_urls or []) if u.lower().endswith(IMAGE_EXTS)]
    doc_urls = [u for u in (req.file_urls or []) if not u.lower().endswith(IMAGE_EXTS)]

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

    try:
        # 2) If the student uploaded images and a vision model is configured, use it.
        if image_urls and vision_client is not None:
            content = [{"type": "text", "text": base_instruction}]
            for u in image_urls[:6]:
                content.append({"type": "image_url", "image_url": {"url": u}})
            response = vision_client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional teacher. Return valid JSON only."},
                    {"role": "user", "content": content},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
        else:
            note = ""
            if image_urls and vision_client is None:
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

        data = json.loads(response.choices[0].message.content.strip())
        score = float(data.get("score", 0) or 0)
        score = max(0.0, min(score, float(req.total_score)))
        return {"score": score, "feedback": data.get("feedback", "")}
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
