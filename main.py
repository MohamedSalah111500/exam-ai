from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import PyPDF2
import docx
from openai import OpenAI
import os
import json

# ================= CONFIG =================
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB
ALLOWED_EXTENSIONS = [".pdf", ".docx"]
AI_MODEL = "gpt-5-nano"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ================= HELPERS =================
def validate_file_size(file: UploadFile):
    file.file.seek(0, os.SEEK_END)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File size must not exceed 2MB."
        )

def extract_text_from_pdf(file: UploadFile) -> str:
    reader = PyPDF2.PdfReader(file.file)
    return "".join(page.extract_text() or "" for page in reader.pages)

def extract_text_from_word(file: UploadFile) -> str:
    document = docx.Document(file.file)
    return "\n".join(p.text for p in document.paragraphs)

# ================= ENDPOINT =================

@app.get("/health")
def health_check():
    openai_key_exists = bool(os.environ.get("OPENAI_API_KEY"))

    return {
        "status": "ok",
        "service": "PLATX Exam AI",
        "openai_key": openai_key_exists,
        "timestamp": datetime.utcnow().isoformat()
    }

    
@app.post("/generate-exam")
async def generate_exam(
    file: Optional[UploadFile] = None,
    text_input: Optional[str] = Form(None),
    language: str = Form(...),
    level: str = Form(...),
    question_count: int = Form(...),
    notes: Optional[str] = Form(None),
):
    # -------- Source validation --------
    if not file and not text_input:
        raise HTTPException(
            status_code=400,
            detail="You must provide either a file or text input."
        )

    if file and text_input:
        raise HTTPException(
            status_code=400,
            detail="Provide only one source: file OR text input."
        )

    # -------- Basic validation --------
    if language not in ["English", "Arabic"]:
        raise HTTPException(status_code=400, detail="Invalid language.")

    if level not in ["easy", "medium", "difficult"]:
        raise HTTPException(status_code=400, detail="Invalid difficulty level.")

    if question_count <= 0 or question_count > 10:
        raise HTTPException(
            status_code=400,
            detail="question_count must be between 1 and 10."
        )

    # -------- Extract text --------
    try:
        if file:
            validate_file_size(file)

            filename = file.filename.lower()
            if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                raise HTTPException(
                    status_code=400,
                    detail="Only PDF and Word (.docx) files are allowed."
                )

            if filename.endswith(".pdf"):
                text = extract_text_from_pdf(file)
            else:
                text = extract_text_from_word(file)

        else:
            text = text_input.strip()

        if not text:
            raise HTTPException(
                status_code=400,
                detail="Provided content is empty."
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Text extraction failed: {str(e)}"
        )

    # -------- Notes handling --------
    notes_block = ""
    if notes and notes.strip():
        notes_block = f"""
IMPORTANT INSTRUCTIONS (MUST FOLLOW):
- {notes.strip()}
"""

    # -------- Prompt --------
    prompt = f"""
You are an expert exam generator.

Generate exactly {question_count} multiple-choice questions based ONLY on the given text.

Constraints:
- Language: {language}
- Difficulty: {level}
- 4 answers per question
- Only ONE correct answer
- Do NOT invent facts
{notes_block}

Return ONLY valid JSON using this structure:

{{
  "questions": [
    {{
      "id": "unique-id",
      "questionHead": "string",
      "answers": ["string", "string", "string", "string"],
      "correctAnswer": 0
    }}
  ]
}}

TEXT:
{text}
"""

    # -------- OpenAI Call --------
    response = client.chat.completions.create(
        model=AI_MODEL,
        messages=[
            {"role": "system", "content": "You strictly follow the rules."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=1200,
    )

    content = response.choices[0].message.content.strip()

    try:
        return JSONResponse(content=json.loads(content))
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="AI response was not valid JSON."
        )
