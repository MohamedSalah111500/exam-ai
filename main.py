from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import PyPDF2
from openai import OpenAI
import os
import json

# Initialize OpenAI client
#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

app = FastAPI()

# CORS config
origins = ["http://localhost:4200", "https://platx.onrender.com", "https://exam-ai-14pq.onrender.com", "https://platx.net"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ExamRequest(BaseModel):
    language: str
    level: str
    question_count: int
    notes: Optional[str] = None

MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB

def extract_text_from_pdf(file: UploadFile) -> str:
    pdf_reader = PyPDF2.PdfReader(file.file)
    return "".join(page.extract_text() or "" for page in pdf_reader.pages)

def extract_text_from_word(file: UploadFile) -> str:
    doc = docx.Document(file.file)
    return "\n".join([p.text for p in doc.paragraphs])

@app.post("/generate-exam")
async def generate_exam(
    pdf_file: Optional[UploadFile] = None,
    text_content: Optional[str] = Form(None),
    language: str = Form(...),
    level: str = Form(...),
    question_count: str = Form(...),
    notes: str = Form("")
):
    # Validate language and level
    if language not in ["English", "Arabic"]:
        raise HTTPException(status_code=400, detail="Invalid language. Use 'English' or 'Arabic'.")
    if level not in ["easy", "medium", "difficult"]:
        raise HTTPException(status_code=400, detail="Invalid level. Use 'easy', 'medium', or 'difficult'.")

    # Validate question_count
    try:
        question_count_int = int(question_count)
        if question_count_int > 10:
            raise HTTPException(status_code=400, detail="Maximum allowed question_count is 10.")
    except ValueError:
        raise HTTPException(status_code=400, detail="question_count must be an integer.")

    # Must provide either file or text
    if not pdf_file and not text_content:
        raise HTTPException(status_code=400, detail="You must provide either a file or text input.")

    # Extract text from file if uploaded
    text = ""
    if pdf_file:
        if pdf_file.spool_max_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File exceeds maximum size of 2 MB.")
        filename = pdf_file.filename.lower()
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(pdf_file)
        elif filename.endswith(".docx"):
            text = extract_text_from_word(pdf_file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF or DOCX allowed.")
    else:
        text = text_content.strip()

    if not text:
        raise HTTPException(status_code=400, detail="No text found to generate questions.")

    # Prepare prompt for Deepseek
    prompt = f"""
ONLY RETURN JSON.
Generate {question_count_int} exam questions and answers from the following text.
Language: {language}, Difficulty: {level}.
Notes: {notes}

Text:
{text}

Return JSON in this format exactly:
{{
  "questions": [
    {{
      "id": "uniqueId",
      "questionHead": "string",
      "answers": ["string", "string", "string", "string"],
      "correctAnswer": "index of correct answer - int"
    }}
  ]
}}
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an educational content generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )

        response_content = response.choices[0].message.content.strip()
        try:
            questions = json.loads(response_content)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse JSON from AI response. Raw output: {response_content}"
            )

        return JSONResponse(content={"questions": questions})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")