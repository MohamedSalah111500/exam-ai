from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import PyPDF2
import docx2txt
from openai import OpenAI
import os
import json
import re

# Initialize OpenAI client
#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:4200",
    "https://platx.onrender.com",
    "https://exam-ai-14pq.onrender.com",
    "https://platx.net"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for validation
class ExamRequest(BaseModel):
    language: str  # "English" or "Arabic"
    level: str     # "easy", "medium", "difficult"
    question_count: int
    notes: Optional[str] = None

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/generate-exam")
async def generate_exam(
    pdf_file: Optional[UploadFile] = None,
    text_content: Optional[str] = Form(None),
    language: str = Form(...),
    level: str = Form(...),
    question_count: int = Form(...),
    notes: Optional[str] = Form(None)
):
    # Validate inputs
    if language not in ["English", "Arabic"]:
        raise HTTPException(status_code=400, detail="Invalid language. Use 'English' or 'Arabic'.")
    if level not in ["easy", "medium", "difficult"]:
        raise HTTPException(status_code=400, detail="Invalid level. Use 'easy', 'medium', or 'difficult'.")
    if question_count < 1 or question_count > 10:
        raise HTTPException(status_code=400, detail="question_count must be between 1 and 10.")

    # Validate that either file or text is provided
    # if not pdf_file and not text_content:
    #     raise HTTPException(status_code=400, detail="You must provide either a file or text input.")

    # Extract text from file if uploaded
    text = ""
    if pdf_file:
        # File size validation (max 2MB)
        contents = await pdf_file.read()
        if len(contents) > 2 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 2MB limit.")
        
        if pdf_file.filename.lower().endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(pdf_file.file)
            text = "".join(page.extract_text() for page in pdf_reader.pages)
        elif pdf_file.filename.lower().endswith(".docx"):
            text = docx2txt.process(pdf_file.file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF or DOCX allowed.")

    if text_content:
        text = text_content.strip()

    if not text:
        raise HTTPException(status_code=400, detail="No text found to generate questions.")

    # Build prompt
    prompt = (
        f"Extract meaningful exam {question_count} questions and answers from the following text. "
        f"Use the language: {language}. "
        f"Questions should be of {level} difficulty. "
        f"{notes or ''}\n\n"
        f"Text:\n{text}\n\n"
        f"Return the questions in this JSON format ONLY:\n"
        f"{{\n"
        f"  \"questions\": [\n"
        f"    {{\n"
        f"      \"id\": \"uniqueId\",\n"
        f"      \"questionHead\": \"string\",\n"
        f"      \"answers\": [\"string\", \"string\", \"string\", \"string\"],\n"
        f"      \"correctAnswer\": int\n"
        f"    }}\n"
        f"  ]\n"
        f"}}"
    )

    try:
        # Query AI
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an educational content generator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )

        response_content = response.choices[0].message.content.strip()

        # Strip Markdown code block if exists
        match = re.search(r"```(?:json)?\n(.*)```", response_content, re.DOTALL)
        if match:
            response_content = match.group(1).strip()

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