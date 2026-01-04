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
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ExamRequest(BaseModel):
    language: str
    level: str
    question_count: str
    notes: Optional[str] = None

MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB

@app.post("/generate-exam")
async def generate_exam(
    pdf_file: Optional[UploadFile] = None,
    text_content: Optional[str] = Form(None),
    language: str = Form(...),
    level: str = Form(...),
    question_count: str = Form(...),
    notes: Optional[str] = Form(None)
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

    # Validate input: must have either file or text
    if not pdf_file and (not text_content or not text_content.strip()):
        raise HTTPException(status_code=400, detail="You must provide either a PDF file or text input.")

    # Check file size
    if pdf_file:
        pdf_file.file.seek(0, os.SEEK_END)
        size = pdf_file.file.tell()
        pdf_file.file.seek(0)
        if size > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File exceeds maximum size of 2 MB.")

    try:
        # Extract text from PDF if file provided
        if pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file.file)
            text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        else:
            text = text_content.strip()

        if not text:
            raise HTTPException(status_code=400, detail="No text found to generate questions.")

        # Prepare AI prompt
        prompt = (
            f"Extract {question_count} meaningful exam questions and answers from the following text. "
            f"Make them in {language} language. Questions should be {level} difficulty. "
        )
        if notes:
            prompt += f"Notes: {notes}\n"
        prompt += f"\nText:\n{text}\n\n"
        prompt += "Output in JSON format like:\n"
        prompt += """{
  "questions": [
    {
      "id": "uniqueId",
      "questionHead": "string",
      "answers": ["string","string","string","string"],
      "correctAnswer": 0
    }
  ]
}"""

        # Call OpenAI GPT-5 Nano
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an educational content generator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )

        response_content = response.choices[0].message.content
        try:
            questions = json.loads(response_content)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse JSON from AI response: {str(e)}")

        return JSONResponse(content={"questions": questions})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Server is running"}
