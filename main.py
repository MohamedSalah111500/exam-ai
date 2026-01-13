from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import docx2txt
from openai import OpenAI
import os
import json
import re
from datetime import datetime
from typing import Optional

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'), 
    base_url="https://api.deepseek.com"
)

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
    # 1. Text Extraction Logic
    source_text = ""
    if pdf_file:
        contents = await pdf_file.read()
        pdf_file.file.seek(0)
        if pdf_file.filename.lower().endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(pdf_file.file)
            source_text = "".join(page.extract_text() for page in pdf_reader.pages)
        elif pdf_file.filename.lower().endswith(".docx"):
            source_text = docx2txt.process(pdf_file.file)

    if text_content and text_content.strip():
        source_text += "\n" + text_content.strip()

    if not source_text.strip():
        raise HTTPException(status_code=400, detail="No source content found.")

    # 2. Refined Prompt
    instruction_notes = f"Note: {notes}" if notes else ""
    prompt = (
        f"SOURCE TEXT:\n{source_text}\n\n"
        f"TASK:\nGenerate {question_count} MCQs in {language} based on the text.\n"
        f"Difficulty: {level}. {instruction_notes}\n\n"
        f"Return JSON ONLY with this structure:\n"
        f"{{\n"
        f"  \"questions\": [\n"
        f"    {{\n"
        f"      \"id\": 1,\n"
        f"      \"questionHead\": \"string\",\n"
        f"      \"answers\": [\"a\", \"b\", \"c\", \"d\"],\n"
        f"      \"correctAnswer\": 0\n"
        f"    }}\n"
        f"  ]\n"
        f"}}"
    )

    # 3. AI Query
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an exam generator. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={'type': 'json_object'},
            temperature=0.3
        )

        response_content = response.choices[0].message.content.strip()
        ai_data = json.loads(response_content)
        
        # Extract the array of questions
        questions_list = ai_data.get("questions", [])

        # 4. Wrap in the "Exam" Interface expected by Frontend
        exam_response = {
            "id": 101,  # Temporary ID
            "name": f"AI Generated Exam - {level}",
            "isAutoCorrect": True,
            "createdBy": 1,
            "creationTime": datetime.now().isoformat(),
            "questions": questions_list,
            "isShowCorrectAnswers": True
        }

        return JSONResponse(content=exam_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))