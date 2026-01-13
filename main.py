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

# Initialize Client
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
    # 1. Validation
    if language not in ["English", "Arabic"]:
        raise HTTPException(status_code=400, detail="Invalid language.")
    if level not in ["easy", "medium", "difficult"]:
        raise HTTPException(status_code=400, detail="Invalid level.")
    if not (1 <= question_count <= 10):
        raise HTTPException(status_code=400, detail="Count must be between 1-10.")

    # 2. Extract Text (Corrected Logic)
    source_text = ""

    # Handle File Upload
    if pdf_file:
        contents = await pdf_file.read()
        if len(contents) > 2 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 2MB.")
        
        # Reset file pointer after reading for the libraries to process it
        pdf_file.file.seek(0)

        try:
            if pdf_file.filename.lower().endswith(".pdf"):
                pdf_reader = PyPDF2.PdfReader(pdf_file.file)
                source_text = "".join(page.extract_text() for page in pdf_reader.pages)
            elif pdf_file.filename.lower().endswith(".docx"):
                source_text = docx2txt.process(pdf_file.file)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

    # Handle Manual Text (Append, don't overwrite)
    if text_content and text_content.strip():
        source_text += "\n" + text_content.strip()

    # Final Check
    if not source_text.strip():
        raise HTTPException(status_code=400, detail="No source content found in file or text field.")

    # 3. Build Prompt (Strict Grounding)
    instruction_notes = f"Special Instructions from user: {notes}" if notes else ""
    
    prompt = (
        f"SOURCE TEXT CONTENT:\n{source_text}\n\n"
        f"TASK:\n"
        f"Generate exactly {question_count} multiple-choice questions based ONLY on the SOURCE TEXT above.\n"
        f"1. Language: {language}\n"
        f"2. Difficulty: {level}\n"
        f"3. {instruction_notes}\n"
        f"4. Do not use external knowledge. If the text is insufficient, extract what is possible.\n\n"
        f"Return the questions in this JSON format ONLY:\n"
        f"{{\n"
        f"  \"questions\": [\n"
        f"    {{\n"
        f"      \"id\": \"unique_string\",\n"
        f"      \"questionHead\": \"string\",\n"
        f"      \"answers\": [\"option 0\", \"option 1\", \"option 2\", \"option 3\"],\n"
        f"      \"correctAnswer\": 0\n"
        f"    }}\n"
        f"  ]\n"
        f"}}"
    )

    # 4. Query AI
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a precise educational assistant. You only output valid JSON based on provided text."},
                {"role": "user", "content": prompt}
            ],
            response_format={'type': 'json_object'}, # Forces JSON output
            max_tokens=2000,
            temperature=0.3 # Lower temperature for better factual accuracy
        )

        response_content = response.choices[0].message.content.strip()

        # Clean JSON if model included markdown blocks
        if "```" in response_content:
            response_content = re.sub(r"```(?:json)?\n?|\n?```", "", response_content).strip()

        questions_data = json.loads(response_content)
        return JSONResponse(content=questions_data)

    except json.JSONDecodeError:
        return JSONResponse(status_code=500, content={"error": "AI returned invalid JSON", "raw": response_content})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing Error: {str(e)}")