from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import PyPDF2
from openai import OpenAI

import os
import json

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
)
app = FastAPI()

class ExamRequest(BaseModel):
    language: str  # "English" or "Arabic"
    level: str     # "easy", "medium", "difficult"
    question_count: str

@app.post("/generate-exam")
async def generate_exam(
    pdf_file: UploadFile,
    language: str = Form(...),
    level: str = Form(...),
    question_count: str = Form(...),

):
    # Validate inputs
    if language not in ["English", "Arabic"]:
        raise HTTPException(status_code=400, detail="Invalid language. Use 'English' or 'Arabic'.")
    if level not in ["easy", "medium", "difficult"]:
        raise HTTPException(status_code=400, detail="Invalid level. Use 'easy', 'medium', or 'difficult'.")

    try:
        # Extract text from the uploaded PDF
        pdf_reader = PyPDF2.PdfReader(pdf_file.file)
        text = "".join(page.extract_text() for page in pdf_reader.pages)

        if not text.strip():
            raise HTTPException(status_code=400, detail="Failed to extract text from the PDF.")

        # Prepare the prompt for gpt-3.5-turbo
        prompt = (
                    f"Extract meaningful exam 2 questions and answers from the following text. "
                    f"Make the {question_count} questions in this language: {language}. "
                    f"Questions should be of {level} difficulty.\n\n"
                    f"Here is the text:\n{text}\n\n"
                    f"Make sure to put the questions in this JSON format:\n"
                    f"{{\n"
                    f"    \"questions\": [\n"
                    f"        {{\n"
                    f"            \"id\": \"uniqueId\",\n"
                    f"            \"questionHead\": \"string\",\n"
                    f"            \"answers\": [\"string\", \"string\", \"string\", \"string\"],\n"
                    f"            \"correctAnswer\": \"index of correct answer - int\"\n"
                    f"        }}\n"
                    f"    ]\n"
                    f"}}"
                    f"Just give me the json format as a response.\n"
                )
        # Query gpt-3.5-turbo for question generation
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an educational content generator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )

        # Parse response using JSON parser
        response_content = response.choices[0].message.content
        print (response_content)
        try:
            questions = json.loads(response_content)  # Use json.loads to safely parse JSON
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse JSON: {str(e)}")

        return JSONResponse(content={"questions": questions})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
