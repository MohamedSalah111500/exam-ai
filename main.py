from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import docx2txt
from openai import OpenAI
import os
import json
from datetime import datetime
from typing import Optional

# إعداد العميل لـ DeepSeek
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
    # 1. استخراج النص من الملفات أو الحقول النصية
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

    # 2. بناء البرومبت ليتوافق مع دالة mapAiQuestions
    # ملاحظة: دالة الـ map تطلب "id" لتحويله لـ Int وتطلب "questionHead"
    instruction_notes = f"ملاحظات إضافية من المستخدم: {notes}" if notes else ""
    
    prompt = (
        f"SOURCE TEXT:\n{source_text}\n\n"
        f"TASK:\nGenerate exactly {question_count} multiple-choice questions in {language}.\n"
        f"Difficulty Level: {level}.\n"
        f"{instruction_notes}\n\n"
        f"CRITICAL: Use ONLY the provided text. Return valid JSON with this EXACT structure:\n"
        f"{{\n"
        f"  \"questions\": [\n"
        f"    {{\n"
        f"      \"id\": \"1\",\n"  # نرسله كـ String لأن الدالة تستخدم parseInt
        f"      \"questionHead\": \"نص السؤال هنا\",\n"
        f"      \"answers\": [\"choice 1\", \"choice 2\", \"choice 3\", \"choice 4\"],\n"
        f"      \"correctAnswer\": 0\n" # رقم الـ index للخيار الصحيح (0-3)
        f"    }}\n"
        f"  ]\n"
        f"}}"
    )

    # 3. طلب البيانات من الذكاء الاصطناعي
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a professional exam creator for PlatX platform. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={'type': 'json_object'},
            temperature=0.3
        )

        response_content = response.choices[0].message.content.strip()
        ai_data = json.loads(response_content)
        questions_list = ai_data.get("questions", [])

        # 4. تغليف البيانات في هيكل الـ Exam المتوقع من الفرونت-اند
        exam_response = {
            "id": 0, # سيتم توليده في قاعدة البيانات لاحقاً
            "name": f"اختبار ذكي - {level}",
            "isAutoCorrect": True,
            "createdBy": 1,
            "creationTime": datetime.now().isoformat(),
            "questions": questions_list,
            "isShowCorrectAnswers": True
        }

        return JSONResponse(content=exam_response)

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="فشل الذكاء الاصطناعي في تنسيق البيانات بشكل صحيح.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))