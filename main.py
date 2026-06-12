from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import docx2txt
from openai import OpenAI
import os
import json
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel

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