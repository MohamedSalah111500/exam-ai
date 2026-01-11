from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os, json, uuid, re
import PyPDF2
import docx2txt
from openai import OpenAI

# ================= CONFIG =================
app = FastAPI(title="AI CV Matcher - DeepSeek")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

# ================= HELPERS =================
def read_pdf(path: str) -> str:
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def read_docx(path: str) -> str:
    return docx2txt.process(path)

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def extract_json(raw: str) -> dict:
    """
    Remove ```json ``` and parse safely
    """
    raw = raw.strip()

    if raw.startswith("```"):
        raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        return json.loads(raw)
    except Exception as e:
        raise ValueError(f"Invalid JSON from AI: {raw}")

# ================= AI =================
def analyze_cv_with_ai(cv_text: str, job_desc: str) -> dict:
    prompt = f"""
You are an ATS system.

Compare the CV with the Job Description.
Return ONLY valid JSON (no markdown).

JSON format:
{{
  "score": number from 0 to 100,
  "comment": "short professional comment"
}}

Job Description:
{job_desc}

CV:
{cv_text[:6000]}
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw_output = response.choices[0].message.content
    return extract_json(raw_output)

# ================= API =================
@app.post("/analyze-cvs")
async def analyze_cvs(
    files: List[UploadFile] = File(...),
    job_description: str = Form(...),
    notes: str = Form("")
):
    if not files:
        raise HTTPException(400, "No CVs uploaded")

    if len(files) > 20:
        raise HTTPException(400, "Maximum 20 CVs allowed")

    results = []

    for file in files:
        ext = file.filename.split(".")[-1].lower()
        file_id = str(uuid.uuid4())
        path = os.path.join(UPLOAD_DIR, f"{file_id}.{ext}")

        with open(path, "wb") as f:
            f.write(await file.read())

        if ext == "pdf":
            text = read_pdf(path)
        elif ext in ["docx", "doc"]:
            text = read_docx(path)
        else:
            raise HTTPException(400, f"Unsupported file: {file.filename}")

        text = clean_text(text)

        try:
            ai_result = analyze_cv_with_ai(text, job_description)
        except Exception as e:
            raise HTTPException(500, str(e))

        results.append({
            "filename": file.filename,
            "score": ai_result["score"],
            "comment": ai_result["comment"]
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "total": len(results),
        "job_description": job_description,
        "notes": notes,
        "ranked_cvs": results
    }

# ================= HEALTH =================
@app.get("/health")
def health():
    return {"status": "ok"}
