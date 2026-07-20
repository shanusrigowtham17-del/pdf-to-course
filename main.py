import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pdf_to_course import extract_text_from_pdf, generate_course_from_text
import google.generativeai as genai  # adjust import to match whatever you use for Gemini calls
from pdf_to_course import extract_text_from_pdf, generate_course_from_text, MAX_CHARS
logger = logging.getLogger("pdf_to_course_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="PDF to Course API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://ai-study-lake.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB cap

# --- Simple in-memory store: swap for Supabase later ---
# NOTE: this resets whenever the server restarts, and doesn't separate
# users/sessions. Fine for a single-user demo, not for production.
document_store = {"text": None, "filename": None}


@app.get("/")
def read_root():
    return {"message": "The PDF Course API is live! Go to /docs to test it."}


@app.post("/api/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_bytes = file.file.read()

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size is {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB."
        )
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info(f"Extracting text from {file.filename}...")
    try:
        pdf_text = extract_text_from_pdf(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("PDF extraction failed")
        raise HTTPException(status_code=422, detail="Could not process this PDF file.")

    # Store it for the chat endpoint to use
    document_store["text"] = pdf_text
    document_store["filename"] = file.filename

    return {"success": True, "filename": file.filename}


class ChatRequest(BaseModel):
    question: str


@app.post("/api/chat")
def chat(payload: ChatRequest):
    if not document_store["text"]:
        raise HTTPException(status_code=400, detail="No document uploaded yet. Please upload a PDF first.")

    try:
        # Reuse your existing Gemini setup — adjust this call to match
        # however generate_course_from_text talks to Gemini internally.
        model = genai.GenerativeModel("gemini-1.5-flash")  # match your existing model choice
        prompt = (
            f"You are answering questions about the following document.\n\n"
            f"DOCUMENT:\n{document_store['text'][:20000]}\n\n"  # crude truncation guard
            f"QUESTION: {payload.question}\n\n"
            f"Answer clearly and concisely based only on the document above."
        )
        response = model.generate_content(prompt)
        answer = response.text
    except Exception:
        logger.exception("Chat generation failed")
        raise HTTPException(status_code=502, detail="Could not generate an answer. Please try again.")

    return {"response": answer}


@app.post("/api/generate-course")
def generate_course(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_bytes = file.file.read()

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size is {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB."
        )
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info(f"Extracting text from {file.filename}...")
    try:
        pdf_text = extract_text_from_pdf(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("PDF extraction failed")
        raise HTTPException(status_code=422, detail="Could not process this PDF file.")

    logger.info("Designing the course with Gemini...")
    try:
        course_data = generate_course_from_text(pdf_text)
    except Exception:
        logger.exception("Course generation failed")
        raise HTTPException(status_code=502, detail="Course generation failed. Please try again.")

    return {"success": True, "course": course_data}
