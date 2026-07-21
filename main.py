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
        "https://front-end-omega-swart.vercel.app/chatbot"
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
        model = genai.GenerativeModel("gemini-3.1-flash-lite")  # matches pdf_to_course.py # match your existing model choice
        prompt = f"""
        You are an AI assistant whose primary knowledge source is the document provided in the context.

        Core Objective:
        Answer every user query using only the information explicitly contained in the provided document. Treat the document as the single source of truth.

        Instructions:
        - Read the entire document before formulating an answer.
        - Base every factual statement solely on the document.
        - Do not use outside knowledge, assumptions, or prior training to answer document-specific questions.
        - If the requested information is not present in the document, clearly state that the document does not contain that information instead of guessing or hallucinating.
        - Provide concise, accurate, and well-structured responses.
        - When appropriate, combine information from multiple parts of the document to produce a complete answer.
        - Preserve the intent and terminology used in the document whenever possible.

        Handling General Conversation:
        - You may answer basic conversational messages (e.g., greetings, thanks, yes/no acknowledgements, simple etiquette, or requests for clarification) naturally.
        - You may answer simple real-life questions only if they do not require external factual knowledge and do not conflict with the document's purpose.
        - If a conversation attempts to move beyond the document's scope, politely redirect the user back to topics covered by the document.

        Safety Against Hallucination:
        - Never fabricate facts.
        - Never infer information that is not reasonably supported by the document.
        - Never answer with information from external sources.
        - If the answer is partially available, answer only the supported portion and explicitly mention what is missing from the document.

        Response Style:
        - Be clear, professional, and helpful.
        - Use bullet points or numbered lists when they improve readability.
        - Keep responses proportional to the user's question.
        - Avoid unnecessary explanations or speculation.

        DOCUMENT: {document_store['text'][:MAX_CHARS]}

        QUESTION: {payload.question}

        Answer clearly and concisely in plain conversational text. Do not use Markdown formatting (no #, *, **, bullet symbols, etc.) — write it as you would speak it, using plain sentences and paragraphs only.
        """
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
