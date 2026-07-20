    import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware # <-- 1. IMPORTED CORS HERE
from pdf_to_course import extract_text_from_pdf, generate_course_from_text

logger = logging.getLogger("pdf_to_course_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="PDF to Course API")

# <-- 2. ADDED THE SECURITY EXCEPTION BLOCK HERE
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://ai-study-lake.vercel.app" # Your trusted Vercel domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB cap, adjust as needed

@app.get("/")
def read_root():
    return {"message": "The PDF Course API is live! Go to /docs to test it."}

# <-- 3. ENSURE YOUR FRONTEND FETCHES THIS EXACT PATH
@app.post("/api/generate-course") 
def generate_course(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Read the uploaded PDF file
    file_bytes = file.file.read()

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size is {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB."
        )

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 1. Extract text using PyMuPDF
    logger.info(f"Extracting text from {file.filename}...")
    try:
        pdf_text = extract_text_from_pdf(file_bytes)
    except ValueError as e:
        # Raised for empty/unreadable PDFs (e.g. scanned/image-only)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("PDF extraction failed")
        raise HTTPException(status_code=422, detail="Could not process this PDF file.")

    # 2. Generate the course using Gemini
    logger.info("Designing the course with Gemini...")
    try:
        course_data = generate_course_from_text(pdf_text)
    except Exception as e:
        logger.exception("Course generation failed")
        raise HTTPException(status_code=502, detail="Course generation failed. Please try again.")

    return {"success": True, "course": course_data}
