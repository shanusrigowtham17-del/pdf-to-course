import fitz  # PyMuPDF
import google.generativeai as genai
import os
import json

# Configure the Gemini API client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Rough safety cap to avoid blowing past the model's context window on huge PDFs.
# ~4 chars/token is a reasonable rule of thumb for English text.
MAX_CHARS = 400_000


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts text from a PDF file byte stream."""
    text = ""
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")

    if not text.strip():
        raise ValueError("No extractable text found in PDF (it may be scanned/image-only).")

    return text


def generate_course_from_text(text: str) -> dict:
    """Sends the extracted text to Gemini and returns a structured JSON course."""
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    model = genai.GenerativeModel('gemini-3.5-flash')

    prompt = f"""
    You are an expert course creator. Based on the following extracted text from a PDF, 
    generate a highly structured course outline in JSON format. 
    The JSON must contain a 'course_title', a 'description', and a list of 'modules'. 
    Each module should have a 'title' and a list of 'lessons'.
    
    PDF Text:
    {text}
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
    except Exception as e:
        raise RuntimeError(f"Gemini API request failed: {e}")

    if not response.candidates or not response.text:
        raise ValueError("Gemini returned an empty response (possibly blocked by safety filters).")

    try:
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini did not return valid JSON: {e}\nRaw response: {response.text[:500]}")
