import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

# Import the logic and data schemas we built in the previous file
from pdf_to_course import Course, generate_course_from_pdf

app = FastAPI(title="AI Course Generator API", version="1.0.0")

# Configure CORS to allow your Next.js frontend to communicate with this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to your specific Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/api/generate-course", response_model=Course)
async def generate_course_endpoint(file: UploadFile = File(...)):
    """
    Accepts a PDF file, processes it through the OpenAI structured output engine,
    and returns a deeply nested JSON course curriculum.
    """
    
    # Validate the file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")
    
    # Create a temporary directory to store the uploaded file
    temp_dir = Path("/tmp/course_generator")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_pdf_path = temp_dir / file.filename
    
    try:
        # Save the incoming file to the disk temporarily
        with open(temp_pdf_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Pass the file path to your core generation logic
        # Using the reliable 08-06 model for strict JSON compliance
        course_data = generate_course_from_pdf(
            client=client,
            pdf_path=temp_pdf_path,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-2024-08-06")
        )
        
        # FastAPI automatically converts the Pydantic 'Course' object into JSON
        return course_data

    except Exception as error:
        # Catch any errors (e.g., token limits, read failures) and send to frontend
        raise HTTPException(status_code=500, detail=str(error))
        
    finally:
        # Clean up: Delete the temporary PDF so the server doesn't run out of storage
        if temp_pdf_path.exists():
            temp_pdf_path.unlink()

# This block allows you to run the server locally for testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
