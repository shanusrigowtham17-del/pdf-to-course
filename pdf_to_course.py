#!/usr/bin/env python3
"""Create a structured learning course from the text in a PDF.

Usage:
    python pdf_to_course.py path/to/book.pdf --output generated_course.json

Install dependencies first:
    python -m pip install --upgrade openai pydantic PyMuPDF
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF
from openai import APIError, OpenAI
from pydantic import BaseModel, ConfigDict, Field


MAX_DOCUMENT_CHARS = 1_500_000


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Lesson(StrictModel):
    title: str
    explanations: str = Field(
        description="Clear, well-structured explanation of the lesson concepts."
    )
    key_takeaways: list[str]
    important_notes: list[str]
    real_world_examples: list[str]
    summary: str


class Subtopic(StrictModel):
    title: str
    lessons: list[Lesson]


class Topic(StrictModel):
    title: str
    subtopics: list[Subtopic]


class Chapter(StrictModel):
    title: str
    topics: list[Topic]


class Course(StrictModel):
    course_title: str
    course_description: str
    estimated_learning_time: str
    learning_objectives: list[str]
    prerequisites: list[str]
    difficulty_level: Literal["Beginner", "Intermediate", "Advanced"]
    table_of_contents: list[str]
    chapters: list[Chapter]


SYSTEM_PROMPT = """
You are a senior instructional designer. Turn the source text into one accurate,
standalone learning course for a motivated learner.

Rules:
- Treat the source text as the source of truth. Do not invent unsupported facts,
  definitions, procedures, references, or source-specific examples.
- If the source is incomplete, ambiguous, or contradictory, explain that in
  important_notes.
- Create a logical progression from foundations to advanced application.
- Use specific, meaningful titles. table_of_contents must exactly match chapter
  titles and order.
- Explanations must teach, not merely list headings.
- Use source examples when available; otherwise use clearly labelled generic examples.
- Learning objectives must be observable.
- Return only data matching the supplied schema.
""".strip()

USER_PROMPT = """
Create the complete course from the document text below. Every lesson must include:
- well-structured explanations
- key takeaways
- important notes
- real-world examples
- a concise summary

The result must be complete and internally consistent.
""".strip()


def validate_pdf(pdf_path: Path) -> None:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, received: {pdf_path.name}")

    if pdf_path.stat().st_size == 0:
        raise ValueError("The PDF is empty.")


def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        with fitz.open(pdf_path) as document:
            pages = []

            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text("text", sort=True).strip()

                if page_text:
                    pages.append(f"--- Page {page_number} ---\n{page_text}")

    except Exception as error:
        raise RuntimeError(f"Failed to read PDF text: {error}") from error

    if not pages:
        raise RuntimeError(
            "No selectable text was found. The PDF may be scanned and require OCR."
        )

    return "\n\n".join(pages)


def generate_course_from_pdf(
    client: OpenAI,
    pdf_path: Path,
    model: str,
) -> Course:
    print(f"Extracting text from {pdf_path.name}...")
    document_text = extract_text_from_pdf(pdf_path)

    if len(document_text) > MAX_DOCUMENT_CHARS:
        raise ValueError(
            f"Extracted text exceeds the {MAX_DOCUMENT_CHARS:,}-character limit. "
            "Split the PDF or implement chunking before sending it to the model."
        )

    print("Designing the course...")

    completion = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{USER_PROMPT}\n\nDocument text:\n{document_text}",
            },
        ],
        response_format=Course,
    )

    message = completion.choices[0].message

    if message.refusal:
        raise RuntimeError(f"The model refused the request: {message.refusal}")

    if message.parsed is None:
        raise RuntimeError("The model did not return a parsed course.")

    return message.parsed


def write_course(course: Course, output_path: Path, overwrite: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(course.model_dump_json(indent=2), encoding="utf-8")
    temporary_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a text-based PDF into a structured learning course."
    )

    parser.add_argument("pdf", type=Path, help="Path to the source PDF.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generated_course.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-2024-08-06"),
        help="OpenAI model to use.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set.")

        validate_pdf(args.pdf)

        course = generate_course_from_pdf(
            client=OpenAI(),
            pdf_path=args.pdf,
            model=args.model,
        )

        write_course(course, args.output, args.overwrite)

        print(f"Success: '{course.course_title}' saved to {args.output}")
        return 0

    except (APIError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
