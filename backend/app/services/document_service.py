"""
Document Translation Service for TXT and PDF files
Extracts structured paragraphs, translates sequentially, and reassembles.
"""

import time
import io
import fitz  # PyMuPDF
from typing import List, Tuple
from fastapi import UploadFile, HTTPException

from .nllb_service import translation_engine


def extract_text_from_file(file: UploadFile, content: bytes) -> List[str]:
    """
    Extracts paragraphs from TXT or PDF documents.
    """
    filename = file.filename.lower()
    paragraphs: List[str] = []

    if filename.endswith(".txt"):
        text = content.decode("utf-8", errors="replace")
        raw_paras = text.split("\n\n")
        paragraphs = [p.strip() for p in raw_paras if p.strip()]

    elif filename.endswith(".pdf"):
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            for page in doc:
                blocks = page.get_text("blocks")
                for block in blocks:
                    block_text = block[4].strip()
                    if block_text and len(block_text) > 3:
                        paragraphs.append(block_text.replace("\n", " "))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF document: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload .txt or .pdf files.")

    if not paragraphs:
        raise HTTPException(status_code=400, detail="The uploaded document contains no readable text.")

    return paragraphs


def translate_document(
    paragraphs: List[str],
    source_lang: str,
    target_lang: str,
    max_paragraphs: int = 50
) -> Tuple[str, float]:
    """
    Translates a list of document paragraphs using the NLLB-200 engine.
    Limits to max_paragraphs per request to prevent timeouts on huge books.
    """
    translated_paras: List[str] = []
    start_time = time.perf_counter()

    for p in paragraphs[:max_paragraphs]:
        res = translation_engine.translate(
            text=p,
            source_lang=source_lang,
            target_lang=target_lang,
            num_beams=1,
            return_alternatives=False
        )
        translated_paras.append(res["translated_text"])

    total_time = time.perf_counter() - start_time
    full_translated_doc = "\n\n".join(translated_paras)
    return full_translated_doc, round(total_time, 2)
