"""
Document Translation Service for TXT and PDF files
Extracts structured paragraphs, translates sequentially using Custom Multilingual Transformer,
and reassembles the translated document preserving paragraph structure.
"""

import time
import io
import fitz  # PyMuPDF
from typing import List, Tuple
from fastapi import UploadFile, HTTPException

from .custom_multilingual_service import custom_multilingual_engine


def extract_text_from_file(file: UploadFile, content: bytes) -> List[str]:
    """
    Extracts ordered text paragraphs from TXT or PDF documents.
    Validates file formats, handles empty files, and provides clear errors.
    """
    filename = (file.filename or "").lower()
    paragraphs: List[str] = []

    if filename.endswith(".txt"):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("latin-1")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to decode text file: {str(e)}")
        raw_paras = text.split("\n\n")
        paragraphs = [p.strip() for p in raw_paras if p.strip()]

    elif filename.endswith(".pdf"):
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            if doc.page_count == 0:
                raise HTTPException(status_code=400, detail="The uploaded PDF document has 0 pages.")

            total_extracted_chars = 0
            for page_idx, page in enumerate(doc):
                blocks = page.get_text("blocks")
                # Sort blocks vertically by y0 coordinate to preserve reading flow
                blocks.sort(key=lambda b: (b[1], b[0]))
                for block in blocks:
                    block_text = block[4].strip()
                    if block_text and len(block_text) > 2:
                        cleaned = " ".join(block_text.split())
                        paragraphs.append(cleaned)
                        total_extracted_chars += len(cleaned)

            if total_extracted_chars == 0:
                raise HTTPException(
                    status_code=400,
                    detail="No extractable text found in this PDF. It may be a scanned or image-only document, which requires OCR."
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF document structure: {str(e)}")
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload .txt or .pdf documents."
        )

    if not paragraphs:
        raise HTTPException(status_code=400, detail="The uploaded document contains no readable paragraphs.")

    return paragraphs


def translate_document(
    paragraphs: List[str],
    source_lang: str,
    target_lang: str,
    max_paragraphs: int = 50
) -> Tuple[str, float]:
    """
    Translates a list of document paragraphs using our custom PyTorch Transformer.
    Limits to max_paragraphs per request to prevent timeouts on massive texts.
    """
    translated_paras: List[str] = []
    start_time = time.perf_counter()

    for p in paragraphs[:max_paragraphs]:
        res = custom_multilingual_engine.translate(
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
