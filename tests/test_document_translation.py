"""
Unit Tests for Document Translation Service
Verifies TXT and PDF text extraction, sequential chunk translation,
and error handling on image-only/empty documents.
"""

import io
import pymupdf
from fastapi import UploadFile, HTTPException
import pytest

from backend.app.services.document_service import extract_text_from_file, translate_document


def create_dummy_pdf(text: str) -> bytes:
    """Generates an in-memory PDF with specified text."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_txt_document_extraction_and_translation():
    content = "Hello world.\n\nTranslateMe is an open-source neural machine translation system.".encode("utf-8")
    upload_file = UploadFile(filename="sample.txt", file=io.BytesIO(content))

    paragraphs = extract_text_from_file(upload_file, content)
    assert len(paragraphs) == 2
    assert "Hello world." in paragraphs[0]

    translated_doc, latency = translate_document(paragraphs, source_lang="en", target_lang="hi")
    assert len(translated_doc) > 0
    assert latency >= 0.0


def test_pdf_document_extraction_and_translation():
    pdf_content = create_dummy_pdf("Deep learning enables artificial intelligence systems to learn representations.")
    upload_file = UploadFile(filename="test.pdf", file=io.BytesIO(pdf_content))

    paragraphs = extract_text_from_file(upload_file, pdf_content)
    assert len(paragraphs) >= 1
    assert "Deep learning" in paragraphs[0]

    translated_doc, latency = translate_document(paragraphs, source_lang="en", target_lang="ta")
    assert len(translated_doc) > 0
    assert latency >= 0.0


def test_empty_document_raises_http_exception():
    upload_file = UploadFile(filename="empty.txt", file=io.BytesIO(b""))
    with pytest.raises(HTTPException) as excinfo:
        extract_text_from_file(upload_file, b"")
    assert excinfo.value.status_code == 400
