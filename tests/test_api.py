"""
Backend API Tests using FastAPI TestClient
Tests translation, detection, languages, history, analytics, and error handling.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data


def test_languages_endpoint():
    response = client.get("/api/languages")
    assert response.status_code == 200
    langs = response.json()
    assert len(langs) > 20
    codes = [l["code"] for l in langs]
    assert "eng_Latn" in codes
    assert "hin_Deva" in codes
    assert "tam_Taml" in codes
    assert "tel_Telu" in codes


def test_language_detection():
    # Test Devanagari script detection
    response = client.post("/api/detect-language", json={"text": "नमस्ते आप कैसे हैं"})
    assert response.status_code == 200
    data = response.json()
    assert data["detected_code"] == "hin_Deva"

    # Test English detection
    response = client.post("/api/detect-language", json={"text": "Deep learning architectures have revolutionized NLP."})
    assert response.status_code == 200
    data = response.json()
    assert data["detected_code"] == "eng_Latn"


def test_translate_validation_errors():
    # Empty text error
    response = client.post("/api/translate", json={
        "text": "   ",
        "source_language": "eng_Latn",
        "target_language": "hin_Deva"
    })
    assert response.status_code == 422 or response.status_code == 400

    # Unsupported language code
    response = client.post("/api/translate", json={
        "text": "Hello world",
        "source_language": "unsupported_XYZ",
        "target_language": "hin_Deva"
    })
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_real_translation_and_history():
    # Real translation: English to Hindi
    response = client.post("/api/translate", json={
        "text": "Artificial Intelligence is transforming translation.",
        "source_language": "eng_Latn",
        "target_language": "hin_Deva",
        "num_beams": 2
    })
    assert response.status_code == 200
    data = response.json()
    assert "translated_text" in data
    assert len(data["translated_text"]) > 0
    assert data["source_language"] == "eng_Latn"
    assert data["target_language"] == "hin_Deva"
    assert data["inference_time"] > 0

    # Verify recorded in history
    hist_resp = client.get("/api/history")
    assert hist_resp.status_code == 200
    history_items = hist_resp.json()
    assert len(history_items) > 0
    latest = history_items[0]
    assert latest["source_language"] == "eng_Latn"

    # Test deleting history entry
    del_resp = client.delete(f"/api/history/{latest['id']}")
    assert del_resp.status_code == 200


def test_model_info():
    response = client.get("/api/model/info")
    assert response.status_code == 200
    info = response.json()
    assert "NLLB-200" in info["model_name"]
    assert "615 Million" in info["num_parameters"]
    assert len(info["limitations"]) > 0


def test_analytics_endpoint():
    response = client.get("/api/analytics")
    assert response.status_code == 200
    analytics = response.json()
    assert "total_translations" in analytics
    assert "top_source_languages" in analytics
