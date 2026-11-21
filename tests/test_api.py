"""
Backend API Tests using FastAPI TestClient
Tests custom multilingual transformer translation, language detection, history, analytics, and attention extraction.
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
    assert "model_engine" in data
    assert "device" in data


def test_languages_endpoint():
    response = client.get("/api/languages")
    assert response.status_code == 200
    langs = response.json()
    assert len(langs) == 4
    codes = [l["code"] for l in langs]
    assert "en" in codes
    assert "hi" in codes
    assert "ta" in codes
    assert "te" in codes


def test_translate_custom_transformer():
    # English to Hindi translation
    response = client.post("/api/translate", json={
        "text": "Hello, how are you today?",
        "source_language": "en",
        "target_language": "hi",
        "num_beams": 1
    })
    assert response.status_code == 200
    data = response.json()
    assert data["original_text"] == "Hello, how are you today?"
    assert len(data["translated_text"]) > 0
    assert data["target_language"] == "hi"
    assert "inference_time" in data
    assert "Custom Multilingual Transformer" in data["model_name"]


def test_translate_multilingual_tamil_telugu():
    # English to Tamil
    res_ta = client.post("/api/translate", json={
        "text": "What is your name?",
        "source_language": "en",
        "target_language": "ta"
    })
    assert res_ta.status_code == 200
    assert len(res_ta.json()["translated_text"]) > 0

    # English to Telugu
    res_te = client.post("/api/translate", json={
        "text": "What is your name?",
        "source_language": "en",
        "target_language": "te"
    })
    assert res_te.status_code == 200
    assert len(res_te.json()["translated_text"]) > 0


def test_translate_validation_errors():
    # Unsupported language rejection
    response = client.post("/api/translate", json={
        "text": "Hello world",
        "source_language": "fr",  # French not trained in custom model
        "target_language": "hi"
    })
    assert response.status_code == 400
    assert "not supported" in response.json()["detail"].lower()


def test_lab_attention_extraction():
    # Test real attention matrix extraction
    response = client.post("/api/lab/attention", json={
        "text": "Hello, how are you today?",
        "source_lang": "en",
        "target_lang": "hi"
    })
    assert response.status_code == 200
    data = response.json()
    assert "attention_matrix" in data
    assert "source_tokens" in data
    assert len(data["source_tokens"]) > 0


def test_model_specs_endpoint():
    response = client.get("/api/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "TranslateMe Custom Multilingual Transformer" in data["model_name"]
    assert "6,033,408" in data["parameters"]
    assert data["layers"] == 3


def test_history_flow():
    # Read history
    res = client.get("/api/history?limit=10")
    assert res.status_code == 200
    assert isinstance(res.json(), list)
