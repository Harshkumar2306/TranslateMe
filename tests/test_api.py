"""
Backend API Tests using FastAPI TestClient
Tests custom multilingual transformer translation, auto-detection, history, analytics,
document upload translation, model specs, and Deep Learning Lab endpoints.
Uses an isolated temporary SQLite database so tests never mutate user history.
"""

import io
import os
import tempfile
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.main import app
from backend.app.database.session import Base, get_db

_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp_db.close()
_test_engine = create_engine(
    f"sqlite:///{_tmp_db.name}",
    connect_args={"check_same_thread": False},
)
_TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_test_engine)
Base.metadata.create_all(bind=_test_engine)


def _override_get_db():
    db = _TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = _override_get_db


@pytest.fixture(scope="module")
def api_client():
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    _test_engine.dispose()
    if os.path.exists(_tmp_db.name):
        os.remove(_tmp_db.name)


def test_health_endpoint(api_client):
    response = api_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_engine" in data
    assert "device" in data
    assert "device_type" in data["device"]


def test_languages_endpoint(api_client):
    response = api_client.get("/api/languages")
    assert response.status_code == 200
    langs = response.json()
    assert len(langs) == 4
    codes = [l["code"] for l in langs]
    assert set(codes) == {"en", "hi", "ta", "te"}
    for lang in langs:
        assert lang["iso"]


def test_detect_language_endpoint(api_client):
    res_hi = api_client.post("/api/detect-language", json={"text": "नमस्ते दुनिया"})
    assert res_hi.status_code == 200
    assert res_hi.json()["detected_code"] == "hi"

    res_ta = api_client.post("/api/detect-language", json={"text": "வணக்கம்"})
    assert res_ta.status_code == 200
    assert res_ta.json()["detected_code"] == "ta"


def test_translate_custom_transformer_and_auto_detect(api_client):
    response = api_client.post("/api/translate", json={
        "text": "Hello, how are you today?",
        "source_language": "en",
        "target_language": "hi",
        "num_beams": 2,
        "return_alternatives": True,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["original_text"] == "Hello, how are you today?"
    assert len(data["translated_text"]) > 0
    assert data["target_language"] == "hi"
    assert "inference_time" in data
    assert "Custom Multilingual Transformer" in data["model_name"]

    auto_res = api_client.post("/api/translate", json={
        "text": "नमस्ते, आज आप कैसे हैं?",
        "source_language": "auto",
        "target_language": "en",
        "num_beams": 1,
    })
    assert auto_res.status_code == 200
    auto_data = auto_res.json()
    assert auto_data["source_language"] == "hi"
    assert auto_data["detected_language"] == "hi"
    assert len(auto_data["translated_text"]) > 0


def test_translate_multilingual_tamil_telugu_and_indic_to_indic(api_client):
    res_ta = api_client.post("/api/translate", json={
        "text": "What is your name?",
        "source_language": "en",
        "target_language": "ta",
    })
    assert res_ta.status_code == 200
    assert len(res_ta.json()["translated_text"]) > 0

    res_te = api_client.post("/api/translate", json={
        "text": "What is your name?",
        "source_language": "en",
        "target_language": "te",
    })
    assert res_te.status_code == 200
    assert len(res_te.json()["translated_text"]) > 0

    res_hi_ta = api_client.post("/api/translate", json={
        "text": "नमस्ते, आज आप कैसे हैं?",
        "source_language": "hi",
        "target_language": "ta",
    })
    assert res_hi_ta.status_code == 200
    assert len(res_hi_ta.json()["translated_text"]) > 0


def test_translate_validation_errors(api_client):
    response = api_client.post("/api/translate", json={
        "text": "Hello world",
        "source_language": "fr",
        "target_language": "hi",
    })
    assert response.status_code == 400
    assert "not supported" in response.json()["detail"].lower()


def test_lab_attention_extraction(api_client):
    response = api_client.post("/api/lab/attention", json={
        "text": "Hello, how are you today?",
        "source_lang": "en",
        "target_lang": "hi",
    })
    assert response.status_code == 200
    data = response.json()
    assert "attention_matrix" in data
    assert len(data["attention_matrix"]) > 0
    assert "source_tokens" in data
    assert "target_tokens" in data
    assert len(data["source_tokens"]) > 0
    assert len(data["target_tokens"]) == len(data["attention_matrix"])
    assert data["decoder_layer"] in (4, 6)
    assert data["num_heads"] in (6, 12)


def test_lab_info_real_checkpoint_metrics(api_client):
    response = api_client.get("/api/lab/info")
    assert response.status_code == 200
    data = response.json()
    ct = data["custom_transformer"]
    assert ct["parameters"] in (113292800, 27974144)
    assert ct["encoder_layers"] in (6, 8)
    assert ct["decoder_layers"] in (4, 6)
    assert ct["shared_vocab_size"] in (8192, 24576)
    assert len(ct["training_curve"]) > 0
    assert ct["best_val_loss"] is not None
    assert data["evaluation_benchmark"] is not None
    assert len(data["evaluation_benchmark"]) >= 12


def test_model_specs_endpoint(api_client):
    response = api_client.get("/api/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "TranslateMe" in data["model_name"]
    assert "Multilingual Transformer" in data["model_name"]
    assert any(p in data["parameters"] for p in ("113,292,800", "27,974,144"))
    assert data["layers"] in (6, 8)
    assert data["encoder_layers"] in (6, 8)
    assert data["decoder_layers"] in (4, 6)
    assert data["checkpoint_size_mb"] > 0


def test_document_translate_endpoint(api_client):
    txt_bytes = b"Hello, how are you today?\n\nWhat is your name?"
    res = api_client.post(
        "/api/document/translate",
        files={"file": ("sample.txt", io.BytesIO(txt_bytes), "text/plain")},
        data={"source_language": "en", "target_language": "hi"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["filename"] == "sample.txt"
    assert data["total_paragraphs"] == 2
    assert data["translated_paragraphs"] == 2
    assert data["truncated"] is False
    assert data["source_language"] == "en"
    assert data["target_language"] == "hi"
    assert len(data["translated_text"]) > 0


def test_history_and_analytics_flow(api_client):
    res = api_client.get("/api/history?limit=10")
    assert res.status_code == 200
    items = res.json()
    assert isinstance(items, list)
    assert len(items) >= 1
    assert items[0]["created_at"].endswith("+00:00") or items[0]["created_at"].endswith("Z")

    item_id = items[0]["id"]
    del_res = api_client.delete(f"/api/history/{item_id}")
    assert del_res.status_code == 200

    ana_res = api_client.get("/api/analytics")
    assert ana_res.status_code == 200
    ana = ana_res.json()
    assert ana["total_translations"] >= 1
    assert ana["active_directions_count"] >= 1
    assert ana["unique_languages_count"] >= 2
    assert len(ana["top_language_pairs"]) >= 1
