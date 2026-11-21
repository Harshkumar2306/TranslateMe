"""
SQLAlchemy Translation History Model
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, DateTime
from backend.app.database.session import Base


class TranslationHistory(Base):
    __tablename__ = "translation_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    source_text = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=False)
    source_language = Column(String(50), nullable=False)
    source_language_name = Column(String(100), nullable=True)
    target_language = Column(String(50), nullable=False)
    target_language_name = Column(String(100), nullable=True)
    detected_language = Column(String(50), nullable=True)
    inference_time = Column(Float, default=0.0)
    model_name = Column(String(100), default="NLLB-200")
    created_at = Column(DateTime, default=datetime.utcnow)
