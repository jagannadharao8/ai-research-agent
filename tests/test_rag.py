import pytest
from unittest.mock import patch, MagicMock

from core.embedding_model import get_embed_model
from tools.web_search import search_web
from evaluation.hallucination_checker import hallucination_check
from app.rag_pipeline import classify_risk, calculate_confidence

def test_get_embed_model():
    model = get_embed_model()
    assert model is not None
    assert hasattr(model, 'encode')

@patch('tools.web_search.DDGS')
def test_web_search(mock_ddgs):
    mock_instance = MagicMock()
    mock_instance.text.return_value = [{'title': 'Test', 'href': 'http://test.com', 'body': 'Test body'}]
    mock_ddgs.return_value.__enter__.return_value = mock_instance
    
    results = search_web("test query")
    assert len(results) > 0
    assert results[0]['title'] == 'Test'
    assert results[0]['source'] == 'web'

def test_classify_risk():
    assert classify_risk(10) == "LOW"
    assert classify_risk(30) == "MEDIUM"
    assert classify_risk(50) == "HIGH"

def test_calculate_confidence():
    assert calculate_confidence(10) == 90.0
    assert calculate_confidence(100) == 0.0
