"""Streamlit UI smoke tests with provider boundaries mocked."""
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from core.llm import LLMError

class UiTests(unittest.TestCase):
    def test_successful_question_has_answer_and_export(self):
        with patch("app.rag_pipeline.prepare_rag", return_value=("prompt", [], "Direct")), patch(
            "app.rag_pipeline.generate_with_reflection", return_value=("Precision measures positive prediction correctness.", "Not evaluated: no retrieved evidence")):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "ui" / "app.py", default_timeout=30).run()
            app.chat_input[0].set_value("Explain precision").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["messages"][-1]["role"], "assistant")
        self.assertIn("Precision", app.session_state["messages"][-1]["content"])
        self.assertIn("last_report", app.session_state)

    def test_provider_error_is_not_cached_or_exported(self):
        with patch("app.rag_pipeline.prepare_rag", return_value=("prompt", [], "Direct")), patch(
            "app.rag_pipeline.generate_with_reflection", side_effect=LLMError("Provider unavailable")):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "ui" / "app.py", default_timeout=30).run()
            app.chat_input[0].set_value("Explain recall").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 1)
        self.assertEqual(app.session_state["answer_cache"], {})
        self.assertNotIn("last_report", app.session_state)

if __name__ == "__main__":
    unittest.main()
