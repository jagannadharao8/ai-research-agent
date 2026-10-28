"""Offline regressions for provider failures, retrieval routing and cache isolation."""
import sys
import types
import unittest
from unittest.mock import patch
from core import llm
from core.cache import context_key, check_cache, add_to_cache, MAX_ENTRIES
from app import rag_pipeline as rag

class ReliabilityTests(unittest.TestCase):
    def test_reasoning_router_has_completion_budget(self):
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="DIRECT"))])
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
        with patch.object(llm, "get_client", return_value=client), patch.object(llm, "model_name", return_value="openai/gpt-oss-20b"):
            self.assertEqual(llm.complete("Route this question", max_tokens=10), "DIRECT")
        self.assertEqual(calls[0]["max_tokens"], 512)

    def test_model_config(self):
        with patch.dict("os.environ", {"GROQ_MODEL": "chosen-model"}):
            self.assertEqual(llm.model_name(), "chosen-model")

    def test_missing_key(self):
        with patch.object(llm, "setting", return_value=""):
            with self.assertRaises(llm.LLMError):
                llm.get_client()

    def test_provider_failure_is_sanitised(self):
        class Failure(Exception):
            status_code = 404
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
            create=lambda **kwargs: (_ for _ in ()).throw(Failure("secret-value")))))
        with patch.object(llm, "get_client", return_value=client):
            with self.assertRaises(llm.LLMError) as failure:
                llm.complete("question")
        self.assertIn("GROQ_MODEL", str(failure.exception))
        self.assertNotIn("secret-value", str(failure.exception))

    def test_empty_completion_fails(self):
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
            create=lambda **kwargs: types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=""))]))))
        with patch.object(llm, "get_client", return_value=client):
            with self.assertRaises(llm.LLMError):
                llm.complete("question")

    def test_no_evidence_is_not_maximum_confidence(self):
        self.assertEqual(rag.evaluate_answer("An answer.", []), (None, "NOT EVALUATED", None))

    def test_cache_context_and_session_isolation(self):
        base = context_key("same", "doc-a", [], "model")
        for other in [context_key("same", "doc-b", [], "model"),
                      context_key("same", "doc-a", [{"role": "user", "content": "prior"}], "model"),
                      context_key("same", "doc-a", [], "other-model")]:
            self.assertNotEqual(base, other)
        first, second = {}, {}
        add_to_cache(first, base, {"answer": "A"}, now=0)
        self.assertEqual(check_cache(first, base, now=1)["answer"], "A")
        self.assertIsNone(check_cache(second, base, now=1))
        self.assertIsNone(check_cache(first, base, now=300))

    def test_cache_size_bounded(self):
        cache = {}
        for n in range(MAX_ENTRIES + 3):
            add_to_cache(cache, str(n), n, now=n)
        self.assertEqual(len(cache), MAX_ENTRIES)
        self.assertNotIn("0", cache)

    def test_planner_rejects_wrong_json_shape(self):
        for response in ['{"bad": "shape"}', '[4]', '[]', 'nonsense']:
            with patch.object(rag, "call_llm", return_value=response):
                self.assertEqual(rag.decompose_query("original"), ["original"])

    def test_unknown_citation_is_rejected_without_provider(self):
        with patch.object(rag, "call_llm") as call:
            self.assertIn("unknown", rag.verify_citations("Claim [9]", [{"citation": 1, "content": "evidence"}]))
            call.assert_not_called()

    def test_missing_citation_is_unsupported(self):
        self.assertTrue(rag.verify_citations("Claim", [{"citation": 1, "content": "evidence"}]).startswith("Unsupported"))

    def test_reflection_rewrites_and_rechecks(self):
        with patch.object(rag, "call_llm", side_effect=["first [1]", "corrected [1]"]), patch.object(
                rag, "verify_citations", side_effect=["Partial support", "Supported by retrieved evidence (LLM review)"]):
            answer, verdict = rag.generate_with_reflection("question", [{"citation": 1}])
        self.assertEqual(answer, "corrected [1]")
        self.assertTrue(verdict.startswith("Supported"))

    def test_document_bypasses_direct_router(self):
        loader = types.ModuleType("tools.document_loader")
        loader.load_document = lambda path: [{"content": "private document", "title": "file"}]
        with patch.dict(sys.modules, {"tools.document_loader": loader}), patch.object(rag, "is_general_query") as router, patch.object(
                rag, "build_vector_store", return_value=(object(), [{"content": "private document"}])), patch.object(
                rag, "retrieve_context", return_value=("private document", [{"citation": 1}])):
            prompt, sources, mode = rag.prepare_rag("explain", "file.pdf")
        router.assert_not_called()
        self.assertEqual(mode, "Document")
        self.assertIn("private document", prompt)

    def test_code_execution_disabled(self):
        from tools.code_runner import run_python_code
        with self.assertRaises(RuntimeError):
            run_python_code("print('must not execute')")

if __name__ == "__main__":
    unittest.main()
