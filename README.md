# AI Research Assistant
Python application by Jalla Jagannadharao for document questions and web-assisted research.

## Implemented path
- Optional PDF/DOCX extraction, overlapping chunks, Sentence Transformers embeddings and FAISS retrieval.
- Without a document: LLM routing to direct answering or up to three sequential web searches.
- Draft generation, citation identifier checks, and one best-effort LLM-guided rewrite.
- Session-private exact-match cache including document digest, recent conversation and model, with a five-minute expiry and twenty-entry limit.
- Markdown, Word and PDF downloads; optional experimental knowledge graph.

## Limits
Search uses snippets rather than full-page verification. Citation review is a model judgment, not proof of truth. No calibrated confidence or benchmark accuracy is claimed. Files without extractable text require OCR elsewhere. Image, CSV/Excel and arbitrary code execution are not exposed in the UI. The legacy subprocess helper is disabled: a temporary directory and timeout do not form a security sandbox.

## Setup
Use Python 3.10+ and a virtual environment:
```sh
pip install -r requirements.txt
streamlit run ui/app.py
```
Set GROQ_API_KEY privately as an environment variable or in Streamlit secrets. The application reads those sources directly; it does not automatically load a .env file. Set GROQ_MODEL to an identifier available to that account. The default is openai/gpt-oss-20b; account access must be verified separately.
See https://console.groq.com/docs/models for model availability.

## Deployment verification
1. Configure the existing Streamlit deployment to use ui/app.py.
2. Set secrets without committing them to Git.
3. Ask a general question, then a web question.
4. Upload two documents with different facts; confirm each answer uses its own sources.
5. Check invalid credentials/model access gives an actionable error with no cached answer or new report.
6. Download each report format and inspect it.
The full offline suite passed 23 tests on 28 October 2026. A separate [real-provider check](https://github.com/jagannadharao8/ai-research-agent/actions/runs/40712070700) completed three actual routing-and-answer queries using the existing private repository credential. Offline tests mock boundaries; a provider test does not establish that every hosted integration works.

GPT-OSS calls use low reasoning effort, a reserved completion allowance and one bounded retry when the provider reports token exhaustion. Internal reasoning is never used as the visible answer. After a deployment, reboot the hosted process if it mixes older imported modules with the updated UI.

## Tests
```sh
python -m unittest discover -s tests -p "test_reliability.py" -v
python -m pytest tests/
```
The first suite is offline and requires only Python's standard library. The broader suite needs the application dependencies and an embedding-model download.

## Privacy
Answers and uploads are not written to the legacy shared cache or query database by the UI. Uploaded temporary files are removed after processing. Browser session data lasts until cleared or disconnected. Historical data files from previous deployments require separate owner cleanup. Clear this session removes current chat, cache and report.

MIT licensed. This repository is an educational portfolio application.
