"""Session-private research interface; unsupported execution modes stay hidden."""
import hashlib
import os
import sys
import tempfile
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.rag_pipeline import prepare_rag, generate_with_reflection
from core.llm import LLMError, model_name
from core.session_cache import context_key, check_cache, add_to_cache
from export.doc_report import generate_docx, generate_markdown

st.set_page_config(page_title="AI Research Assistant", page_icon="🧠", layout="wide")
st.title("🧠 AI Research Assistant")
st.caption("Built by Jalla Jagannadharao · Python, retrieval and citation review")
st.info("Answers and citation reviews can be wrong. Verify important claims against the linked sources.")
for name, default in [("messages", []), ("answer_cache", {})]:
    if name not in st.session_state:
        st.session_state[name] = default
if st.button("Clear this session"):
    st.session_state.messages = []
    st.session_state.answer_cache = {}
    st.session_state.pop("last_report", None)
    st.rerun()
uploaded = st.file_uploader("Optional PDF or Word document (maximum 10 MB)", type=["pdf", "docx"])
show_graph = st.checkbox("Generate an experimental knowledge graph", value=False)
st.caption("Document questions use the uploaded file. Without a file, the assistant chooses direct answering or web search.")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question"):
    history = list(st.session_state.messages[-5:])
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        document_path = None
        try:
            data = uploaded.getvalue() if uploaded else b""
            if len(data) > 10 * 1024 * 1024:
                raise ValueError("The document is too large. Maximum size is 10 MB.")
            digest = hashlib.sha256(data).hexdigest() if uploaded else ""
            key = context_key(query, digest, history, model_name())
            result = check_cache(st.session_state.answer_cache, key)
            if result:
                st.caption("Reused a recent answer from this session.")
            else:
                if uploaded:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as handle:
                        handle.write(data)
                        document_path = handle.name
                with st.spinner("Retrieving evidence and reviewing the answer..."):
                    prompt, sources, mode = prepare_rag(query, document_path, history)
                    if uploaded:
                        for source in sources:
                            source['title'] = uploaded.name
                    answer, verification = generate_with_reflection(prompt, sources)
                result = {"answer": answer, "sources": sources, "mode": mode, "verification": verification}
                # Failed provider calls raise and never reach cache/export.
                add_to_cache(st.session_state.answer_cache, key, result)
            st.markdown(result["answer"])
            st.caption("Mode: " + result["mode"])
            if result["sources"]:
                st.caption("Retrieval: " + result["sources"][0].get("retrieval", "not recorded"))
            st.caption("Citation review: " + result["verification"])
            st.caption("Citation review is an LLM heuristic, not a confidence percentage or guarantee of truth.")
            if result["sources"]:
                with st.expander("Retrieved sources"):
                    for source in result["sources"]:
                        st.write(f"[{source['citation']}] {source.get('title', 'Document')}")
                        url = source.get("url", "")
                        if url.startswith(("https://", "http://")):
                            st.link_button("Open source", url)
                        st.text(source["content"])
            if show_graph:
                try:
                    from core.graph_builder import build_knowledge_graph, render_graph
                    nodes, edges = build_knowledge_graph(result["answer"])
                    render_graph(nodes, edges)
                except Exception:
                    st.warning("The optional graph could not be generated. The answer is still available.")
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
            st.session_state.last_report = (query, result)
        except (LLMError, ValueError) as exc:
            st.error(str(exc))
        except Exception:
            st.error("The request could not be processed. Please retry or use another document.")
        finally:
            if document_path:
                Path(document_path).unlink(missing_ok=True)

# Keep download controls available across Streamlit reruns.
if "last_report" in st.session_state:
    question, result = st.session_state.last_report
    st.subheader("Download the last completed report")
    st.download_button("Markdown", generate_markdown(question, result["answer"], result["sources"]), "research.md")
    try:
        st.download_button("Word", generate_docx(question, result["answer"], result["sources"]), "research.docx")
        from export.pdf_report import generate_pdf_report
        with tempfile.TemporaryDirectory() as folder:
            destination = os.path.join(folder, "research.pdf")
            generate_pdf_report(destination, question, result["answer"], None, None, "NOT EVALUATED", result["sources"])
            st.download_button("PDF", Path(destination).read_bytes(), "research.pdf", mime="application/pdf")
    except Exception:
        st.caption("Some export formats are unavailable; Markdown remains available.")
