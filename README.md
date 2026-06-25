# 🧠 Autonomous AI Research Agent (V3 Enterprise Edition)

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Groq](https://img.shields.io/badge/Groq-000000?style=for-the-badge&logo=groq&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

The **Autonomous AI Research Agent** is a next-generation research assistant powered by Groq's high-speed inference and Llama-3. It dynamically routes queries, searches the web or uploaded PDFs, and provides fully cited, highly accurate responses with real-time hallucination checking.

---

## 🌟 V3 Enterprise Features

- **🧠 Agentic Query Planner**: Automatically decomposes massive, complex questions into multiple sub-queries for parallel web search.
- **🕸️ Interactive Knowledge Graph**: Extracts Entities and Relationships from the generated text and renders an interactive drag-and-drop web using `streamlit-agraph`.
- **🕵️ Double-Pass Citation Verification**: A second LLM pass strictly fact-checks the final answer to guarantee zero hallucinated citations.
- **💬 Multi-Turn Chat Memory**: Remembers your previous questions in the session for seamless conversational follow-ups.
- **⚡ Semantic Query Caching**: Encodes past questions in FAISS. If you ask a similar question, it serves the answer instantly from cache, saving API costs and time.
- **📄 Multi-Format Export**: Download reports natively in **PDF, Microsoft Word (.docx), and Markdown**.
- **🐳 Docker Ready**: Full containerization support with `docker-compose`.

---

## 📁 Project Architecture

```text
AI-Research-Agent/
├── app/
│   └── rag_pipeline.py          # Core RAG, LLM Planner, Cache integration
├── core/
│   ├── cache.py                 # Semantic FAISS caching layer
│   ├── database.py              # SQLite analytics and query logging
│   ├── embedding_model.py       # SentenceTransformers singleton
│   ├── graph_builder.py         # Extracts Nodes/Edges for Knowledge Graph
│   └── logger.py                # Structured Python logging
├── evaluation/
│   └── hallucination_checker.py # Vector-based hallucination scoring
├── export/
│   ├── doc_report.py            # .docx and .md generators
│   └── pdf_report.py            # PDF report generator (ReportLab)
├── tests/
│   └── test_rag.py              # Pytest automated testing suite
├── tools/
│   ├── pdf_loader.py            # PDF parsing and chunking
│   └── web_search.py            # DuckDuckGo integration
├── ui/
│   └── app.py                   # Streamlit Frontend (Chat interface)
├── Dockerfile                   
├── docker-compose.yml           
├── .env.example                 
├── requirements.txt             
└── README.md                    
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+ or Docker
- A [Groq API Key](https://console.groq.com/keys)

### 2. Standard Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/jagannadharao8/ai-research-agent.git
cd ai-research-agent
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

Create a `.env` file from the example and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

Launch the Streamlit app:
```bash
streamlit run ui/app.py
```

### 3. Docker Installation
You can instantly run the entire project in a container:
```bash
docker-compose up --build
```

---

## 🛠️ Built With

- **[Streamlit](https://streamlit.io/)**: Frontend UI and Session State
- **[Groq](https://groq.com/)**: Fast LLM Inference (Llama 3)
- **[SentenceTransformers](https://www.sbert.net/)**: Local Embedding Generation
- **[FAISS](https://github.com/facebookresearch/faiss)**: Fast Vector Similarity Search
- **[streamlit-agraph](https://github.com/Chris7/streamlit-agraph)**: Knowledge Graph UI
- **[python-docx](https://python-docx.readthedocs.io/)**: Word Document Export

---

## 📄 License
This project is for educational and research purposes.
