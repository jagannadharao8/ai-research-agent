# 🧠 Autonomous AI Research Agent

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Groq](https://img.shields.io/badge/Groq-000000?style=for-the-badge&logo=groq&logoColor=white)

The **Autonomous AI Research Agent** is a next-generation research assistant powered by Groq's high-speed inference and Llama-3. It dynamically routes queries, searches the web or uploaded PDFs, and provides fully cited, highly accurate responses with real-time hallucination checking.

---

## 🌟 Key Features

- **⚡ Blazing Fast Inference**: Powered by Groq's Llama-3.3-70B model.
- **🔍 Intelligent Routing**: Automatically decides between direct answering and real-time Web Search / RAG.
- **🌐 Live Web Search**: Fetches the latest information from the web via DuckDuckGo.
- **📄 PDF Context Support**: Upload your own PDFs to provide custom knowledge bases.
- **🛡️ Real-Time Hallucination Checking**: Every sentence is mathematically evaluated against the source context using FAISS vector similarity.
- **📊 Live Analytics**: Tracks query history, confidence metrics, and risk levels in real-time.
- **📥 PDF Export**: Download beautiful, structured PDF reports of your research.

---

## 📁 Project Architecture

```text
AI-Research-Agent/
├── app/
│   ├── __init__.py
│   └── rag_pipeline.py          # Core RAG, LLM calls, and query routing
├── core/
│   ├── __init__.py
│   ├── database.py              # SQLite analytics and query logging
│   └── embedding_model.py       # SentenceTransformers embedding lazy-loader
├── evaluation/
│   ├── __init__.py
│   └── hallucination_checker.py # Vector-based hallucination scoring
├── export/
│   ├── __init__.py
│   └── pdf_report.py            # PDF report generation (ReportLab)
├── tools/
│   ├── __init__.py
│   ├── pdf_loader.py            # PDF parsing and semantic chunking
│   └── web_search.py            # DuckDuckGo integration with retry logic
├── ui/
│   ├── __init__.py
│   └── app.py                   # Streamlit Frontend
├── .env                         # Environment variables (API keys)
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- A [Groq API Key](https://console.groq.com/keys)

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/jagannadharao8/ai-research-agent.git
cd ai-research-agent
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the Application

Launch the Streamlit app:
```bash
streamlit run ui/app.py
```

---

## 🛠️ Built With

- **[Streamlit](https://streamlit.io/)**: Frontend UI and Session State
- **[Groq](https://groq.com/)**: Fast LLM Inference (Llama 3)
- **[SentenceTransformers](https://www.sbert.net/)**: Local Embedding Generation
- **[FAISS](https://github.com/facebookresearch/faiss)**: Fast Vector Similarity Search
- **[DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/)**: Live Web Search
- **[ReportLab](https://www.reportlab.com/)**: PDF Export

---

## 📄 License
This project is for educational and research purposes.
