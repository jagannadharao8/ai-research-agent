# 🧠 Autonomous AI Research Agent

**Created by Jalla Jagannadharao**

An enterprise-grade, autonomous AI research assistant built with Streamlit and powered by advanced Agentic Workflows. This application goes beyond standard RAG (Retrieval-Augmented Generation) by incorporating a multi-agent architecture capable of query decomposition, self-correcting fact-checks, multimodal vision, and dynamic Python code execution.

![App Screenshot](https://raw.githubusercontent.com/jagannadharao8/ai-research-agent/main/docs/screenshot.png) *(Note: Add a screenshot of your app to a `docs/` folder to display it here)*

## 🚀 Features

### 1. Agentic Query Planner
Instead of executing a single search, the AI acts as an autonomous planner. It intercepts complex user queries, breaks them down into multiple sub-tasks, executes parallel web searches via DuckDuckGo, and synthesizes a comprehensive final report.

### 2. Self-Correcting Reflection Loops
AI hallucinations are mathematically mitigated through an autonomous reflection loop. A secondary "Fact-Checker" agent reviews the primary agent's draft against the retrieved sources. If citations are unsupported, the Fact-Checker rejects the draft and forces the main agent to rewrite the answer before it is displayed to the user.

### 3. Sandboxed Code Interpreter
Equipped with a secure Python execution environment, the agent can write and run Python scripts dynamically. Users can upload `.csv` or `.xlsx` datasets, and the agent will use Pandas and Matplotlib to analyze the data, compute statistics, and render mathematical charts directly in the chat interface.

### 4. Interactive Knowledge Graphs
Powered by `streamlit-agraph`, the system automatically extracts entities and relationships from the generated research report and renders them as a drag-and-drop interactive Knowledge Graph, allowing users to visually map complex topics.

### 5. Multimodal Vision
Supports image uploads (PNG, JPG) using Groq's Vision models (Llama 3.2 Vision), allowing users to chat with and analyze visual documents, charts, and diagrams.

### 6. Semantic Caching & Enterprise Exports
- **FAISS Vector Caching**: Identical or semantically similar queries are served instantly from a FAISS vector database, reducing API latency and saving token costs.
- **Automated Reports**: Download research results as beautifully formatted PDF, Microsoft Word (.docx), or Markdown files.

## 🛠️ Technology Stack
- **Frontend**: Streamlit, Streamlit-Agraph
- **AI Models**: Groq (Llama 3.3 70B for reasoning, Llama 3.2 Vision for Multimodal)
- **Vector Database**: FAISS (Facebook AI Similarity Search), HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)
- **Tools**: DuckDuckGo API (Web Search), PyPDF2 & Python-Docx (Document Processing), Subprocess (Python Sandbox)

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jagannadharao8/ai-research-agent.git
   cd ai-research-agent
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the Application:**
   ```bash
   streamlit run ui/app.py
   ```

## 🔒 Security Note (Code Interpreter)
The Python Code Interpreter runs in a temporary directory with a strict 15-second execution timeout to prevent infinite loops. In a production cloud environment, this sandbox should be further isolated using Docker containers or gVisor.

---
*Built with ❤️ for the future of Autonomous AI.*
