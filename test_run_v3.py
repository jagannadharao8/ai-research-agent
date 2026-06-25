import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.rag_pipeline import prepare_rag, call_llm, evaluate_answer, verify_citations
from core.graph_builder import build_knowledge_graph

def run_test():
    query = "Compare the GDP growth and inflation of India and China in 2023."
    print(f"QUERY: {query}")
    print("-" * 50)
    
    print("1. Preparing RAG Context (Includes Decomposition & Search)...")
    prompt, sources, mode = prepare_rag(query)
    
    print(f"MODE: {mode}")
    print(f"FOUND {len(sources)} SOURCES")
    print("-" * 50)
    
    print("2. Calling LLM (Generation)...")
    answer = call_llm(prompt, max_tokens=1000)
    print(f"ANSWER:\n{answer}\n")
    print("-" * 50)
    
    print("3. Evaluating & Verifying...")
    score, risk, conf = evaluate_answer(answer, sources)
    verification = verify_citations(answer, sources)
    print(f"Score: {score} | Risk: {risk} | Confidence: {conf}")
    print(f"Verification: {verification}")
    print("-" * 50)
    
    print("4. Building Knowledge Graph...")
    nodes, edges = build_knowledge_graph(answer)
    print(f"Extracted {len(nodes)} nodes and {len(edges)} edges.")
    if nodes:
        print(f"Sample Node: {nodes[0].label}")
    if edges:
        print(f"Sample Edge: {edges[0].source} -> {edges[0].target} ({edges[0].label})")
        
if __name__ == "__main__":
    run_test()
