import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.rag_pipeline import prepare_rag, stream_llm, evaluate_answer
from core.database import log_query, get_analytics

query = "who won the ipl 2026"
print(f"Running query: {query}")
print("---")

prompt, sources, mode = prepare_rag(query)
print(f"Mode: {mode}")

print("\nStreaming Answer: ", end="")
stream = stream_llm(prompt)
full_answer = ""
for chunk in stream:
    print(chunk, end="", flush=True)
    full_answer += chunk
print()

print("\n--- Metrics ---")
score, risk, confidence = evaluate_answer(full_answer, sources)
print(f"Hallucination Score: {score}%")
print(f"Confidence: {confidence}%")
print(f"Risk Level: {risk}")

log_query(query, mode, score, confidence, risk)
print("\n--- Analytics ---")
analytics = get_analytics()
print(f"Total queries logged: {len(analytics)}")
