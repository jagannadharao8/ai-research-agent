import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.web_search import search_web

results = search_web("What is Retrieval Augmented Generation in AI?")

for i, r in enumerate(results, 1):
    print(f"\nResult {i}:\n{r}")