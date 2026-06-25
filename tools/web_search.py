from ddgs import DDGS
import time

def search_web(query, max_results=5, max_retries=3):
    """
    Returns structured documents from DuckDuckGo search with retry logic.
    """
    results = []

    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "source": "web",
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "content": r.get("body", "")
                    })
            return results

        except Exception as e:
            print(f"⚠ Web search failed (Attempt {attempt+1}/{max_retries}):", e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt) # Exponential backoff
            else:
                return []

    return results