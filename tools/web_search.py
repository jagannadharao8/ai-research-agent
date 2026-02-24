from ddgs import DDGS


def search_web(query, max_results=5):
    """
    Returns structured documents:
    [
        {
            "source": "web",
            "title": "...",
            "url": "...",
            "content": "..."
        }
    ]
    """

    results = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):

                results.append({
                    "source": "web",
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "content": r.get("body", "")
                })

    except Exception as e:
        print("⚠ Web search failed:", e)
        return []

    return results