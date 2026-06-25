import json
from streamlit_agraph import Node, Edge, Config, agraph
from app.rag_pipeline import call_llm

def build_knowledge_graph(answer):
    """
    Uses the LLM to extract entities and relationships from the text to build a graph.
    """
    prompt = f"""
    Extract a knowledge graph from the following text.
    Return ONLY a valid JSON object with 'nodes' (list of strings) and 'edges' (list of dicts with 'source', 'target', 'label').
    Keep nodes very concise (1-2 words). Max 10 nodes, 15 edges.
    
    Text:
    {answer}
    """
    
    response = call_llm(prompt, max_tokens=1000)
    
    try:
        # Extract JSON from potential markdown wrapping
        json_str = response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]
            
        data = json.loads(json_str.strip())
        
        nodes = []
        edges = []
        
        for n in data.get('nodes', []):
            nodes.append(Node(id=n, label=n, size=25, color="#1E88E5"))
            
        for e in data.get('edges', []):
            edges.append(Edge(source=e['source'], target=e['target'], label=e['label']))
            
        return nodes, edges
        
    except Exception as e:
        print(f"Error parsing graph JSON: {e}")
        return [], []

def render_graph(nodes, edges):
    if not nodes or not edges:
        return
        
    config = Config(
        width=700,
        height=500,
        directed=True,
        physics=True,
        hierarchical=False,
    )
    
    agraph(nodes=nodes, edges=edges, config=config)
