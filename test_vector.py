import ollama

response = ollama.chat(
    model="llama3:8b",
    messages=[
        {"role": "user", "content": "Explain RAG in 3 simple sentences."}
    ]
)

print(response["message"]["content"])