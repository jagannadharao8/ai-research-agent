from tools.pdf_loader import load_pdf_as_documents

pdf_path = input("Enter PDF path: ")

docs = load_pdf_as_documents(pdf_path)

print("\nNumber of chunks extracted:", len(docs))
print("\nSample chunk:\n")
print(docs[0]["content"])