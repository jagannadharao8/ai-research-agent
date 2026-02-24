from tools.image_ocr import extract_text_from_image

image_path = input("Enter image path: ")

text = extract_text_from_image(image_path)

print("\nExtracted Text:\n")
print(text)