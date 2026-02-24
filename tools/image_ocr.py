import easyocr
import os


# Load OCR reader once (English only for now)
reader = easyocr.Reader(['en'], gpu=False)


def clean_ocr_text(text_blocks):
    """
    Cleans OCR output:
    - Removes very short fragments
    - Joins meaningful lines
    - Removes excessive whitespace
    """
    cleaned_lines = []

    for block in text_blocks:
        text = block.strip()

        # Remove very short noisy fragments
        if len(text) < 4:
            continue

        cleaned_lines.append(text)

    # Join lines into paragraph
    final_text = " ".join(cleaned_lines)

    # Remove extra spaces
    final_text = " ".join(final_text.split())

    return final_text


def extract_text_from_image(image_path):
    """
    Extract text from image using EasyOCR
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError("Image path does not exist.")

    results = reader.readtext(image_path, detail=0)

    cleaned_text = clean_ocr_text(results)

    return cleaned_text