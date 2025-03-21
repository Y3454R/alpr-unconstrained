import easyocr


def run_easyocr(image):
    """
    Perform OCR using EasyOCR on the given image.

    :param image: Input image (NumPy array) containing the license plate.
    :return: Extracted text as a string.
    """
    # Initialize EasyOCR with Bangla language (disable GPU for CPU execution)
    reader = easyocr.Reader(["bn"], gpu=False)

    # Run OCR on the image
    result = reader.readtext(image)

    # Extract text from OCR results
    extracted_text = " ".join([res[1] for res in result])

    return extracted_text
