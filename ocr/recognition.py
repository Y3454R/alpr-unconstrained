import pytesseract
import cv2
import numpy as np


def preprocess_image(image):

    if image.dtype != np.uint8:
        image = cv2.convertScaleAbs(image)

    # Convert to grayscale if not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = get_grayscale(image)
    # contrast_img = increase_contrast(gray)
    # threshold_img = adaptive_threshold(contrast_img)
    # smoothed_img = smooth_image(threshold_img)
    # final_img = dilate_then_erode(smoothed_img, kernel_size=2)

    thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)[1]
    # kernel = np.ones((2,2),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    final_img = 255 - opening

    # # erosion = cv2.erode(final_img, kernel, iterations = 1)
    # # final_img = erosion

    final_img = cv2.resize(final_img, (0, 0), fx=1.0, fy=0.70)

    return final_img


def read_single_bangla_character(image):
    if image is None:
        raise ValueError("Error: Image is None!")

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Tesseract configuration for single character recognition
    custom_config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=০১২৩৪৫৬৭৮৯"
    # custom_config = r"--oem 3 --psm 13 numbers"
    character = pytesseract.image_to_string(
        processed_image, lang="ben", config=custom_config
    )

    return character.strip()


def remove_hyphen(text):
    text = text.replace("-", "")
    return text


def read_number_plate(image):
    combined_text = ""
    character = read_single_bangla_character(image)
    if len(character) > 6:
        character = character[:2] + character[3:]
    combined_text += character
    print(f"text: {combined_text}")
    return combined_text
