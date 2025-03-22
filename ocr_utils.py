import cv2
import numpy as np

from ocr.read import run_easyocr

# import pytesseract
# from ocr.partition import image_partition
# from ocr.preprocess import preprocess
# from ocr.recognition import read_number_plate


def ocr_process(cropped_lp_img):

    if cropped_lp_img is None:
        raise ValueError("Error: The input image is None!")

    # # Print image properties for debugging
    # print(f"Image Type: {type(cropped_lp_img)}")
    # print(f"Image Shape: {cropped_lp_img.shape}")
    # print(f"Image Data Type: {cropped_lp_img.dtype}")
    # print(
    #     f"Min Pixel Value: {cropped_lp_img.min()}, Max Pixel Value: {cropped_lp_img.max()}"
    # )

    # Convert image to uint8
    if cropped_lp_img.dtype != np.uint8:
        if cropped_lp_img.max() <= 1.0:
            cropped_lp_img = (cropped_lp_img * 255).astype(np.uint8)
        else:
            cropped_lp_img = cv2.convertScaleAbs(cropped_lp_img)

    extracted_text = run_easyocr(cropped_lp_img)
    # print("Extracted License Plate Text:", extracted_text)
    return extracted_text

    # result = image_partition(cropped_lp_img, show_images=False)

    # # Get the color partitions
    # upper_part = result["upper_color"]
    # lower_part = result["lower_color"]
    # # preprocess(lower_part)

    # number = read_number_plate(lower_part)
