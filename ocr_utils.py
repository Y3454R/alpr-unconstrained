import cv2
import pytesseract


def ocr_process(cropped_lp_img):
    # Convert the image to grayscale for better OCR accuracy
    gray_img = cv2.cvtColor(cropped_lp_img, cv2.COLOR_BGR2GRAY)

    # Ensure the image is in uint8 format (needed for Tesseract)
    gray_img = cv2.convertScaleAbs(gray_img)

    cv2.imshow("grayscaled: ", gray_img)
    cv2.waitKey(5000)

    # # Perform OCR (with Bangla language)
    custom_oem_psm_config = r"--oem 3 --psm 6 -l ben"  # 'ben' is for Bangla language
    ocr_result = pytesseract.image_to_string(gray_img, config=custom_oem_psm_config)

    # # Display the detected text
    print("Detected Text: ", ocr_result)

    # Optionally, show the image
    cv2.imshow("Detected License Plate", cropped_lp_img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
