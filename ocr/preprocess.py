import cv2
import numpy as np
import pytesseract


def preprocess(image, show=True):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ✅ Ensure grayscale is in uint8 format
    gray = np.uint8(gray)

    # Apply Gaussian Blur to reduce noise (Optional)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Create a rectangular structuring element (kernel)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))

    # Apply the Blackhat operation (to extract dark regions)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    # Apply Otsu's threshold to get binary output (Optional)
    _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    custom_config = r"--oem 3 --psm 13 -c tessedit_char_whitelist=০১২৩৪৫৬৭৮৯"
    # custom_config = r"--oem 3 --psm 13 numbers"
    character = pytesseract.image_to_string(binary, lang="ben", config=custom_config)

    print(character)

    # Show results
    cv2.imshow("Original Image", image)
    cv2.waitKey(5000)
    cv2.imshow("Grayscale", gray)
    cv2.waitKey(5000)
    cv2.imshow("Blackhat", blackhat)
    cv2.waitKey(5000)
    cv2.imshow("Binary (Thresholded)", binary)
    cv2.waitKey(5000)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()


# # ✅ Load image first!
# img_path = "dekha_jak/test_detected_lp.png"
# image = cv2.imread(img_path)  # Load image
# if image is None:
#     print("Error: Could not load image. Check the path!")
# else:
#     processed_img = preprocess(image, show=True)
