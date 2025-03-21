import cv2
import numpy as np


def image_partition(image, show_images=True):
    """Splits a BGR license plate image into text and number plate regions dynamically."""

    if image is None or len(image.shape) != 3:
        raise ValueError(
            "Invalid image! Ensure the image is correctly loaded and in BGR format."
        )

    # Convert to grayscale for edge detection (not for output)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Detect horizontal lines
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
    )

    # Determine the split position
    if lines is not None and len(lines) > 0:
        y_positions = [line[0][1] for line in lines]
        split_y = int(np.median(y_positions))
    else:
        split_y = image.shape[0] // 2  # Default to midpoint

    # Ensure valid split position
    split_y = max(1, min(split_y, image.shape[0] - 1))

    # Partition the original color image
    upper_color = image[:split_y, :]
    lower_color = image[split_y:, :]

    if show_images:
        cv2.imshow("Upper Color", upper_color)
        cv2.waitKey(5000)
        cv2.imshow("Lower Color", lower_color)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

    return {
        "upper_color": upper_color,
        "lower_color": lower_color,
    }
