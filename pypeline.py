import cv2
import numpy as np
import traceback
import darknet.python.darknet as dn
from darknet.python.darknet import detect
import sys, os
import traceback
from src.keras_utils import load_model
from glob import glob
from os.path import splitext, basename
from src.utils import im2single
from src.keras_utils import load_model, detect_lp
from src.label import Label, Shape, writeShapes
from ocr_utils import ocr_process

wpod_net_path = "data/lp-detector/wpod-net_update1.h5"
wpod_net = load_model(wpod_net_path)


def adjust_pts(pts, lroi):
    return pts * lroi.wh().reshape((2, 1)) + lroi.tl().reshape((2, 1))


def lp_detection(Ivehicle, lp_threshold=0.5):
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * 288.0)
    bound_dim = min(side + (side % (2**4)), 608)
    Llp, LlpImgs, _ = detect_lp(
        wpod_net, im2single(Ivehicle), bound_dim, 2**4, (240, 80), lp_threshold
    )
    for lp, lp_img in zip(Llp, LlpImgs):
        ocr_process(lp_img)


# def check_lp_bbox(Ivehicle, lp, lp_img):
#     # Check if we have exactly 2 points (top-left and bottom-right)
#     print(f"len(lp.pts): {len(lp.pts)}")
#     if len(lp.pts) == 2:
#         top_left = (
#             int(lp.pts[0][0] * Ivehicle.shape[1]),
#             int(lp.pts[0][1] * Ivehicle.shape[0]),
#         )
#         bottom_right = (
#             int(lp.pts[1][0] * Ivehicle.shape[1]),
#             int(lp.pts[1][1] * Ivehicle.shape[0]),
#         )
#     else:
#         print(f"Warning: Unexpected number of points ({len(lp.pts)})")
#         return

#     # Draw rectangle on the original image
#     cv2.rectangle(Ivehicle, top_left, bottom_right, (0, 255, 0), 2)  # not fitting

#     # Display the image
#     cv2.imshow("Detected License Plate", lp_img)
#     cv2.waitKey(5000)
#     cv2.destroyAllWindows()


def load_yolo_model(cfg_path, weights_path, data_path):
    net = dn.load_net(cfg_path.encode("utf-8"), weights_path.encode("utf-8"), 0)
    meta = dn.load_meta(data_path.encode("utf-8"))
    return net, meta


def detect_vehicles(net, meta, img_path, threshold=0.5):
    detections, _ = detect(net, meta, img_path.encode("utf-8"), thresh=threshold)
    return [r for r in detections if r[0].decode("utf-8") in ["car", "bus"]]


def crop_vehicle(Iorig, x1, y1, x2, y2):
    return Iorig[y1:y2, x1:x2]


def process_image(img_path, detections):
    """
    Processes an image by drawing bounding boxes around detected vehicles.
    Instead of saving the result, returns the processed image as a NumPy array.
    """
    if not detections:
        return None, []

    Iorig = cv2.imread(img_path)
    WH = np.array(Iorig.shape[1::-1], dtype=float)
    # bname = basename(splitext(img_path)[0])
    Lcars = []

    for r in detections:
        # print(f"r: {r}")
        cx, cy, w, h = (np.array(r[2]) / np.concatenate((WH, WH))).tolist()
        x1, y1 = int((cx - w / 2) * WH[0]), int((cy - h / 2) * WH[1])
        x2, y2 = int((cx + w / 2) * WH[0]), int((cy + h / 2) * WH[1])

        # Draw bounding box
        cv2.rectangle(Iorig, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add label text
        text = f"{r[0].decode()} {r[1]:.2f}"
        cv2.putText(
            Iorig, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

        # Store label info
        Lcars.append(Label(0, np.array([x1, y1]), np.array([x2, y2])))
        cropped_vehicle = crop_vehicle(Iorig, x1, y1, x2, y2)
        if cropped_vehicle is not None and cropped_vehicle.size > 0:
            lp_detection(cropped_vehicle)

        # Save or pass the cropped vehicle image
        cropped_path = f"cropped_vehicle_{x1}_{y1}.png"
        cv2.imwrite(cropped_path, cropped_vehicle)

    return Iorig, Lcars  # Return processed image and bounding boxes


vehicle_net, vehicle_meta = load_yolo_model(
    "data/vehicle-detector/yolo-voc.cfg",
    "data/vehicle-detector/yolo-voc.weights",
    "data/vehicle-detector/voc.data",
)


def vehicle_detection(img_path):
    """Runs vehicle detection on all images in input_dir and returns processed images + labels."""
    try:

        results = []

        detections = detect_vehicles(vehicle_net, vehicle_meta, img_path, threshold=0.5)
        processed_img, labels = process_image(img_path, detections)
        if processed_img is not None:
            results.append({"image": processed_img, "labels": labels})

        # print("results: ", results)

        return results

    except Exception as e:
        print(f"Error during vehicle detection: {str(e)}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = vehicle_detection(
        "/Users/adibasubah/alpr-unconstrained/test_ball/test.jpg"
    )
