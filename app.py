from flask import Flask, request, jsonify
from pypeline import vehicle_detection
import os

app = Flask(__name__)


@app.route("/", methods=["GET"])
def read_root():
    return jsonify({"message": "Hello, Flask!"})


@app.route("/alpr/detect", methods=["POST"])
def lp_detection_route():
    if "image" not in request.files:
        return jsonify({"error": "Image file is required"}), 400

    image_file = request.files["image"]

    # Save the image (optional)
    save_path = os.path.join("uploads", image_file.filename)
    os.makedirs("uploads", exist_ok=True)
    image_file.save(save_path)

    # return jsonify({"message": "got your pic, bro!"})

    results = vehicle_detection(save_path)

    return jsonify(
        {
            "message": "Detection complete",
            "vehicle_detection": results,
        }
    )


def vehicle_detection(image_path):
    # Dummy function to simulate processing
    return {"image_path": image_path, "vehicles_detected": 2}


if __name__ == "__main__":
    app.run(debug=True)
