# Backend: Flask Server
from flask import Flask, request, jsonify, render_template
import numpy as np
from keras.models import load_model
import cv2
import base64
import os

app = Flask(__name__)
MODEL = load_model("mnist_model.pkl")
LABELS = {
    0: "Zero",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get base64 image data from the request
        data = request.json
        image_data = base64.b64decode(data["image"])

        # Convert image data to NumPy array
        np_img = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)

        # Preprocess image
        image = cv2.resize(img, (28, 28))
        image = np.pad(image, (10, 10), mode="constant", constant_values=0)
        image = cv2.resize(image, (28, 28)) / 255.0

        # Predict digit
        prediction = np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))
        label = LABELS[prediction]
        return jsonify({"label": label})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)