import os
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = load_model("../results/models/covid_cnn_model.h5")

def prepare_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (299, 299))
    img = img / 255.0
    img = img.reshape(1, 299, 299, 1)
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    uploaded_img_path = None
    confidence = None
    report_text = ""

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = file.filename
            uploaded_img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(uploaded_img_path)

            img = prepare_image(uploaded_img_path)
            result = model.predict(img)[0][0]
            prediction = "COVID" if result > 0.5 else "Normal"
            confidence = round(float(result if result > 0.5 else 1 - result), 2)

    # Load report file
    try:
        with open("../results/raports/classification_report.txt", "r") as f:
            report_text = f.read()
    except FileNotFoundError:
        report_text = "Classification report not found."

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           image_path=uploaded_img_path,
                           report_text=report_text,
                           heatmap_url="/visualizations/confusion_matrix.png",
                           accuracy_loss_url="/visualizations/accuracy_loss.png",
                           precision_recall_url="/visualizations/precision_recall.png")

@app.route('/visualizations/<filename>')
def visualizations(filename):
    return send_from_directory("../results/visualisation", filename)

if __name__ == "__main__":
    app.run(debug=True)
