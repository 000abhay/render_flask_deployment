from flask import Flask, render_template, request
from model import predict_pneumonia  # Import function from model.py
from PIL import Image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"  # Create a folder for uploaded images
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)  # Save uploaded file
            result = predict_pneumonia(file_path)  # Predict pneumonia

    return render_template("interface.html", result=result)

if __name__ == "__main__":
    app.run()