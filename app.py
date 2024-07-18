import os
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

# Load the pretrained Inception V3 model
model = tf.keras.applications.InceptionV3(weights="imagenet")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
app.config["MAX_CONTENT_LENGTH"] = 1000 * 1024 * 1024  # 1000 MB limit

# Ensure the upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video = request.files["video"]
    if video.filename == "":
        return jsonify({"error": "No video file selected"}), 400

    video_path = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
    video.save(video_path)

    print(f"Video saved to: {video_path}")

    # Return the video path for the form to use
    return video_path


def extract_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frames = []
    count = 0

    while success:
        frame_filename = f"frame_{count}.jpg"
        frame_path = os.path.join(app.config["UPLOAD_FOLDER"], frame_filename)
        cv2.imwrite(frame_path, image)
        frames.append(frame_filename)  # Store only the filename
        success, image = vidcap.read()
        count += 1

    print(f"Extracted {len(frames)} frames from video.")

    return frames


def detect_objects(frames):
    detected_objects = {}

    for frame in frames:
        frame_path = os.path.join(app.config["UPLOAD_FOLDER"], frame)
        img = tf.keras.preprocessing.image.load_img(frame_path, target_size=(299, 299))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)

        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.inception_v3.decode_predictions(
            predictions, top=1
        )[0]

        detected_objects[frame] = decoded_predictions[0][1]  # Store the object name

    print(f"Detected objects in frames: {detected_objects}")

    return detected_objects


@app.route("/search", methods=["POST"])
def search_object():
    search_query = request.form["query"]
    video_path = request.form["video_path"]

    print(f"Searching for '{search_query}' in video '{video_path}'")

    frames = extract_frames(video_path)
    detected_objects = detect_objects(frames)

    matching_frames = [
        os.path.join(app.config["UPLOAD_FOLDER"], frame)
        for frame, obj in detected_objects.items()
        if search_query.lower() in obj.lower()
    ]

    print(f"Matching frames: {matching_frames}")

    if matching_frames:
        return jsonify({"matching_frames": matching_frames}), 200
    else:
        return jsonify({"error": "Object doesn't exist!!!"}), 404


if __name__ == "__main__":
    app.run(debug=True, port=5000)
