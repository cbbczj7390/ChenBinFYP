from flask import Flask, request, render_template, jsonify
import os
from features import extract_features
from database import save_image_features, query_database, init_db
from object_detection import detect_objects

app = Flask(__name__)

# Directory to save uploaded images
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Detect objects in the uploaded image using YOLOv5
        detected_objects = detect_objects(filepath, conf_threshold=0.5)

        # Extract features for similarity comparison
        features = extract_features(filepath)
        save_image_features(filepath, features)

        # Find similar images in the database
        similar_images = query_database(features, uploaded_path=filepath)
        # Render the template with detected objects and similar images
        return render_template(
            'similar_images.html',
            uploaded_image=filepath,
            similar_images=similar_images,
            detection_results=detected_objects
        )

if __name__ == '__main__':
    init_db()
    app.run(debug=True)