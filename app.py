from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
from features import extract_features
from database import save_image_features, query_database, init_db
from object_detection import detect_objects

app = Flask(__name__)

# Directory to save uploaded images
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def remove_duplicates(detections):
    seen_names = set()
    unique_detections = []
    for obj in detections:
        if obj['name'] not in seen_names:
            seen_names.add(obj['name'])
            unique_detections.append(obj)
    return unique_detections

@app.route('/')
def index():
    return render_template('index.html')

# Single Image Upload Route
@app.route('/upload_single', methods=['POST'])
def upload_single_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    model_type = request.form['model']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        detected_objects = detect_objects(filepath, model_type=model_type, conf_threshold=0.5)

        if isinstance(detected_objects, pd.DataFrame):
            detected_objects = detected_objects.to_dict(orient='records')

        detected_objects = remove_duplicates(detected_objects)

        features = extract_features(filepath)
        save_image_features(filepath, features)

        similar_images = query_database(features, uploaded_path=filepath)

        return render_template('batch_results.html', results=[{
            'uploaded_image': filepath,
            'detected_objects': detected_objects,
            'similar_images': similar_images
        }])

# Multiple Images Upload Route
@app.route('/upload_multiple', methods=['POST'])
def upload_multiple_images():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'})

    files = request.files.getlist('files')
    model_type = request.form['model']

    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No selected files'})

    results = []

    for file in files:
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            detected_objects = detect_objects(filepath, model_type=model_type, conf_threshold=0.5)

            if isinstance(detected_objects, pd.DataFrame):
                detected_objects = detected_objects.to_dict(orient='records')

            detected_objects = remove_duplicates(detected_objects)

            features = extract_features(filepath)
            save_image_features(filepath, features)

            similar_images = query_database(features, uploaded_path=filepath)

            results.append({
                'uploaded_image': filepath,
                'detected_objects': detected_objects,
                'similar_images': similar_images
            })

    return render_template('batch_results.html', results=results)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)