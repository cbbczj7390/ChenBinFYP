from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
from werkzeug.utils import secure_filename
from features import extract_features
from database import (
    save_image_features,
    query_database,
    init_db,
    search_by_keyword,
    update_tags_for_existing_images,
    update_tags_for_images_with_empty_tags
)
from object_detection import detect_objects

app = Flask(__name__)

# Directory to save uploaded images
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed image file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """
    Checks if the uploaded file has a valid image extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def remove_duplicates(detections):
    """
    Removes duplicate detections based on object names.
    """
    seen_names = set()
    unique_detections = []
    for obj in detections:
        if obj['name'] not in seen_names:
            seen_names.add(obj['name'])
            unique_detections.append(obj)
    return unique_detections

@app.route('/')
def index():
    """
    Renders the homepage with the upload forms and search bar.
    """
    return render_template('index.html')

@app.route('/upload_single', methods=['POST'])
def upload_single_image():
    """
    Handles single image uploads, extracts features, generates tags,
    saves to the database, and returns similar images.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    model_type = request.form['model']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Only images are allowed.'})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)

    # Detect objects and generate tags
    detected_objects = detect_objects(filepath, model_type=model_type, conf_threshold=0.5)
    detected_objects = remove_duplicates(detected_objects)
    tags = [obj['name'] for obj in detected_objects]

    # Extract features and save to database with tags
    features = extract_features(filepath)
    save_image_features(filepath, features, tags)

    similar_images = query_database(features, uploaded_path=filepath)

    return render_template('batch_results.html', results=[{
        'uploaded_image': filepath,
        'detected_objects': detected_objects,
        'tags': tags,
        'similar_images': similar_images
    }])

@app.route('/upload_multiple', methods=['POST'])
def upload_multiple_images():
    """
    Handles multiple image uploads, extracts features, generates tags,
    saves to the database, and returns similar images.
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'})

    files = request.files.getlist('files')
    model_type = request.form['model']

    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No selected files'})

    results = []

    for file in files:
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filepath)

            # Detect objects and generate tags
            detected_objects = detect_objects(filepath, model_type=model_type, conf_threshold=0.5)
            detected_objects = remove_duplicates(detected_objects)
            tags = [obj['name'] for obj in detected_objects]

            # Extract features and save to database with tags
            features = extract_features(filepath)
            save_image_features(filepath, features, tags)

            similar_images = query_database(features, uploaded_path=filepath)

            results.append({
                'uploaded_image': filepath,
                'detected_objects': detected_objects,
                'tags': tags,
                'similar_images': similar_images
            })
        else:
            return jsonify({'error': 'One or more files have an invalid format. Only images are allowed.'})

    return render_template('batch_results.html', results=results)

@app.route('/search', methods=['GET'])
def search():
    """
    Handles keyword-based search for images by querying the database for tags.
    """
    query = request.args.get('query', '').strip()
    if not query:
        return jsonify({'error': 'No search query provided'})

    results = search_by_keyword(query)
    return render_template('search_results.html', results=results, query=query)

@app.route('/update_tags', methods=['POST'])
def update_tags():
    """
    Updates tags for all existing images in the database by running object detection.
    """
    model_type = request.form.get('model', 'yolov5')  # Default model is YOLOv5
    update_tags_for_existing_images(detect_objects, model_type=model_type)
    return jsonify({'success': 'All tags updated successfully!'})

@app.route('/update_empty_tags', methods=['POST'])
def update_empty_tags():
    """
    Updates tags only for images with empty tags in the database by running object detection.
    """
    model_type = request.form.get('model', 'yolov5')  # Default model is YOLOv5
    update_tags_for_images_with_empty_tags(detect_objects, model_type=model_type)
    return jsonify({'success': 'Tags updated successfully for images with no tags!'})

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
