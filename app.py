# app.py
from flask import Flask, request, render_template
import os
from features import extract_features
from database import init_db, save_image_features, query_database, get_all_images

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
        # Extract features and save to database
        features = extract_features(filepath)
        save_image_features(filepath, features)
        # Find similar images
        similar_images = query_database(features)
        # Render the template with similar images
        return render_template('similar_images.html', uploaded_image=filepath, similar_images=similar_images)

@app.route('/view_images')
def view_images():
    image_paths = get_all_images()
    return render_template('view_images.html', image_paths=image_paths)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)