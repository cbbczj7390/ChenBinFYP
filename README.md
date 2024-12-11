# ChenBinFYP


# ChenBinFYP: Image Recognition System with YOLOv5 & ResNet50

## 🚀 Features

1. **Single and Batch Image Upload**: Upload one or multiple images and get results in a structured format.

2. **Object Detection**: Uses YOLOv5 to detect objects within the uploaded images.

3. **Similarity Search**: Compares the uploaded images to those stored in the database using ResNet50 features.

4. **User-Friendly UI**: A modern, responsive design using Bootstrap.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
First, clone the repository using the following command:

```bash
git clone https://github.com/cbbczj7390/ChenBinFYP.git
cd ChenBinFYP
```
Install Flask using pip:
pip install Flask

If your project has a requirements.txt file, you can install all dependencies at once:
pip install -r requirements.txt



Step 3: Verify Flask Installation
pip show Flask

pip install torch

pip install pandas

python3 add_images_to_db.py



### 2. Create a Virtual Environment
Create a virtual environment to manage dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Install the required Python packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Run the Application
After setting up the environment, run the Flask app:

```bash
python3 app.py
```

Visit `http://127.0.0.1:5000` on your browser to use the application.

---

## 📝 How to Use

1. **Upload Single Image**: Use the "Upload Single Image" option to upload one image for object detection and similarity search.
2. **Upload Multiple Images**: Use the "Upload Multiple Images" option to upload several images at once, and results for each image will be displayed sequentially.

---

## 💻 Tech Stack

- **Backend**: Flask (Python)
- **Object Detection**: YOLOv5 (for detecting objects in images)
- **Similarity Search**: ResNet50 features (for comparing images)
- **Frontend**: HTML, CSS, Bootstrap (for responsive design)

---




pip install Flask

1. Ensure Dependencies are Installed
Install the necessary libraries if you haven’t already:
pip install Flask tensorflow keras numpy pillow

1. Install Homebrew
Run the following command in your terminal to install Homebrew:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

Add Homebrew to Your PATH
Run these commands in your terminal:
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/chenbin/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

Verify Homebrew Installation
Check if Homebrew is now available by running:
brew --version

Next Step: Install MySQL
Now that Homebrew is set up, install MySQL:
brew install mysql


🚀 Step-by-Step Guide for Object Detection Integration with YOLOv5

We will:
	1.	Set up YOLOv5 for object detection.
	2.	Modify your Flask app to detect objects in uploaded images.
	3.	Display detected objects with bounding boxes.
	4.	Integrate object detection results with the similarity comparison.