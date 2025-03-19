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

### 2. Create a Virtual Environment
Create a virtual environment to manage dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```
3. Dependencies Overview

This project requires the following dependencies:
	•	Python 3.11
	•	Flask: A lightweight web framework for Python.
	•	PyTorch: An open-source machine learning library for image recognition tasks.
	•	TorchVision: Provides datasets, model architectures, and image transformations for computer vision.
	•	Pillow: Python Imaging Library for image processing.
	•	OpenCV: Library for computer vision tasks like image reading and manipulation.
	•	NumPy: For numerical computing and array manipulation.
	•	TensorFlow: Machine learning framework for model training and inference.
	
### 3. Install Dependencies
Install the required Python packages using `requirements.txt`:

```bash
pip install --upgrade pip
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





TAKE NOTE!!! IMPORTANT !
IF ANY ISSUES WITH THE TENSOR FLOW - please run the following

# Setup and Execution (SOE) Guide

## 1️⃣ Setting Up the Virtual Environment

Run the following commands in the terminal:

```sh
# Remove any existing virtual environment (optional)
rm -rf venv/

# Create a new virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel

# Install required dependencies
pip install -r requirements.txt

	`` EXAMPLE USED ON MY PROGRAM
	rm -rf /Users/chenbin/desktop/FYP_CB/ChenBinFYP/venv/
	python3.11 -m venv /Users/chenbin/desktop/FYP_CB/ChenBinFYP/venv/
	source /Users/chenbin/desktop/FYP_CB/ChenBinFYP/venv/bin/activate
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
