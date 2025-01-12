# Ragging Detection Project

This project aims to detect violent behavior in videos using deep learning. It includes frame processing, violence detection, and face recognition functionalities.

## Files and Directories

- **ragging_detection.py**: The main script for detecting violent behavior and handling frame processing.
- **encode.py**: A helper script for encoding face data.
- **fin.py**: Script for face recognition.
- **violence-detection.ipynb**: Jupyter notebook for training the violence detection model using TensorFlow. Includes data preprocessing, model building, and training steps.
- **models**: Directory containing pre-trained models (`.h5` files) used for violence detection.
- **faces/base**: Directory with base face images for encoding.
- **faces/processed_faces**: Contains processed face encodings.
- **requirements.txt**: List of dependencies required for the project.
- **readme.md**: Documentation for understanding and setting up the project.

### Ignored Files and Directories

The following files and directories are ignored and not included in version control:
- Generated or processed files (`detected_individuals.txt`, etc.).
- Temporary or intermediate directories (`proc_ragging_frames`, `ragging_frames`).
- Images, encodings, and other large or unnecessary files listed in `.gitignore`.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/iamrahulr30/ragging_detection.git
   cd ragging_detection


2. add faces jpg in faces/base
   ```bash 
    python encode.py

3. Run the requirements file to download important dependencies
   ```bash
    pip install -r requirements.txt

4. run voilence detection
   ```bash 
    python voilence-detection.py

5. run facial_recognition file
   ```bash 
    python fin.py
