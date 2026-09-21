# Deep Face Detection with Augmentation-Based Training

## Overview
This project focuses on building a deep face detection and recognition system for attendance and identity verification. The model was trained using a large dataset of nearly 5,000 face images in Google Colab, where the augmentation pipeline was applied to improve robustness against pose variation, lighting changes, scale differences, and partial face visibility.

The repository contains the project analysis notebooks, the implementation logic, and the deployment files for the web application. The large raw and augmented image folders were intentionally removed from the GitHub repo to keep the repository lightweight and because these datasets are generated and processed in the training pipeline during Colab execution.

## Objective
The goal is to train a reliable face detection model that can work on real-world images collected from different conditions, while also improving performance through data augmentation. The approach emphasizes the idea that a model trained with a strong augmentation pipeline learns more generalized facial features and can perform better across diverse scenarios.

## Training Workflow

### 1. Data Collection
- Nearly 5,000 face images were used for training and evaluation.
- Images were organized into subject folders and then processed in Google Colab for experimentation.
- The dataset was prepared so that the model could learn identity-specific and face-detection-specific features effectively.

### 2. Augmentation Pipeline
The augmentation pipeline is a critical part of the project and was used to expand the training data without needing to collect thousands of new images manually. Typical transformations included:

- Rotation
- Horizontal flipping
- Zooming / scaling
- Shift in width and height
- Brightness and contrast adjustment
- Shearing
- Gaussian blur or noise variation
- Normalization and resizing

This helps the system become more robust to real-world variations such as tilted faces, different camera angles, partial occlusion, and inconsistent lighting.

### 3. Model Training in Google Colab
The training and validation process was performed in Google Colab because it offers GPU acceleration and a practical workflow for large image datasets. The notebooks in this repository were designed to run on custom data and should be adapted to the dataset structure being used locally or in the cloud.

### 4. Deployment and Recognition
The trained face detection and recognition system is integrated with the web application in the `/web` folder. The deployment workflow uses the processed face embeddings and detection pipeline to perform live attendance or recognition tasks.

## Tools & Libraries
- Python
- OpenCV
- TensorFlow / Keras
- NumPy / Pandas
- Matplotlib / Seaborn
- Scikit-learn
- Google Colab

## Repository Structure
- `analysis/` — research notebooks and experiment files
- `data/users/` — user directories used for recognition and attendance-related workflows
- `data/Attendance/` — attendance logs
- `model/` — trained model files and Haar cascade resources
- `web/` — Flask web app, embedding generation, and deployment scripts

## Important Note About Large Datasets
The large augmentation folders previously stored in the repository were removed from the GitHub repo because they were used only as temporary training artifacts generated during the Colab pipeline. These folders can be recreated locally from the notebook workflow whenever needed.

This means the repository now focuses on the reusable project code and workflow rather than storing thousands of large image files directly in GitHub.

## Notebook Usage
The notebooks in the `analysis/` folder are meant to be used as a reference for learning the augmentation-based training process. If you want to reuse the project on your own data, the recommended flow is:

1. Prepare your dataset.
2. Learn the augmentation pipeline used in the notebook.
3. Apply the same transformations to your training images.
4. Modify the dataset paths and labels in the notebook.
5. Run the training notebook in Google Colab or another GPU-enabled environment.

## Files in This Repository
| File | Description |
|---|---|
| `analysis/DeepFaceDetection (1).ipynb` | Face detection experimentation and augmentation-based training workflow |
| `analysis/deepfaceRecognition.ipynb` | Face recognition evaluation and embedding analysis |
| `web/app.py` | Web application entry point |
| `web/build_embeddings.py` | Embedding generation pipeline |
| `web/facetracker.py` | Face tracking and detection helper logic |

## Relation to the Deployment Project
The production-ready recognition and detection implementation is maintained in the `/web` folder. The notebooks in the analysis folder document the experimental design and training strategy that informed the final deployment system.

## Academic Context
- Project type: Final Year Project
- Focus: Deep face detection and recognition with augmentation for improved robustness
- Environment: Google Colab, Python, Keras, OpenCV
- Application: Real-time attendance and face-based identification system

## Final Note
This project demonstrates that strong augmentation pipelines are essential when training deep face detection systems on a medium-sized dataset. By generating many varied versions of the same face images, the model learns to generalize better, which is especially important in real-world environments where pose, lighting, and camera conditions vary.
