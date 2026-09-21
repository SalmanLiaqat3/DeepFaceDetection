# Deep Face Detection with Enhanced Robustness to Pose Variations

## Overview
This folder contains the core analytical and research work of my Final Year Project, 
**"Deep Face Detection Model with Enhanced Robustness to Variation of Poses."** 
It focuses on the experimental design, statistical evaluation, and performance 
analysis of a deep learning-based face detection and recognition pipeline — 
independent of the deployment/web application, which is maintained separately 
in the `/web` folder.

## Objective
To investigate how variations in face pose (yaw, pitch, roll) affect the 
detection and recognition accuracy of deep learning models, and to design 
an approach that improves robustness under these conditions.

## Methodology

### 1. Data Preparation & Exploratory Analysis
- Dataset composition and class distribution analysis
- Pose-angle distribution across the dataset (frontal vs. angled/tilted faces)
- Image quality assessment (resolution, lighting variance, occlusion checks)
- Preprocessing pipeline: face alignment, normalization, augmentation strategy

### 2. Embedding & Similarity Analysis
- Analysis of deep face embeddings produced by the FaceNet model 
  (embedding generation pipeline maintained in [`/web/build_embeddings.py`](../web/build_embeddings.py))
- Dimensionality and embedding space analysis
- Cosine similarity distribution between same-identity vs. different-identity pairs

### 3. Model Evaluation
- Detection accuracy across varying pose angles (quantitative breakdown)
- Recognition performance metrics: **accuracy, precision, recall, F1-score**
- Confusion matrix analysis
- ROC curve and threshold tuning for similarity-based matching
- Comparative performance: baseline detection vs. pose-robust approach

### 4. Statistical Analysis
- Hypothesis testing / statistical significance of performance differences 
  across pose categories (where applicable)
- Error analysis: identifying failure cases and correlating them with 
  pose extremity, lighting, or occlusion

## Tools & Libraries
- **Python** — core analysis language
- **NumPy, Pandas** — data handling and statistical computation
- **Scikit-learn** — evaluation metrics, similarity/threshold analysis
- **TensorFlow / Keras (FaceNet)** — deep embedding evaluation
- **OpenCV** — image preprocessing
- **Matplotlib / Seaborn** — data visualization
- **Google Colab** — experimentation environment (GPU-accelerated)

## Files in this Folder
| File | Description |
|---|---|
| `DeepFaceDetection (1).ipynb` | Face detection model experimentation, pose-variation testing, and detection performance analysis |
| `deepfaceRecognition.ipynb` | Face recognition evaluation: embedding similarity analysis and performance metrics |

## Key Findings
*(Fill in with your actual results, e.g.:)*
- Detection accuracy dropped by **X%** for faces with yaw angles beyond **Y°**, 
  motivating the pose-robust preprocessing approach
- The proposed method improved recognition F1-score from **X%** (baseline) 
  to **Y%** under high pose variation
- Cosine similarity threshold of **X** was found optimal for balancing 
  false acceptance and false rejection rates

## Relation to Deployment
The embedding generation pipeline (`build_embeddings.py`) and full production 
implementation are maintained in [`/web`](../web). This folder documents the 
underlying analysis and evaluation that informed key design decisions in that 
system — including similarity thresholds and pose-handling strategy.

## Academic Context
- **Project Type:** Final Year Project (BSc Software Engineering)
- **Institution:** University of Lahore
- **Focus Area:** Applied Deep Learning & Statistical Model Evaluation combined with deep face recognition** can be effectively used to build a reliable, real-time, and intelligent attendance system through a modern web dashboard.
