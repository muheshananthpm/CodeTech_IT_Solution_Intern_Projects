# 🧠 AI & Data Science Project Portfolio

This repository contains four mini-projects in the fields of machine learning, deep learning, data engineering, and model deployment. Each project demonstrates key concepts and real-world problem-solving techniques in the AI and Data Science domain.

---

## 📁 Projects Overview

### 1. 🌸 Iris Classifier – End-to-End ML Project with FastAPI

**Goal:** Build and deploy a machine learning model using FastAPI to predict the species of an iris flower based on petal and sepal measurements.

- 💻 Trained using `RandomForestClassifier`
- ⚙️ Pipeline includes: Data Preprocessing → Model Training → API Deployment
- 🌐 REST API developed using FastAPI
- 🔬 Dataset: Scikit-learn Iris Dataset

📂 Folder: `iris_classifier/`  
📎 API Docs: `/docs` when server is running

---

### 2. 🧱 Data Pipeline Development

**Goal:** Build a reusable and modular data preprocessing pipeline using Scikit-learn's `Pipeline` and `ColumnTransformer`.

- 🧹 Handles missing values, scaling, and one-hot encoding
- 📦 Saves the processed data for future use
- 📄 Implemented in Jupyter Notebook

📂 File: `Data_Pipeline_Developement.ipynb`

---

### 3. 🖼️ Image Classification using TensorFlow

**Goal:** Classify images using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

- 🧠 CNN model architecture
- 📊 Model accuracy visualization
- 🔍 Trained on a custom or standard image dataset

📂 File: `Image_classification_Using_tensorflow.ipynb`

---

### 4. 📈 Optimization Model

**Goal:** Solve a mathematical optimization problem using Python and libraries such as PuLP or SciPy.

- 🧮 Linear programming or constraint-based optimization
- ✅ Real-world use-case (e.g., cost minimization, resource allocation)

📂 File: `Optimization_Model.ipynb`

---

## 🔧 Getting Started

### Requirements

Install required packages using:

```bash
pip install -r iris_classifier/requirements.txt

-----

### Running the Iris Classifier API

cd iris_classifier
python train_model.py
uvicorn app.main:app --reload

