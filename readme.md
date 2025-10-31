# 🐟 Multiclass Fish Image Classification  

## 📘 Overview
This project focuses on classifying fish images into multiple species using **Deep Learning**.  
It involves building and comparing **Convolutional Neural Network (CNN)** architectures — both trained from scratch and fine-tuned using **Transfer Learning** with pre-trained models such as **VGG16, ResNet50, MobileNetV2, InceptionV3, and EfficientNetB0**.  

A **Streamlit web application** is developed for real-time prediction of fish species from user-uploaded images.  

---

## 🎯 Objective
To develop a robust image classification model that can accurately identify fish species and deploy it as an interactive web app for real-time predictions.

---

## 🧠 Skills Demonstrated
- Deep Learning & CNNs  
- TensorFlow / Keras  
- Data Preprocessing & Augmentation  
- Transfer Learning  
- Model Evaluation & Visualization  
- Streamlit Deployment  
- Python Programming  

---

## 🌐 Domain
**Computer Vision → Image Classification**

---

## 🧩 Problem Statement
The task is to classify fish images into multiple categories. The project includes:
- Training CNN and Transfer Learning models  
- Evaluating model performance using metrics like accuracy, precision, recall, and F1-score  
- Saving the best model  
- Deploying it via a user-friendly **Streamlit app**

---

## 🏢 Business Use Cases
1. **Enhanced Accuracy:** Determine the most efficient model architecture for fish classification.  
2. **Deployment Ready:** Build a Streamlit app for real-time image prediction.  
3. **Model Comparison:** Evaluate and compare pre-trained models for practical implementation.

---

## 🧮 Approach

### 1️⃣ Data Preprocessing & Augmentation
- Images are resized to (224, 224)  
- Rescaled to `[0, 1]`  
- Augmentation includes rotation, zoom, shifting, and flipping  
- Implemented using `ImageDataGenerator` in TensorFlow  

### 2️⃣ Model Training
- Trained a CNN model from scratch  
- Experimented with five pre-trained architectures:
  - VGG16  
  - ResNet50  
  - MobileNetV2  
  - InceptionV3  
  - EfficientNetB0  
- Fine-tuned each model  
- Saved trained models in `.h5` format  

### 3️⃣ Model Evaluation
- Compared accuracy, precision, recall, and F1-score  
- Visualized training vs validation accuracy and loss  
- Selected best-performing model (VGG16) for deployment  

### 4️⃣ Deployment
- Developed a **Streamlit app** for user interaction  
- Users can upload fish images and get:
  - Predicted species name  
  - Model confidence score  

---

## 📊 Model Performance Summary

| Model | Accuracy | Loss |
|:------|:---------:|:----:|
| EfficientNetB0 | 0.1631 | 2.2961 |
| InceptionV3 | 0.9981 | 0.0159 |
| MobileNetV2 | 0.9965 | 0.0102 |
| ResNet50 | 0.7562 | 0.7087 |
| **VGG16 (Best)** | **0.9981** | **0.0084** |

---

## 🚀 Streamlit Application

### 🧩 Key Features
- Upload any fish image (JPG, PNG)  
- Get predicted fish species instantly  
- View model confidence score  

### 💻 Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/fish-image-classification.git
   cd fish-image-classification
