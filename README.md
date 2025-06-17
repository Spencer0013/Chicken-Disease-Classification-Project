# 🐔 Chicken Disease Classification: Binary Image Recognition (Coccidiosis vs. Healthy)

[![Azure Deployment](https://img.shields.io/badge/Deployed%20on-Azure%20Web%20Apps-blue)](https://drive.google.com/file/d/17Z2CfxA1oRAweGsSEKo_LlSi9hbj76pS/view?usp=sharing)
[![Model Accuracy](https://img.shields.io/badge/Accuracy-94.83%25-brightgreen)](scores.json)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<div align="center">
  <img src="https://i.imgur.com/chicken-disease-banner.jpg" alt="Project Banner" width="800">
</div>

## 🚀 Project Overview
This production-grade MLOps solution classifies chicken diseases from fecal images into two categories. Coccidiosis-infected or Healthy. with **94.83% accuracy**. Built with TensorFlow/Keras, it leverages transfer learning to enable rapid, non-invasive disease detection for poultry farmers and veterinarians. The system:

- Ingests and preprocesses image data

- Trains a VGG16-based model using transfer learning

- Evaluates model performance

- Provides a web interface for predictions

- Automatically deploys to Azure via GitHub Actions

## Key Features

- Binary classification (Coccidiosis vs. Healthy)

- DVC-managed reproducible pipeline

- Streamlit web interface

- Azure container deployment

- TensorFlow/Keras implementation

- Modular codebase with MLOps best practices

- Achieves 94.8% validation accuracy

## CI/CD Pipeline Features:

- Automatic Builds: Triggered on every push to main branch

- Containerized Deployment: Docker-based deployment to Azure

- Environment Management: Separate production slot

- Secret Management: Secure credential handling

- Rolling Updates: Zero-downtime deployments


## 🗂️ Dataset

Binary Classes:

- coccidiosis (diseased)

- healthy

# Key Statistics:

Metric	Value
- **Total images**	390
- **coccidiosis images**	195 (50%)
- **healthy images**	195 (50%)
- **Train/Validation/Test**	70%/20%/10%

# Preprocessing:

Resized to 224×224 pixels

Normalized using ImageNet mean/std

Augmentation techniques:

Random rotation (±20°)

Horizontal/Vertical flipping

Brightness/Contrast adjustment


##  🛠️ Technology Stack
- **Category**	         :           Technologies
- **Deep Learning**	     :     TensorFlow, Keras, VGG16, Image Augmentation
- **MLOps**	             :    DVC, MLflow, Hyperparameter Tracking
- **Web Framework**	     :     Streamlit
- **Cloud Deployment**	 :     Azure Web Apps, Azure Container Registry, GitHub Actions
- **Data Processing**	   :       Pandas, NumPy, OpenCV
- **Configuration**	     :     PyYAML, python-box
- **Testing**	           :       Pytest TensorFlow Model Validation

## Hyperparameters (params.yaml)

   AUGMENTATION: True               # Enable data augmentation
IMAGE_SIZE: [224, 224, 3]        # Input dimensions (compatible with VGG16)
BATCH_SIZE: 16                   # Training batch size
INCLUDE_TOP: False               # Exclude top layers of base model
EPOCHS: 5                        # Training epochs
CLASSES: 2                       # Binary classification
WEIGHTS: imagenet                # Pre-trained weights
LEARNING_RATE: 0.01              # SGD learning rate

# 📈 Performance Metrics

 {
    "loss": 0.5259,
    "accuracy": 0.9483
}

# Validation Results:

Accuracy: 94.83%

Loss: 0.5259

# Load trained model
model = load_model("artifacts/training/model.h5")

st.title("Chicken Disease Classifier")
uploaded_file = st.file_uploader("Upload fecal image", type=["jpg", "png"])

if uploaded_file:
    # Display and process image
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)
    
    # Preprocess for model input
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Make prediction
    pred = model.predict(img_array)
    result = "Healthy" if pred[0][1] > 0.5 else "Coccidiosis"
    
    # Display result
    st.success(f"**Prediction:** {result}")
    st.metric("Confidence", f"{max(pred[0])*100:.2f}%")

## 🌐 Web Application

## Application Features:

- Intuitive Interface: Simple image upload workflow

- Real-time Predictions: Instant classification results

- Confidence Scores: Probability visualization

- Mobile Responsive: Works on all devices

- Error Handling: Graceful failure on invalid inputs


  # Installation

 # Clone repository
- git clone https://github.com/Spencer0013/Chicken-Disease-Classification-Project.git
- cd Chicken-Disease-Classification-Project

# Create virtual environment
- python -m venv chicken-env
- source chicken-env/bin/activate  # Linux/Mac
- chicken-env\Scripts\activate    # Windows

# Install dependencies
- pip install -r requirements.txt

# Initialize DVC
- dvc init

 # Running the Application
 - streamlit run app.py

 Executing MLOps Pipeline

 # Run full DVC pipeline
- dvc repro

# Or execute stages individually
- python main.py

## ☁️ Azure Deployment

- The CI/CD pipeline automates:

- Docker image builds on code commits

- Container registry pushes to ACR

- Zero-downtime deployments to Azure Web Apps

# Manual Deployment:
- docker build -t chicken-disease-classifier .
- docker run -p 8501:8501 chicken-disease-classifier

## 📝 Portfolio Highlights

- End-to-End MLOps Implementation: From data ingestion to deployment

- Production Readiness: Dockerized application with health checks

- Reproducible Workflows: DVC-tracked experiments

- Cloud Integration: Azure deployment pipeline

- Performance Optimization: 94.83% validation accuracy

- Modular Design: Component-based architecture

## 📜 License
- This project is licensed under the MIT License - see the LICENSE file for details.
