# Chicken Disease Classification: Binary Image Recognition (Coccidiosis vs. Healthy)

[![Deployed on Azure Web Apps](https://img.shields.io/badge/Deployed%20on-Azure%20Web%20Apps-blue)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-94.83%25-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

<div align="center"> 
  <img src="https://drive.google.com/file/d/1ngPEyAuz6JKzWm2QqBQCIw6v4yaBcLqf/view?usp=sharing" alt="Chicken Disease Classification" width="800"> 
</div>

## Access the Web App
**Deployed Application URL**:  
`https://<your-app-name>.azurewebsites.net`  

1. Visit the URL above (replace `<your-app-name>` with your actual Azure app name)
2. Upload a chicken image
3. Get instant classification: Healthy or Coccidiosis

🚀 Project Overview
This production-grade MLOps solution classifies chicken diseases from fecal images into two categories: Coccidiosis-infected or Healthy with 94.83% accuracy. Built with TensorFlow/Keras, it enables rapid, non-invasive disease detection for poultry farmers and veterinarians.

Key Features:

- Binary classification (Coccidiosis vs. Healthy)

- DVC-managed reproducible pipeline

- Streamlit web interface

- Azure container deployment

- Achieves 94.8% validation accuracy

- Modular MLOps architecture

## 📊 Performance Metrics
{  
    "loss": 0.5259,  
    "accuracy": 0.9483  
}  

# Validation Results:
- Accuracy: 94.83%

- Loss: 0.5259

## 🗂️ Dataset
- coccidiosis (diseased): 195 images (50%)

- healthy: 195 images (50%)

## ⚙️ Technology Stack
- **Category**	  Technologies
- **Deep Learning**	  TensorFlow, Keras, VGG16
- **MLOps**	DVC, MLflow, Hyperparameter Tracking
- **Web Framework**	Streamlit
- **Cloud Deployment**	Azure Web Apps, Azure Container Registry
- **Data Processing**	Pandas, NumPy, OpenCV
- **Configuration**	PyYAML, python-box

# 🧠 Model Architecture
- Base Model: VGG16 with ImageNet weights

- Transfer Learning: Freeze convolutional layers, train custom top layers

- Output Layer: 2 neurons with softmax activation

# 🛠️ Hyperparameters (params.yaml)
AUGMENTATION: True  
IMAGE_SIZE: [224, 224, 3]  
BATCH_SIZE: 16  
INCLUDE_TOP: False  
EPOCHS: 5  
CLASSES: 2  
WEIGHTS: imagenet  
LEARNING_RATE: 0.01  

# 🌐 Web Application Features
- Intuitive image upload interface

- Real-time predictions

- Mobile-responsive design

- Prediction confidence display

# app.py 
st.title("Chicken Disease Classifier")  
uploaded_file = st.file_uploader("Upload fecal image", type=["jpg", "png"])  
if uploaded_file:  
    st.image(uploaded_file)  
    prediction = model.predict(preprocess_image(uploaded_file))  
    st.success(f"**Prediction:** {'Healthy' if prediction == 1 else 'Coccidiosis'}")  

# 🚦 MLOps Pipeline
Pipeline Stages:

Data Ingestion: Download and extract dataset

Prepare Base Model: Configure VGG16 with custom layers

Training: Train model with callbacks and augmentation

Evaluation: Validate performance and save metrics

Deployment: Containerize and deploy to Azure

# 💻 Installation
# Clone repository  
git clone https://github.com/Spencer0013/Chicken-Disease-Classification-Project.git  
cd Chicken-Disease-Classification-Project  

# Create virtual environment  
python -m venv chicken-env  
source chicken-env/bin/activate  # Linux/Mac  
chicken-env\Scripts\activate    # Windows  

# Install dependencies  
pip install -r requirements.txt  

# Initialize DVC  
dvc init  

# 🚀 Usage
dvc repro  

# Execute Stages Individually:
python main.py  

# Start Web Application:
streamlit run app.py 

# Prediction Pipeline:
from predict import PredictionPipeline  

predictor = PredictionPipeline("path/to/image.jpg")  
result = predictor.predict()  # Returns {'image': 'Healthy'} or {'image': 'Coccidiosis'}  

# ☁️ Azure Deployment
The CI/CD pipeline automates:

Docker image builds on code commits

Container registry pushes to ACR

Zero-downtime deployments to Azure Web Apps

# Manual Deployment:
docker build -t chicken-disease-classifier .  
docker run -p 8501:8501 chicken-disease-classifier

# 🏆 Portfolio Highlights
- End-to-End MLOps Implementation: From data ingestion to deployment

- Production-Ready: Dockerized with health checks

- Reproducible Workflows: DVC-tracked experiments

- Cloud Integration: Azure deployment pipeline

- Modular Design: Component-based architecture

# 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

