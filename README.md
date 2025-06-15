# 🐔 Chicken Disease Classification - End-to-End MLOps Solution

[![Azure Deployment](https://img.shields.io/badge/Deployed%20on-Azure%20Web%20Apps-blue)](https://drive.google.com/file/d/17Z2CfxA1oRAweGsSEKo_LlSi9hbj76pS/view?usp=sharing)
[![Model Accuracy](https://img.shields.io/badge/Accuracy-94.83%25-brightgreen)](scores.json)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<div align="center">
  <img src="https://i.imgur.com/chicken-disease-banner.jpg" alt="Project Banner" width="800">
</div>

## 🚀 Project Overview
This production-grade MLOps solution classifies chicken diseases from fecal images with **94.83% accuracy**. The system implements a complete deep learning pipeline:
- **VGG16 transfer learning** for image classification
- **DVC-managed ML workflows** with pipeline versioning
- **Streamlit web interface** for real-time predictions
- **Azure deployment** via Docker and GitHub Actions
- **TensorBoard integration** for experiment tracking

## ✨ Key Features
- **Automated MLOps Pipeline**: Data ingestion → Model prep → Training → Evaluation
- **Real-time Prediction API**: Streamlit interface for instant classification
- **Production Deployment**: Docker containerization with Azure CI/CD
- **Model Optimization**: Layer freezing, data augmentation, learning rate scheduling
- **Reproducible Experiments**: DVC-tracked parameters and metrics


##  🛠️ Technology Stack
- **Category**	         :           Technologies
- **Deep Learning**	     :     TensorFlow, Keras, VGG16, Image Augmentation
- **MLOps**	             :    DVC, MLflow, Hyperparameter Tracking
- **Web Framework**	     :     Streamlit
- **Cloud Deployment**	 :     Azure Web Apps, Azure Container Registry, GitHub Actions
- **Data Processing**	   :       Pandas, NumPy, OpenCV
- **Configuration**	     :     PyYAML, python-box
- **Testing**	           :       Pytest TensorFlow Model Validation



  # 📈 Performance Metrics

 {
    "loss": 0.5259,
    "accuracy": 0.9483
}

 # Clone repository
git clone https://github.com/Spencer0013/Chicken-Disease-Classification-Project.git
cd Chicken-Disease-Classification-Project

# Create virtual environment
- python -m venv chicken-env
- source chicken-env/bin/activate  # Linux/Mac
- chicken-env\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Initialize DVC
dvc init

 # Running the Application
 streamlit run app.py

 Executing MLOps Pipeline

 # Run full DVC pipeline
dvc repro

# Or execute stages individually
python main.py

## ☁️ Azure Deployment

The CI/CD pipeline automates:

Docker image builds on code commits

Container registry pushes to ACR

Zero-downtime deployments to Azure Web Apps

# Manual Deployment:
docker build -t chicken-disease-classifier .
docker run -p 8501:8501 chicken-disease-classifier

## 📝 Portfolio Highlights

End-to-End MLOps Implementation: From data ingestion to deployment

Production Readiness: Dockerized application with health checks

Reproducible Workflows: DVC-tracked experiments

Cloud Integration: Azure deployment pipeline

Performance Optimization: 94.83% validation accuracy

Modular Design: Component-based architecture

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
