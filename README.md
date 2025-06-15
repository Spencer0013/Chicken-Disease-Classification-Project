# 🐔 Chicken Disease Classification - End-to-End MLOps Solution

[![Azure Deployment](https://img.shields.io/badge/Deployed%20on-Azure%20Web%20Apps-blue)](https://your-app.azurewebsites.net)
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

## 🧠 Technical Architecture
```mermaid
graph TD
    A[Data Ingestion] -->|DVC| B[Base Model Prep]
    B -->|TensorFlow| C[Model Training]
    C -->|Callbacks| D[Model Evaluation]
    D -->|scores.json| E[Prediction API]
    E -->|Streamlit| F[Azure Deployment]

    🛠️ Technology Stack
Category	Technologies
Deep Learning	TensorFlow, Keras, VGG16, Image Augmentation
MLOps	DVC, MLflow, Hyperparameter Tracking
Web Framework	Streamlit
Cloud Deployment	Azure Web Apps, Azure Container Registry, GitHub Actions
Data Processing	Pandas, NumPy, OpenCV
Configuration	PyYAML, python-box
Testing	Pytest, TensorFlow Model Validation


Chicken-Disease-Classification-Project/
├── artifacts/               # DVC-tracked outputs
│   ├── data_ingestion/      # Raw and processed images
│   ├── prepare_base_model/  # VGG16 base models
│   └── training/            # Trained model files
├── config/
│   ├── config.yaml          # Path configurations
│   └── params.yaml          # Model hyperparameters
├── src/
│   └── cnnClassifier/
│       ├── components/      # Pipeline components
│       │   ├── data_ingestion.py
│       │   ├── prepare_base_model.py
│       │   ├── prepare_callbacks.py
│       │   ├── training.py
│       │   └── evaluation.py
│       ├── pipeline/        # DVC stage implementations
│       │   ├── stage_01_data_ingestion.py
│       │   ├── stage_02_prepare_base_model.py
│       │   ├── stage_03_training.py
│       │   └── stage_04_evaluation.py
│       ├── utils/           # Helper functions
│       ├── entity/          # Configuration schemas
│       └── constants/       # Project constants
├── tests/                   # Unit and integration tests
├── .github/workflows/       # CI/CD pipelines
├── research/                # Experimental notebooks
├── app.py                   # Streamlit prediction interface
├── predict.py               # Prediction pipeline
├── main.py                  # DVC pipeline executor
├── dvc.yaml                 # DVC pipeline definition
├── Dockerfile               # Container configuration
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

📈 Performance Metrics

 {
    "loss": 0.5259,
    "accuracy": 0.9483
}

 Installation

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

 ## Running the Application
 streamlit run app.py

 Executing MLOps Pipeline

 # Run full DVC pipeline
dvc repro

# Or execute stages individually
python main.py

☁️ Azure Deployment

The CI/CD pipeline automates:

Docker image builds on code commits

Container registry pushes to ACR

Zero-downtime deployments to Azure Web Apps

Manual Deployment:
docker build -t chicken-disease-classifier .
docker run -p 8501:8501 chicken-disease-classifier

📝 Portfolio Highlights
End-to-End MLOps Implementation: From data ingestion to deployment

Production Readiness: Dockerized application with health checks

Reproducible Workflows: DVC-tracked experiments

Cloud Integration: Azure deployment pipeline

Performance Optimization: 94.83% validation accuracy

Modular Design: Component-based architecture

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.