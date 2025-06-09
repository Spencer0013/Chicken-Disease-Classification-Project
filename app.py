import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from cnnClassifier.utils.common import read_yaml, create_directories
from pathlib import Path

# Load configuration and parameters
config = read_yaml(Path("config/config.yaml"))
params = read_yaml(Path("params.yaml"))
model_dir = Path(config.training.trained_model_path).parent
create_directories([model_dir])

# Streamlit interface
st.title("Chicken Disease Classifier")

# Load model
try:
    model = load_model(config.training.trained_model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload fecal image", type=["jpg", "png"])
if uploaded_file:
    # 1) Display the uploaded image
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    try:
        # 2) Preprocess for prediction
        img_size = tuple(params.IMAGE_SIZE[:2])  # e.g. (224, 224)
        img = image.load_img(uploaded_file, target_size=img_size)
        x   = image.img_to_array(img)
        x   = np.expand_dims(x, axis=0)

        # 3) Predict
        preds = model.predict(x)
        result = np.argmax(preds, axis=1)[0]
        prediction = "Healthy" if result == 1 else "Coccidiosis"

        # 4) Show the result
        st.write(f"**Prediction:** {prediction}")

    except Exception as e:
        st.error(f"Error processing image: {e}")
