import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from PIL import Image
import base64
from gtts import gTTS
import io

# Function to get base64 encoding of a file
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set the background of the Streamlit app
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/jpeg;base64,%s");
    background-position: center;
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown('<style>h1 { color: Black ; }</style>', unsafe_allow_html=True)
    st.markdown('<style>p { color: Black; }</style>', unsafe_allow_html=True)
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('background/2.jpg')

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from PIL import Image
import base64
from gtts import gTTS
import io

# Function to get base64 encoding of a file
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set the background of the Streamlit app
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/jpeg;base64,%s");
    background-position: center;
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown('<style>h1 { color: Black ; }</style>', unsafe_allow_html=True)
    st.markdown('<style>p { color: Black; }</style>', unsafe_allow_html=True)
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('background/2.jpg')

import pandas as pd  # Import pandas for DataFrame
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image as PILImage

# Load pre-trained models
mobilenet_model = load_model('fake_currrecy_detection.h5')
cnn_model = load_model('cnn_model.h5')
resnet_model = load_model('resnet_model.h5')

# Class labels (assuming they are 'Fake' and 'Real')
class_labels = ['Fake', 'Real']

# Streamlit UI elements
st.title("Fake Currency Detection Using AI Models")
st.write("Upload an image of a banknote to predict if it's fake or real.")

uploaded_image = st.file_uploader("Choose an image...", type="jpg")

if uploaded_image is not None:
    # Open image using PIL
    img = PILImage.open(uploaded_image)

    # Display uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    img = img.resize((224, 224))  # Resize image to fit model input
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Predictions from different models
    def predict_class(model):
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction[0])

        # Ensure predicted index is within bounds of class_labels
        predicted_class_index = min(predicted_class_index, len(class_labels) - 1)

        predicted_class_label = class_labels[predicted_class_index]
        return predicted_class_label, prediction[0][predicted_class_index]
    
    # Show predictions for each model
    st.subheader("Model Predictions")

    # MobileNet prediction
    mobilenet_class, mobilenet_confidence = predict_class(mobilenet_model)
    st.write(f"MobileNet Prediction: {mobilenet_class} (Confidence: {mobilenet_confidence*100:.2f}%)")

    # CNN prediction
    cnn_class, cnn_confidence = predict_class(cnn_model)
    st.write(f"CNN Model Prediction: {cnn_class} (Confidence: {cnn_confidence*100:.2f}%)")

    # ResNet50 prediction
    resnet_class, resnet_confidence = predict_class(resnet_model)
    st.write(f"ResNet50 Prediction: {resnet_class} (Confidence: {resnet_confidence*100:.2f}%)")

    # Plot comparison of model accuracies
    model_names = ['MobileNet', 'CNN', 'ResNet50']
    accuracies = [mobilenet_confidence, cnn_confidence, resnet_confidence]

    data = {'Model': model_names, 'Accuracy': accuracies}
    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 6))
    sns.barplot(x='Model', y='Accuracy', data=df)
    plt.title('Model Comparison on Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    st.pyplot(plt)

