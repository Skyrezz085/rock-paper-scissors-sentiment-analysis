# Import necessary libraries
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import cv2
import numpy as np
import os
import random

# Load the trained model
model = load_model('rock_paper_scissors_tl_vgg16.keras')
target_size = (220, 220)

# --- Prediction Section ---
def import_and_predict(image_data, model):
    image = load_img(image_data, target_size=target_size)
    img_array = img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    # Normalize the image
    img_array = img_array / 255.0

    # Make prediction
    predictions = model.predict(img_array)

    # Get the class with the highest probability
    predicted_class = np.argmax(predictions)

    # Define class labels
    class_labels = ['paper', 'rock', 'scissors']

    # Get the predicted label
    predicted_label = class_labels[predicted_class]

    result = f"Prediction: {predicted_label}"

    return result

def prediction_section():
    st.title("Rock-Paper-Scissors Image Classification")
    file = st.file_uploader("Upload an image", type=["png"])

    if file is None:
        st.text("Please upload an image file")
    else:
        result = import_and_predict(file, model)
        st.image(file)
        st.write(result)

# --- EDA Section ---
def eda_section():
    st.title("Rock-Paper-Scissors EDA")

    # Path to image folders
    paper_folder = 'data/paper'
    rock_folder = 'data/rock'
    scissors_folder = 'data/scissors'

    # Get all image paths
    def get_image_paths(folder):
        return [os.path.join(folder, filename) for filename in os.listdir(folder) if filename.endswith(('.png'))]

    paper = get_image_paths(paper_folder)
    rock = get_image_paths(rock_folder)
    scissors = get_image_paths(scissors_folder)

    # Function to get image sizes
    def get_image_sizes(image_paths):
        sizes = []
        for image_path in image_paths:
            img = cv2.imread(image_path)
            if img is not None:
                height, width, _ = img.shape
                sizes.append((width, height))
        return sizes

    # Get image sizes
    paper_sizes = get_image_sizes(paper)
    rock_sizes = get_image_sizes(rock)
    scissors_sizes = get_image_sizes(scissors)

    # Create DataFrame
    df_sizes = pd.DataFrame(
        data={
            "Class": ["paper"] * len(paper_sizes) + ["rock"] * len(rock_sizes) + ["scissors"] * len(scissors_sizes),
            "Width": [size[0] for size in paper_sizes + rock_sizes + scissors_sizes],
            "Height": [size[1] for size in paper_sizes + rock_sizes + scissors_sizes],
        }
    )

    # --- Display Data ---
    st.subheader("Size Distribution of Images")
    st.write(df_sizes.groupby("Class").describe())

    st.subheader("Class Distribution of Images")
    class_counts = df_sizes['Class'].value_counts()
    st.write(class_counts)

    st.subheader("Sample Images")
    def display_sample_images(image_paths, label, num_images=5):
        random_paths = random.sample(image_paths, min(len(image_paths), num_images))
        for image_path in random_paths:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for Streamlit
            st.image(img, caption=f'{label}', use_column_width=True)

    # Display sample images
    display_sample_images(paper, 'Paper')
    display_sample_images(rock, 'Rock')
    display_sample_images(scissors, 'Scissors')

# --- Main Function with Sidebar ---
def run():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose Mode", ["EDA", "Prediction"])

    if app_mode == "EDA":
        eda_section()
    elif app_mode == "Prediction":
        prediction_section()

if __name__ == "__main__":
    run()