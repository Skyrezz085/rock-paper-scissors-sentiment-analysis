# Import necessary libraries
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = load_model('rock_paper_scissors_tl_vgg16.keras')
target_size = (220, 220)

# Function to import and predict
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

# Streamlit app
def run():
    st.title("Rock-Paper-Scissors Image Classification")
    file = st.file_uploader("Upload an image", type=["png"])

    if file is None:
        st.text("Please upload an image file")
    else:
        result = import_and_predict(file, model)
        st.image(file)
        st.write(result)

if __name__ == "__main__":
    run()