import streamlit as st
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from PIL import Image
import io

load_dotenv(find_dotenv())

def img2text(image):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(image)[0]["generated_text"]
    return text

# Streamlit interface
st.title("Image Caption Generator")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Add generate button
    if st.button('Generate Caption'):
        with st.spinner('Generating caption...'):
            # Generate caption
            caption = img2text(image)
            st.write("**Generated Caption:**")
            st.write(caption)

