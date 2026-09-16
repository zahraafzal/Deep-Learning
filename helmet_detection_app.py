import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# Hide Streamlit menu and notifications
st.set_page_config(page_title="Helmet Detection", layout="centered")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Helmet Detection System")
st.write("Upload an image to detect helmets in the image")
st.write("")

# Load model
@st.cache_resource
def load_model():
    model = YOLO('yolov8n.pt')
    return model

# Upload image
st.write("Step 1: Upload Image")
uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    st.write("")
    st.write("Step 2: Click Detect Button")
    
    if st.button("Detect Helmets"):
        st.write("")
        st.write("Results:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.write("Detection Result")
            
            # Load model and detect
            model = load_model()
            results = model(image)
            
            # Show result
            result_img = results[0].plot()
            result_img = Image.fromarray(result_img)
            st.image(result_img, use_container_width=True)
        
        st.write("")
        st.write("Detection Details:")
        
        detections = results[0].boxes
        
        if len(detections) > 0:
            st.write(f"Total objects found: {len(detections)}")
            st.write("")
            
            
