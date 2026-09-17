import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import ssl
import certifi

# Fix SSL certificate issue
ssl._create_default_https_context = ssl._create_unverified_context

st.title("Cat vs Dog Image Classification")

# Pre-trained Deep Learning model
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()

# Image transformation
transform = weights.transforms()

uploaded_file = st.file_uploader(
    "Upload a cat or dog image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    # Prepare image
    image_tensor = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        output = model(image_tensor)

    prediction = output.argmax(1).item()
    label = weights.meta["categories"][prediction]

    st.success(f"Prediction: {label}")