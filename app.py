from ultralytics import YOLO
import streamlit as st
from PIL import Image

st.title("Fake Brand Detection")

model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    results = model.predict(image)

    probs = results[0].probs
    names = results[0].names

    top1 = probs.top1
    confidence = probs.top1conf

    st.success(f"Prediction: {names[top1]}")
    st.write(f"Confidence: {confidence:.2f}")
