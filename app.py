import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Fake Brand Detection using Deep Learning")

model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)

    results = model.predict(img)

    probs = results[0].probs
    names = results[0].names

    predicted_class = names[probs.top1]
    confidence = float(probs.top1conf)

    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
