import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load trained model
model = YOLO("best.pt")

st.title("Counterfeit Brand Detection")

uploaded_file = st.file_uploader("Upload Shoe Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    results = model.predict(image)

    probs = results[0].probs.data.tolist()
    names = results[0].names

    predicted_class = names[probs.index(max(probs))]
    confidence = max(probs)

    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")