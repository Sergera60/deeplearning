import streamlit as st
import cv2, numpy as np
from utils.inference_utils import load_models, preprocess_image, predict_emotion

st.set_page_config(page_title="Image Emotion Recognition 🖼️", layout="wide")

st.title("🖼️ Image Emotion Recognition")
st.write("Upload a facial image and detect the emotion. (We repeat the image to T=16 frames.)")

use_face_crop = st.toggle("Use face crop", value=True)

uploaded_img = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_img:
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Could not read the image. Please try another file.")
    else:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

        ae, model = load_models()
        with st.spinner("Running inference…"):
            x = preprocess_image(img_bgr, use_face_crop=use_face_crop)   # [1,16,1,112,112]
            emotion, probs = predict_emotion(model, ae, x)

        st.markdown(f"### 🧠 Predicted Emotion: **{emotion}**")
        st.bar_chart(probs)
