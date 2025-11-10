import streamlit as st
import cv2, numpy as np, tempfile
from utils.inference_utils import load_models, preprocess_frames, predict_emotion

st.set_page_config(page_title="Video Emotion Recognition 🎥", layout="wide")

st.title("🎥 Video Emotion Recognition")
st.write("Upload a short video clip and let the model predict the dominant emotion (T=16, grayscale 112×112).")

use_face_crop = st.toggle("Use face crop", value=True)

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

def _sample_preview_frames(video_path, num_preview=5, size=(224, 224)):
    frames_rgb = []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return frames_rgb
    idx = np.linspace(0, total - 1, num_preview).astype(int)
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok: continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        frames_rgb.append(frame)
    cap.release()
    return frames_rgb

if uploaded_video:
    suffix = "." + uploaded_video.name.split(".")[-1].lower()
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    st.video(video_path)

    with st.spinner("Processing video and extracting frames…"):
        x = preprocess_frames(video_path, frame_count=16, use_face_crop=use_face_crop)  # [1,16,1,112,112]

    st.success(f"✅ Extracted {int(x.shape[1])} frames. Running prediction...")

    ae, model = load_models()

    with st.spinner("Analyzing emotions..."):
        emotion, probs = predict_emotion(model, ae, x)

    st.markdown(f"### 🧠 Predicted Emotion: **{emotion}**")
    st.bar_chart(probs)

    st.markdown("### 🎞️ Sample Frames")
    previews = _sample_preview_frames(video_path, num_preview=5)
    if previews:
        cols = st.columns(len(previews))
        for img, c in zip(previews, cols):
            c.image(img, use_container_width=True)
    else:
        st.info("Couldn’t extract preview frames (video may be too short or unreadable).")
