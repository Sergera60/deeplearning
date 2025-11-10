import streamlit as st

# Page config
st.set_page_config(
    page_title="Emotion Recognition AI 🎭",
    page_icon="🎭",
    layout="wide",
)

# --- Custom CSS for modern look ---
st.markdown("""
    <style>
        /* Page background + font */
        .stApp {
            background: linear-gradient(120deg, #141E30, #243B55);
            color: white;
            font-family: "Poppins", sans-serif;
        }
        /* Center container a bit */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .main-title {
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 1.1rem;
            text-align: center;
            color: #d0d0d0;
            margin-bottom: 2.5rem;
        }
        .nav-card {
            background-color: rgba(255, 255, 255, 0.10);
            padding: 2rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 4px 24px rgba(0,0,0,0.35);
            transition: all 0.25s ease-in-out;
            border: 1px solid rgba(255,255,255,0.15);
        }
        .nav-card:hover {
            transform: translateY(-4px);
            background-color: rgba(255, 255, 255, 0.14);
        }
        .nav-title {
            font-size: 1.2rem;
            margin-top: 0.75rem;
            margin-bottom: 0.25rem;
            font-weight: 600;
        }
        .nav-desc {
            font-size: 0.95rem;
            color: #e6e6e690;
            margin-bottom: 0.75rem;
        }
        .copyright {
            color: #c7c7c7;
            opacity: 0.85;
            font-size: 0.9rem;
            margin-top: 2.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- UI Content ---
st.markdown("<h1 class='main-title'>Emotion Recognition AI 🎭</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Detect emotions from faces in <b>videos</b> or <b>images</b> using deep learning (CNN + LSTM + Autoencoder).</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='nav-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:2rem;'>🎥</div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-title'>Video Emotion Recognition</div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-desc'>Upload a short clip; we extract 16 frames and analyze the emotion.</div>", unsafe_allow_html=True)
    # ✅ Use Streamlit's native page link (reliable navigation)
    st.page_link("pages/1_Video_Emotion_Recognition.py", label="Open Video Page", icon="🎬")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='nav-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:2rem;'>🖼️</div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-title'>Image Emotion Recognition</div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-desc'>Upload a single image; we simulate a 16-frame sequence for prediction.</div>", unsafe_allow_html=True)
    # ✅ Use Streamlit's native page link (reliable navigation)
    st.page_link("pages/2_Image_Emotion_Recognition.py", label="Open Image Page", icon="🧩")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='copyright'><center>© 2025 Emotion AI Project – Deep Learning Exam</center></div>", unsafe_allow_html=True)
