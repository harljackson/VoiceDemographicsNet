# UI

# Imports
import streamlit as st
import numpy as np
import librosa
import os
from tensorflow.keras.models import load_model
from joblib import load

# Load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



# Load model and scaler
@st.cache_resource
def load_models():
    gender_model = load_model(
        "models/mlp_gender_classification_model.keras",
        compile=False
    )
    scaler = load("models/feature_scaler.joblib")
    return gender_model, scaler


gender_model, scaler = load_models()


# MFCC extraction
def extract_mfcc_from_audio(path, n_mfcc=13):
    audio, sr = librosa.load(path, sr=16000)
    audio, _ = librosa.effects.trim(audio)
    audio = librosa.util.normalize(audio)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1)


# Streamlit UI
st.set_page_config(
    page_title="Voice Gender Classifier",
    page_icon="🎙️",
    layout="centered"
)

# Custom CSS
load_css("styles.css")

# Header
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🎙️ Voice Gender Classifier")
st.markdown("**Interactive machine learning demo for voice-based gender classification**")
st.markdown('</div>', unsafe_allow_html=True)

# Info
with st.expander("ℹ️ How it works"):
    st.write("""
    This application uses a neural network trained on Mel-Frequency Cepstral Coefficients (MFCCs)
    to classify gender from short speech recordings.

    **Pipeline**
    1. Audio upload
    2. Feature extraction (MFCCs)
    3. Neural network inference
    4. Probability & confidence reporting
    """)

st.divider()

# File uploader
uploaded_file = st.file_uploader(
    "📁 Upload an audio file",
    type=["wav", "mp3"]
)

# Prediction
if uploaded_file is not None:
    with st.spinner("🎧 Extracting features and running inference..."):
        try:
            with open("temp_audio.wav", "wb") as f:
                f.write(uploaded_file.read())

            mfcc_features = extract_mfcc_from_audio("temp_audio.wav")
            mfcc_scaled = scaler.transform(mfcc_features.reshape(1, -1))

            pred = gender_model.predict(mfcc_scaled, verbose=0)
            male_prob, female_prob = pred[0]

            # Decision rule
            threshold = 0.45
            predicted_gender = "Female" if female_prob >= threshold else "Male"

            gender_icons = {"Female": "♀️", "Male": "♂️"}
            gender_icon = gender_icons[predicted_gender]

            # Confidence
            confidence_gap = abs(male_prob - female_prob)
            if confidence_gap >= 0.30:
                confidence_label = "High confidence"
                confidence_icon = "🟢"
            elif confidence_gap >= 0.15:
                confidence_label = "Moderate confidence"
                confidence_icon = "🟡"
            else:
                confidence_label = "Low confidence"
                confidence_icon = "🔴"

            # Result box
            st.markdown(f"""
            <div class="result-box">
                <div style="font-size:3rem;">{gender_icon}</div>
                <div class="result-gender">{predicted_gender}</div>
                <div class="result-confidence">
                    {confidence_icon} {confidence_label}
                </div>
            </div>
            """, unsafe_allow_html=True)


            # Probabilities
            st.subheader("📈 Model Output")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="prob-card">', unsafe_allow_html=True)
                st.markdown("**♀️ Female**")
                st.progress(float(female_prob))
                st.metric("Probability", f"{female_prob:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="prob-card">', unsafe_allow_html=True)
                st.markdown("**♂️ Male**")
                st.progress(float(male_prob))
                st.metric("Probability", f"{male_prob:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)

            st.divider()
            st.subheader("🎵 Uploaded Audio")
            st.audio(uploaded_file)

        except Exception as e:
            st.error(f"⚠️ Audio processing failed: {e}")
        finally:
            if os.path.exists("temp_audio.wav"):
                os.remove("temp_audio.wav")

else:
    st.info("👆 Upload an audio file to begin")
    with st.expander("💡 Tips for best results"):
        st.markdown("""
        - Clear speech, minimal background noise  
        - At least **1–2 seconds** of speech  
        - Normal speaking volume  
        """)

st.divider()
st.caption("🤖 Powered by TensorFlow & Librosa")
