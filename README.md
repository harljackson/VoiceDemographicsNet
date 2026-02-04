🎙️ VoiceDemographicsNet

Speech-Based Gender Classification with Machine Learning

🔍 Overview

VoiceDemographicsNet is a machine learning project that explores automatic speaker gender classification from short speech recordings. The system extracts acoustic features from audio and uses a neural network to predict gender while prioritising fairness, generalisation, and confidence-aware predictions.

This project was developed as part of a university Machine Learning coursework and focuses on responsible and reproducible speech processing.

✨ Features

🎧 Acoustic feature extraction using MFCCs

📈 Optional pitch feature extraction using the YIN algorithm

🧠 Multi-Layer Perceptron (MLP) neural network classifier

⚖️ Class-weighted training to reduce majority-class bias

🔎 Confidence-aware inference with probability thresholds

🧪 Quantitative + qualitative evaluation (including unseen samples)

🌐 Interactive Streamlit web app for real-time predictions

📊 Dataset

The models were trained and evaluated on a 1,000-sample subset of the Mozilla Common Voice v23.0 (English) dataset.

⚠️ Note
Raw audio data is not included due to dataset size and licensing.
All preprocessing and experimental details are documented in the report.

🗂️ Project Structure
VoiceDemographicsNet/
├── notebooks/        # Data processing & model training notebooks
├── src/              # Core feature extraction and training code
├── models/           # Trained models (.keras) and scalers (.joblib)
├── app.py            # Streamlit application
├── styles.css        # Custom UI styling
├── requirements.txt  # Python dependencies
└── README.md

🛠️ Tech Stack

🐍 Python 3

📦 NumPy, Pandas

🎼 Librosa

🤖 TensorFlow / Keras

📐 Scikit-learn

📊 Matplotlib & Seaborn

🌐 Streamlit

▶️ Run the App

To launch the interactive demo locally:

pip install -r requirements.txt
streamlit run app.py


Upload a short speech recording (.wav or .mp3) to receive a prediction with confidence feedback.

⚖️ Ethics & Responsible AI

Speech-based demographic inference is ethically sensitive. This project incorporates:

class imbalance mitigation

probabilistic outputs instead of hard decisions

explicit confidence reporting

The system is intended for research and educational use only.

👤 Author
Harley Jackson
