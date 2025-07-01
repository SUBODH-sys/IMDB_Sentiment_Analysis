import streamlit as st
import pickle
import os
from streamlit.components.v1 import html

# Path to the model and vectorizer
model_path = r'D:\DeviXy'

# Load Logistic Regression model and vectorizer
try:
    with open(os.path.join(model_path, 'lr_model.pkl'), 'rb') as f:
        model_lr = pickle.load(f)
    with open(os.path.join(model_path, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("Model or vectorizer not found. Please ensure 'lr_model.pkl' and 'vectorizer.pkl' are in D:\\DeviXy\\")
    st.stop()

# Streamlit app
st.title("IMDb Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive or Negative) using Logistic Regression.")

# Text input
review = st.text_area("Enter your review:", "This movie was amazing!", height=150)

# CSS for custom styling
st.markdown(
    """
    <style>
    .prediction-box {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
        margin: 20px auto;
        width: 50%;
    }
    .positive {
        background-color: #c8e6c9;
        color: #2e7d32;
    }
    .negative {
        background-color: #ffcdd2;
        color: #c62828;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if st.button("Predict"):
    # Preprocess and predict
    review_vectorized = vectorizer.transform([review])
    prediction = model_lr.predict(review_vectorized)[0]
    label = "Positive" if prediction == 1 else "Negative"
    css_class = "positive" if prediction == 1 else "negative"

    # Display prediction with styled box
    st.markdown(
        f'<div class="prediction-box {css_class}">Prediction: {label}</div>',
        unsafe_allow_html=True
    )

    # Animation based on prediction
    if prediction == 1:
        st.balloons()
    else:
        st.snow()
st.write("**Note**: This app uses a Logistic Regression model trained on the IMDb dataset with TF-IDF vectorization for fast and accurate sentiment prediction.")
