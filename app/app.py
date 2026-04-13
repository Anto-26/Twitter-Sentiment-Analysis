import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


# Title
st.title("🐦 Twitter Sentiment Analysis App")
st.write("Enter a tweet and the model will predict sentiment.")

# Input box
user_input = st.text_area("Enter Tweet:")

# Predict button
if st.button("Predict Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        # Transform input
        X = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(X)[0]

        # Map output
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😡"

        st.success(f"Prediction: {sentiment}")