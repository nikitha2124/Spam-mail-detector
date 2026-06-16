import streamlit as st
from src.predict import predict_message

st.title("📧 Spam Mail Detector")

message = st.text_area("Enter Message")

if st.button("Predict"):

    prediction, confidence = predict_message(message)

    if prediction == 1:
        st.error("SPAM 🚨")
    else:
        st.success("HAM ✅")

    st.write(f"Confidence: {confidence*100:.2f}%")