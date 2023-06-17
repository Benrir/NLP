import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from transformers import XLNetTokenizer, TFXLNetModel

st.title("News Headline Clickbait Detection")
st.markdown("---")

headline = st.text_input("Enter the news headline")

if st.button("Detect"):
    # Load the trained model with custom objects
    model = load_model('clickbait_model.h5', custom_objects={'TFXLNetModel': TFXLNetModel})

    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    input_data = tokenizer.encode_plus(
        headline,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_tensors='tf'
    )['input_ids']

    input_data = np.reshape(input_data, (1, 128))  # Reshape input data

    prediction = model.predict(input_data)
    prediction_label = "Clickbait" if prediction[0][0] >= 0.5 else "Not Clickbait"

    st.markdown("### Prediction Result:")
    st.markdown(f"The input news headline is classified as **{prediction_label}**.")
