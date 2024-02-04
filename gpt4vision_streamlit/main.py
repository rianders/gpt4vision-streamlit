import streamlit as st
import openai
import base64
import requests
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def encode_image_to_base64(image_file):
    # Convert the image file to base64
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

def get_openai_vision_response(image_base64, question):
    # Prepare the payload
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        "max_tokens": 300
    }

    # Make the request to OpenAI
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        },
        json=payload
    )

    return response.json()

st.title("Image Analysis with GPT-4 Vision")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
question = st.text_input("Ask a question about the image:")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Process the image and send to OpenAI Vision API
    if st.button("Analyze Image"):
        base64_image = encode_image_to_base64(uploaded_file)
        response = get_openai_vision_response(base64_image, question)
        st.write("Response from OpenAI Vision API:")
        st.json(response)
