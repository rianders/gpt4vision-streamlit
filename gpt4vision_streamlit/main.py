import streamlit as st
import openai
from dotenv import load_dotenv
import requests
from io import BytesIO
import os

# Load environment variables
load_dotenv()

# Set your OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_vision_response(image, question):
    response = openai.Image.create_vision(
        image=image,
        task="text",
        prompt=question
    )
    return response

st.title("Image Question Answering with OpenAI Vision API")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
question = st.text_input("Ask a question about the image:")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Process the image and send to OpenAI Vision API
    if st.button("Analyze Image"):
        with BytesIO() as buffer:
            buffer.write(uploaded_file.getvalue())
            buffer.seek(0)
            image_data = buffer.read()

            response = get_openai_vision_response(image_data, question)
            st.write("Response from OpenAI Vision API:")
            st.write(response)

# if __name__ == "__main__":
#     st.run()
