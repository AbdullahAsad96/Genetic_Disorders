import streamlit as st
from transformers import pipeline
import torch

# Set the model and device
model_name_or_path = "m42-health/Llama3-Med42-8B"  # Update to your model

# Initialize the pipeline for text generation
pipe = pipeline(
    "text-generation",
    model=model_name_or_path,
    torch_dtype=torch.bfloat16,
    device=0,  # Use GPU if available
)

# Function to get the model's response
def get_response(user_input):
    messages = [
        {"role": "system", "content": "You are a medical assistant."},
        {"role": "user", "content": user_input},
    ]
    
    # Generate response from the model
    response = pipe(messages)
    return response[0]["generated_text"]

# Streamlit UI
st.title("Medical Chatbot")
st.write("Ask a question about genetic disorders or any related health query.")

user_input = st.text_input("Your Question:")

if user_input:
    response = get_response(user_input)
    st.write(f"Answer: {response}")
