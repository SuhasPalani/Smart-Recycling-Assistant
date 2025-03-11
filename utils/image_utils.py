import streamlit as st

def upload_image():
    """
    Allow users to upload an image file.
    """
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    return uploaded_file
