import streamlit as st
from PIL import Image
import io

def upload_image():
    """
    Handle image upload with proper validation and preprocessing.
    
    Returns:
        PIL.Image or None: The uploaded image if successful, None otherwise.
    """
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of the item you want to recycle"
    )
    
    if uploaded_file is not None:
        try:
            # Read the file and convert to PIL Image
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Check if image is valid
            image.verify()  # Verify that it's an image
            
            # Reopen the image after verify (which closes it)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert RGBA to RGB if needed
            if image.mode == 'RGBA':
                image = image.convert('RGB')
                
            # Create a byte buffer for the processed image
            buf = io.BytesIO()
            image.save(buf, format='JPEG')
            
            return io.BytesIO(buf.getvalue())
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None
    
    return None