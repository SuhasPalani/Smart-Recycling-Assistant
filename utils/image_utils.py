import streamlit as st
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the saved model
loaded_model = tf.keras.models.load_model("recycling_model.keras")

# Define class labels
class_labels = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Mixed",
    "Paper",
    "Plastic",
    "Porcelain",
    "Rubber",
    "Textile",
    "Vegetation",
]


def upload_image():
    """
    Handle image upload with proper validation and preprocessing.

    Returns:
        PIL.Image or None: The uploaded image if successful, None otherwise.
    """
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of the item you want to recycle",
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
            if image.mode == "RGBA":
                image = image.convert("RGB")

            # Create a byte buffer for the processed image
            buf = io.BytesIO()
            image.save(buf, format="JPEG")

            # Display the uploaded image
            st.image(image, caption="Uploaded Image")

            # Classify the image using the trained model
            img_array = load_img(io.BytesIO(image_bytes), target_size=(224, 224))
            img_array = img_to_array(img_array)
            img_array = np.expand_dims(img_array, axis=0)  # Convert to batch format
            img_array = img_array / 255.0  # Normalize

            predictions = loaded_model.predict(img_array)
            predicted_class_index = np.argmax(
                predictions
            )  # Get index of highest probability
            detected_class = class_labels[predicted_class_index]  # Map to class name

            # Display the detected class
            st.write(f"Detected Class: **{detected_class}**")

            # Provide disposal guidance based on the detected class
            # Generate disposal guidance
            disposal_guidelines = {
                "Cardboard": "Recycle it in a paper recycling bin.",
                "Food Organics": "Compost it if organic and uncooked, otherwise dispose of it in landfill waste.",
                "Glass": "Recycle it in a glass recycling bin.",
                "Metal": "Recycle it in a metal recycling bin.",
                "Miscellaneous Trash": "Dispose of it in landfill waste.",
                "Mixed": "Sort and recycle components if possible.",
                "Paper": "Recycle it in a paper recycling bin.",
                "Plastic": "Recycle it in a plastic recycling bin if accepted by local facilities.",
                "Porcelain": "Donate or dispose of it in landfill waste.",
                "Rubber": "Donate or recycle if possible.",
                "Textile": "Donate or recycle if possible.",
                "Vegetation": "Compost it.",
            }

            # Display disposal guidance
            st.subheader("Disposal Guidance")
            st.write(
                disposal_guidelines.get(
                    st.session_state.recycling_type, "Unknown disposal method."
                )
            )

            return io.BytesIO(buf.getvalue())

        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None

    return None
