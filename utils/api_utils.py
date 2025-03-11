import tensorflow as tf
import numpy as np
from PIL import Image

def analyze_image_with_tensorflow(image):
    """
    Analyze the uploaded image using TensorFlow's InceptionV3 model.
    """
    try:
        # Load the pre-trained InceptionV3 model + weights
        model = tf.keras.applications.InceptionV3(weights="imagenet")
        
        # Preprocess the image for the model
        img = Image.open(image).resize((299, 299))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
        
        # Predict the object in the image
        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.inception_v3.decode_predictions(predictions, top=1)
        
        # Get the most likely label
        top_prediction = decoded_predictions[0][0]
        label = top_prediction[1]  # Human-readable label
        confidence = top_prediction[2]  # Confidence score
        
        return f"{label} ({confidence * 100:.2f}% confidence)"
    
    except Exception as e:
        print(f"Error with TensorFlow model: {e}")
        return None

def generate_disposal_insight(detected_object):
    """
    Use OpenAI to generate insights about the detected object and recommend a disposal method.
    """
    import openai
    from dotenv import load_dotenv
    import os
    
    # Load environment variables from .env file
    load_dotenv()
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai.api_key = OPENAI_API_KEY
    
    prompt = f"Describe the material and recommended disposal method for a {detected_object}."
    
    try:
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=100,
            temperature=0.7,
        )
        
        insight = response.choices[0].text.strip()
        return insight
    
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return "Unable to generate insight."
