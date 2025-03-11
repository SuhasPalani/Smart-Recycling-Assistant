import tensorflow as tf
import numpy as np
from PIL import Image
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import pandas as pd
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Path to your OAuth 2.0 client secrets JSON file
client_secret_file = "client_secret2.json"

# API details
API_NAME = "places"
API_VERSION = "v1"
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Token file to store credentials
token_file = "token.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


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
        decoded_predictions = tf.keras.applications.inception_v3.decode_predictions(
            predictions, top=1
        )

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


def get_nearby_disposal_locations(detected_object, latitude, longitude):
    # Load credentials from file if they exist, otherwise run the OAuth flow
    credentials = None
    if os.path.exists(token_file):
        credentials = Credentials.from_authorized_user_file(token_file, SCOPES)

    # If there are no valid credentials available, prompt for authentication
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES)
            credentials = flow.run_local_server(port=0)

        # Save the credentials for future use
        with open(token_file, "w") as token:
            token.write(credentials.to_json())

    # Create the service object
    service = build(API_NAME, API_VERSION, credentials=credentials)

    # Map detected objects to valid Google Places API v1 place types
    disposal_types = {
        "plastic": ["hardware_store", "store"],
        "paper": ["hardware_store", "store"],
        "glass": ["hardware_store", "store"],
        "metal": ["hardware_store", "store"],
        "food": ["store"],
    }

    # Get the appropriate place types based on the detected object
    place_types = disposal_types.get(detected_object.lower(), ["store"])

    try:
        # Create a request to find nearby disposal locations
        request = service.places().searchNearby(
            body={
                "locationRestriction": {
                    "circle": {
                        "center": {"latitude": latitude, "longitude": longitude},
                        "radius": 5000.0,
                    }
                },
                "includedTypes": place_types,
                "maxResultCount": 5,
            },
            # Request additional fields specifically for display name and detailed location
            fields="places.id,places.displayName,places.location",
        )

        # Execute the request
        response = request.execute()

        # Print the raw response for debugging
        print("API Response:", response)

        # Extract and return the locations
        places_list = response.get("places", [])
        locations = []
        for place in places_list:
            # Get display name from the appropriate field
            name = place.get("displayName", {}).get("text", place.get("id", "Unknown"))

            # Get location coordinates
            location = place.get("location", {})

            locations.append({"name": name, "location": location})

        return locations

    except Exception as e:
        print(f"Error in get_nearby_disposal_locations: {e}")
        # Return empty list in case of error
        return []
