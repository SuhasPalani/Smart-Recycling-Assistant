import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tensorflow.keras.applications import InceptionV3, ResNet50V2, EfficientNetB4
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess
import time


# Load the saved model
loaded_model = tf.keras.models.load_model('recycling_model.keras')

# Define class labels
class_labels = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Mixed', 'Paper', 'Plastic', 'Porcelain', 'Rubber', 'Textile', 'Vegetation']
# Silence TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Specific recycling categories and common items
RECYCLING_CATEGORIES = {
    "plastic": ["bottle", "container", "cup", "plastic", "packaging"],
    "paper": ["newspaper", "cardboard", "carton", "paper", "book", "magazine"],
    "glass": ["bottle", "jar", "glass"],
    "metal": ["can", "aluminum", "tin", "foil", "metal"],
    "electronic": ["phone", "computer", "laptop", "electronic", "device", "battery"],
    "organic": ["food", "fruit", "vegetable", "plant", "coffee", "tea"],
    "textile": ["clothing", "fabric", "textile", "cloth", "shoe"],
    "composite": ["tetra pak", "composite", "multi-material", "toy"]
}

def load_models():
    """
    Load multiple pre-trained models for ensemble prediction.
    """
    models = {
        "inception": {
            "model": InceptionV3(weights="imagenet"),
            "preprocess": inception_preprocess,
            "size": (299, 299)
        },
        "resnet": {
            "model": ResNet50V2(weights="imagenet"),
            "preprocess": resnet_preprocess,
            "size": (224, 224)
        },
        "efficientnet": {
            "model": EfficientNetB4(weights="imagenet"),
            "preprocess": efficient_preprocess,
            "size": (380, 380)
        }
    }
    return models

def analyze_image_with_tensorflow(image):
    try:
        start_time = time.time()
        
        # Preprocess the image
        img = load_img(image, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        
        # Get predictions
        predictions = loaded_model.predict(img_array)
        
        # Determine the recycling category
        predicted_class_index = np.argmax(predictions)
        recycling_type = class_labels[predicted_class_index]
        confidence = np.max(predictions)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        return {
            "recycling_type": recycling_type,
            "matched_labels": [recycling_type],  # Simplified for this example
            "confidence": confidence,
            "process_time": process_time
        }
    
    except Exception as e:
        print(f"Error with TensorFlow model: {e}")
        return None

def determine_recycling_category(labels, confidences):
    """
    Map predicted labels to recycling categories with weighted confidence.
    """
    category_scores = {category: 0.0 for category in RECYCLING_CATEGORIES}
    matched_labels = {category: [] for category in RECYCLING_CATEGORIES}
    
    # Calculate score for each category
    for label, confidence in zip(labels, confidences):
        for category, keywords in RECYCLING_CATEGORIES.items():
            for keyword in keywords:
                if keyword in label.lower():
                    category_scores[category] += confidence
                    if label not in matched_labels[category]:
                        matched_labels[category].append(label)
    
    # Get the category with the highest score
    if max(category_scores.values()) > 0:
        best_category = max(category_scores.items(), key=lambda x: x[1])
        return best_category[0], matched_labels[best_category[0]], best_category[1]
    else:
        # Return None if no category matches
        return None, [], 0.0

def generate_disposal_insight(detected_object, recycling_type, matched_labels):
    """
    Use OpenAI to generate better insights about the detected object based on recycling type and matched labels.
    """
    import openai
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai.api_key = OPENAI_API_KEY
    
    if not openai.api_key:
        return "API key not found. Please check your environment variables."
    
    # Create a more informative prompt
    prompt = f"""
    I've detected an object that appears to be: {detected_object}.
    The recycling category seems to be: {recycling_type}.
    The model also identified these related items: {', '.join(matched_labels)}.

    Please provide:
    1. A brief description of what materials this item is likely made from
    2. The correct disposal method (recycling, composting, e-waste, landfill, etc.)
    3. Any special instructions for preparing this item for disposal (cleaning, removing labels, etc.)
    
    Keep the response under 100 words.
    """

    try:
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=150,
            temperature=0.7,
        )

        insight = response.choices[0].text.strip()
        return insight

    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return "Unable to generate insight."

def get_nearby_disposal_locations(recycling_type, latitude, longitude):
    """
    Find appropriate disposal locations based on the recycling type using supported place types.
    """
    from googleapiclient.discovery import build
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    import os
    
    # Path to your OAuth 2.0 client secrets JSON file
    client_secret_file = "client_secret2.json"
    
    # API details
    API_NAME = "places"
    API_VERSION = "v1"
    SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
    
    # Token file to store credentials
    token_file = "token.json"
    
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
    
    # Map recycling types to supported Google Places API v1 place types
    # Using verified supported types from the Places API documentation
    disposal_types = {
        "plastic": ["store", "hardware_store", "supermarket"],
        "paper": ["store", "supermarket", "library"],
        "glass": ["hardware_store", "store", "supermarket"],
        "metal": ["hardware_store", "store"],
        "electronic": ["electronics_store", "store"],
        "organic": ["grocery_store", "supermarket", "store"],
        "textile": ["clothing_store", "department_store", "store"],
        "composite": ["hardware_store", "store"],
        "unknown": ["hardware_store", "store", "supermarket"]
    }
    
    # Get the appropriate place types based on the recycling type
    place_types = disposal_types.get(recycling_type.lower(), ["store"])
    
    # Let's also add a query parameter to help find recycling locations
    query_keywords = {
        "plastic": "recycling plastic",
        "paper": "recycling paper",
        "glass": "recycling glass",
        "metal": "recycling metal",
        "electronic": "electronics recycling",
        "organic": "compost",
        "textile": "clothing donation",
        "composite": "recycling",
        "unknown": "recycling"
    }
    
    query = query_keywords.get(recycling_type.lower(), "recycling")
    
    try:
        # First attempt: search by included types
        request = service.places().searchNearby(
            body={
                "locationRestriction": {
                    "circle": {
                        "center": {"latitude": latitude, "longitude": longitude},
                        "radius": 5000.0,
                    }
                },
                "includedTypes": place_types[:1],  # Use just the first type to avoid errors
                "maxResultCount": 5,
            },
            fields="places.id,places.displayName,places.location,places.formattedAddress"
        )
        
        response = request.execute()
        places_list = response.get("places", [])
        
        # If we didn't find enough places, try a text search with the keyword
        if len(places_list) < 3:
            text_request = service.places().searchText(
                body={
                    "textQuery": f"{query} near me",
                    "locationBias": {
                        "circle": {
                            "center": {"latitude": latitude, "longitude": longitude},
                            "radius": 5000.0,
                        }
                    },
                    "maxResultCount": 5,
                },
                fields="places.id,places.displayName,places.location,places.formattedAddress"
            )
            
            text_response = text_request.execute()
            # Add these results to our list
            places_list.extend(text_response.get("places", []))
        
        # Extract and return the locations
        locations = []
        seen_ids = set()  # To avoid duplicates
        
        for place in places_list:
            place_id = place.get("id")
            
            # Skip if we've seen this place before
            if place_id in seen_ids:
                continue
                
            seen_ids.add(place_id)
            
            # Get display name
            name = place.get("displayName", {}).get("text", place.get("id", "Unknown"))
    
            # Get location coordinates
            location = place.get("location", {})
            
            # Get formatted address if available
            address = place.get("formattedAddress", "Address unavailable")
    
            locations.append({
                "name": name, 
                "location": location,
                "address": address
            })
        
        # Sort by name for consistency
        locations.sort(key=lambda x: x["name"])
        
        return locations[:5]  # Return at most 5 results
    
    except Exception as e:
        print(f"Error in get_nearby_disposal_locations: {e}")
        # Try fallback to a simple text search
        try:
            # FIXED: Use the correct structure for fallback search
            text_request = service.places().searchText(
                body={
                    "textQuery": f"recycling near me",
                    "locationBias": {
                        "circle": {
                            "center": {"latitude": latitude, "longitude": longitude},
                            "radius": 5000.0,
                        }
                    },
                    "maxResultCount": 5,
                },
                fields="places.id,places.displayName,places.location,places.formattedAddress"
            )
            
            text_response = text_request.execute()
            places_list = text_response.get("places", [])
            
            locations = []
            for place in places_list:
                name = place.get("displayName", {}).get("text", place.get("id", "Unknown"))
                location = place.get("location", {})
                address = place.get("formattedAddress", "Address unavailable")
                
                locations.append({
                    "name": name, 
                    "location": location,
                    "address": address
                })
                
            return locations
            
        except Exception as fallback_error:
            print(f"Error in fallback search: {fallback_error}")
            return []