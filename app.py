import streamlit as st
from utils.image_utils import upload_image
from utils.api_utils import analyze_image_with_tensorflow, generate_disposal_insight, get_nearby_disposal_locations
from utils.disposal_methods import determine_disposal

def main():
    st.title("Smart Recycling Assistant")
   
    st.subheader("Upload an Image")
    uploaded_image = upload_image()
   
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
       
        analysis_result = analyze_image_with_tensorflow(uploaded_image)
       
        if analysis_result:
            detected_object = analysis_result.split("(")[0].strip()
            st.write(f"Detected Object: {analysis_result}")
           
            insight = generate_disposal_insight(detected_object)
            st.write(f"Insight and Disposal Method: {insight}")
           
            disposal_method = determine_disposal(detected_object, insight)
            st.write(f"Recommended Disposal Method: {disposal_method}")
           
            # Get user's current location (you need to implement this part)
            # For demonstration purposes, use a fixed location
            latitude = 41.83353290852288
            longitude = -87.61154986850791
           
            # Find nearby disposal locations
            locations = get_nearby_disposal_locations(detected_object, latitude, longitude)
           
            if locations:
                st.write("Nearby Disposal Locations:")
                for location in locations:
                    st.write(f"- **{location['name']}**")
                    
                    # Check if location has latitude and longitude
                    lat = location['location'].get('latitude', '')
                    lng = location['location'].get('longitude', '')
                    
                    if lat and lng:
                        st.write(f"  Latitude: {lat}")
                        st.write(f"  Longitude: {lng}")
                        
                        # Add a Google Maps link for convenience
                        maps_url = f"https://www.google.com/maps?q={lat},{lng}"
                        st.markdown(f"  [View on Google Maps]({maps_url})")
            else:
                st.write("No nearby disposal locations found.")
        else:
            st.error("Could not analyze the image. Please try again.")

if __name__ == "__main__":
    main()