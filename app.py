import streamlit as st
from PIL import Image
import io
import pandas as pd
from utils.image_utils import upload_image
from utils.api_utils import (
    analyze_image_with_tensorflow,
    generate_disposal_insight,
    get_nearby_disposal_locations,
)

def main():
    st.title("Smart Recycling Assistant")
    st.write("Upload an image of an item to get recycling recommendations and nearby disposal locations.")

    # Initialize session state variables if they don't exist
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'recycling_type' not in st.session_state:
        st.session_state.recycling_type = None
    if 'matched_labels' not in st.session_state:
        st.session_state.matched_labels = []
    if 'confidence' not in st.session_state:
        st.session_state.confidence = 0
    if 'insight' not in st.session_state:
        st.session_state.insight = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None

    # Image upload section
    st.subheader("Upload an Image")
    uploaded_image = upload_image()

    # Button to trigger analysis
    if uploaded_image and not st.session_state.analysis_complete:
        st.session_state.uploaded_image = uploaded_image
        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                analysis_result = analyze_image_with_tensorflow(uploaded_image)
                
                if analysis_result:
                    # Store results in session state
                    st.session_state.analysis_complete = True
                    st.session_state.recycling_type = analysis_result["recycling_type"]
                    st.session_state.matched_labels = analysis_result["matched_labels"]
                    st.session_state.confidence = analysis_result["confidence"]
                    st.session_state.process_time = analysis_result["process_time"]
                    
                    # Generate insight immediately
                    with st.spinner("Generating disposal guidance..."):
                        st.session_state.insight = generate_disposal_insight(
                            st.session_state.recycling_type, 
                            st.session_state.recycling_type,
                            st.session_state.matched_labels
                        )
                    
                    # Force a rerun to update the UI with the analysis results
                    st.rerun()
                else:
                    st.error("Could not analyze the image. Please try again with a clearer image.")

    # Reset button (visible only after analysis is complete)
    if st.session_state.analysis_complete:
        if st.button("Reset Analysis"):
            # Reset session state
            st.session_state.analysis_complete = False
            st.session_state.recycling_type = None
            st.session_state.matched_labels = []
            st.session_state.confidence = 0
            st.session_state.insight = None
            st.session_state.uploaded_image = None
            # Force a rerun to update the UI
            st.rerun()

    # Display results after analysis is complete
    if st.session_state.analysis_complete and st.session_state.uploaded_image:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(st.session_state.uploaded_image, caption="Uploaded Item", use_container_width=True)
        
        with col2:
            # Display analysis result
            st.success(f"**Analysis complete in {st.session_state.process_time:.2f} seconds**")
            
            # Create a styled box for the detected object
            st.markdown(f"""
            <div style="padding:10px; border-radius:5px; background-color:#f0f7fb; border-left:5px solid #2196F3;">
                <h3 style="margin:0;">Detected: {st.session_state.recycling_type.upper()}</h3>
                <p>Matched items: {', '.join(st.session_state.matched_labels)}</p>
                <p>Confidence: {st.session_state.confidence * 100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display insight
        st.subheader("Disposal Guidance")
        st.write(st.session_state.insight)
        
        # Location finder section
        st.subheader("Find Nearby Disposal Locations")
        
        # Option to enter custom location
        use_custom_location = st.checkbox("Enter a custom location")
        
        if use_custom_location:
            col1, col2 = st.columns(2)
            with col1:
                latitude = st.number_input("Latitude", value=41.83353290852288)
            with col2:
                longitude = st.number_input("Longitude", value=-87.61154986850791)
        else:
            # For demonstration purposes, use a fixed location
            latitude = 41.83353290852288
            longitude = -87.61154986850791
            st.info("Using default location (Chicago, IL). Check the box above to enter a custom location.")
        
        # Get nearby disposal locations when user clicks the button
        if st.button("Find Nearby Disposal Locations"):
            with st.spinner("Finding locations..."):
                locations = get_nearby_disposal_locations(st.session_state.recycling_type, latitude, longitude)
            
            if locations:
                st.success(f"Found {len(locations)} nearby locations for {st.session_state.recycling_type} disposal")
                
                # Display locations in a nicer format
                for i, location in enumerate(locations, 1):
                    with st.expander(f"{i}. {location['name']}"):
                        # Check if location has latitude and longitude
                        lat = location["location"].get("latitude", "")
                        lng = location["location"].get("longitude", "")
                        address = location.get("address", "Address unavailable")
                        
                        st.write(f"**Address:** {address}")
                        
                        if lat and lng:
                            # Add a Google Maps link
                            maps_url = f"https://www.google.com/maps?q={lat},{lng}"
                            st.markdown(f"[View on Google Maps]({maps_url})")
                            
                            # Display mini map if available
                            map_data = pd.DataFrame({
                                'lat': [lat],
                                'lon': [lng]
                            })
                            st.map(map_data)
            else:
                st.warning("No nearby disposal locations found. Try expanding your search area.")

    # Add some helpful information
    with st.expander("How to use this app"):
        st.write("""
        1. Upload a photo of the item you want to recycle
        2. Click "Analyze Image" to identify the item (this happens only once)
        3. You'll receive guidance on how to properly dispose of the item
        4. Click "Find Nearby Locations" to see where you can take the item for disposal
        5. Use "Reset Analysis" to start over with a new image
        """)
        
    with st.expander("Supported recycling categories"):
        categories = {
            "Plastic": "Bottles, containers, packaging",
            "Paper": "Newspaper, cardboard, cartons",
            "Glass": "Bottles, jars",
            "Metal": "Cans, aluminum, tin foil",
            "Electronic": "Phones, computers, batteries",
            "Organic": "Food waste, plant material",
            "Textile": "Clothing, fabric, shoes",
            "Composite": "Multi-material items"
        }
        
        for category, examples in categories.items():
            st.write(f"**{category}**: {examples}")


if __name__ == "__main__":
    main()