import streamlit as st
from utils.image_utils import upload_image
from utils.api_utils import analyze_image_with_tensorflow, generate_disposal_insight
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
        else:
            st.error("Could not analyze the image. Please try again.")

if __name__ == "__main__":
    main()
