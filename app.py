import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile

# Set the page configuration
st.set_page_config(
    page_title="Vehicle Number Plate Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

def detect_plate(image_path):
    try:
        # Detect objects in the image
        results = model(image_path)
        
        # Extract the first result (assuming only one detection)
        result = results[0]
        
        # Convert the tensor to a NumPy array and access the first bounding box
        bounding_boxes = result.boxes.xyxy.cpu().numpy()
        
        if len(bounding_boxes) == 0:
            st.warning("No number plate detected.")
            return None, None
        
        # For multiple detections, you can iterate through bounding_boxes
        # Here, we take the first detected plate
        x_min, y_min, x_max, y_max = bounding_boxes[0].astype(int)
        
        # Load the original image
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
        # Draw the bounding box on the image
        cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
        # Extract the cropped region from the original image
        cropped_image = original_image[y_min:y_max, x_min:x_max]
    
        return original_image, cropped_image
    except Exception as e:
        st.error(f"An error occurred during detection: {e}")
        return None, None

# Streamlit UI
st.title('🚗 Vehicle Number Plate Detection')
st.write("""
Upload an image of a vehicle, and the app will detect and highlight the number plate for you.
""")

# File uploader
uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            img = Image.open(uploaded_file).convert("RGB")
            img.save(tmp_file, format='PNG')
            uploaded_image_path = tmp_file.name
        
        # Display the uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(img, use_container_width=True)
        
        if st.button('🔍 Detect Number Plate'):
            with st.spinner('Detecting...'):
                original_image, cropped_image = detect_plate(uploaded_image_path)
            
            if original_image is not None and cropped_image is not None:
                with col2:
                    st.subheader("✅ Detected Number Plate")
                    st.image(original_image, use_container_width=True)
                
                with st.expander("📄 View Cropped Number Plate"):
                    cropped_pil = Image.fromarray(cropped_image)
                    st.image(cropped_pil, use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
