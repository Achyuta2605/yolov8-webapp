#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import pandas as pd
from ultralytics import YOLO
import tempfile

# Configure Streamlit page
st.set_page_config(
    page_title="YOLOv8 Rice Disease Segmentation with DSI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for responsive design
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        color: #2E8B57;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .dsi-box {
        background-color: #f0f8ff;
        border: 2px solid #4682b4;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    @media (max-width: 768px) {
        .stButton > button {
            width: 100%;
        }
        .stSelectbox > div {
            width: 100%;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_yolo_model(model_path):
    """Load YOLO model exactly like in your test script"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def calculate_dsi_like_test_script(image, boxes, class_names):
    """Calculate DSI exactly like your test script"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    img_h, img_w = image.shape[:2]
    total_image_area = img_h * img_w
    diseased_area = 0

    # Calculate area of non-healthy detections (like your test script)
    for box, cls in zip(boxes.xyxy, boxes.cls):
        class_name = class_names[int(cls)]
        if class_name.lower() != 'healthy':
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            diseased_area += area

    # Compute DSI (exactly like your test script)
    dsi = (diseased_area / total_image_area) * 100 if total_image_area > 0 else 0
    return round(dsi, 2)

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 YOLOv8 Rice Disease Segmentation with DSI</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model path (your specific .pt model path)
    model_path = r"C:\Major Project\MajorProjectWebApp\yolov8-streamlitapp\best2.pt"
    
    # Confidence threshold (exactly like your test script)
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.01, 
        max_value=1.0, 
        value=0.05,  # Same as your test script default
        step=0.01
    )
    
    # Check if model file exists
    if os.path.exists(model_path):
        # Load model exactly like your test script
        model = load_yolo_model(model_path)
        
        if model is not None:
            st.success("✅ YOLOv8 Segmentation Model loaded successfully!")
            
            # Get class names exactly like your test script
            class_names = model.names
            
            # Display model info
            st.sidebar.subheader("📋 Model Information")
            st.sidebar.write(f"**Model Type:** YOLOv8 Segmentation")
            st.sidebar.write(f"**Classes:** {len(class_names)}")
            st.sidebar.write(f"**Class Names:** {list(class_names.values())}")
            
            # Image upload section
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "📸 Choose a rice leaf image from your gallery",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                help="Upload an image for disease segmentation and DSI calculation"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            if uploaded_file is not None:
                # Load and display original image
                image = Image.open(uploaded_file)
                
                # Save uploaded image temporarily (like your test script uses img_path)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image.save(tmp_file.name)
                    temp_img_path = tmp_file.name
                
                # Create responsive columns
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📷 Original Image")
                    st.image(image, caption="Uploaded Image")
                
                # Run inference EXACTLY like your test script
                with st.spinner("🔍 Running segmentation and calculating DSI..."):
                    try:
                        # Use predict method with image path (exactly like your test script)
                        results = model.predict(temp_img_path, conf=conf_threshold)
                        result = results[0]
                        boxes = result.boxes
                        
                        # Get plotted image (exactly like your test script)
                        plotted_img = result.plot(line_width=2)
                        
                        # Calculate DSI exactly like your test script
                        dsi_value = calculate_dsi_like_test_script(image, boxes, class_names)
                        
                        # Annotate DSI on image (exactly like your test script)
                        annotated_img = plotted_img.copy()
                        cv2.putText(annotated_img,
                                    f"DSI: {dsi_value:.2f}%",
                                    (10, 15),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.3, (0, 0, 255), 1, cv2.LINE_AA)
                        
                        # Clean up temp file
                        os.unlink(temp_img_path)
                        
                    except Exception as e:
                        st.error(f"❌ Inference failed: {str(e)}")
                        if os.path.exists(temp_img_path):
                            os.unlink(temp_img_path)
                        st.stop()
                
                # Display results
                if boxes is not None and len(boxes) > 0:
                    with col2:
                        st.subheader("🎯 Segmentation Results")
                        st.image(annotated_img, caption="Disease Segmentation with DSI")
                    
                    # DSI Display Box
                    st.markdown(f"""
                    <div class="dsi-box">
                        <h3>🌾 Disease Severity Index (DSI)</h3>
                        <h2 style="color: {'red' if dsi_value > 20 else 'orange' if dsi_value > 10 else 'green'};">
                            {dsi_value:.2f}%
                        </h2>
                        <p>Severity Level: {'High' if dsi_value > 20 else 'Medium' if dsi_value > 10 else 'Low'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display detection statistics
                    st.subheader("📊 Detection Summary")
                    detection_data = []
                    diseased_count = 0
                    healthy_count = 0
                    
                    for box, cls in zip(boxes.xyxy, boxes.cls):
                        class_name = class_names[int(cls)]
                        conf = boxes.conf[list(boxes.cls).index(cls)]
                        
                        detection_data.append({
                            "Object": class_name,
                            "Confidence": f"{conf:.2%}",
                            "Bounding Box": f"({box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f})"
                        })
                        
                        if class_name.lower() == 'healthy':
                            healthy_count += 1
                        else:
                            diseased_count += 1
                    
                    st.dataframe(detection_data)
                    
                    # Summary metrics
                    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                    with col_metrics1:
                        st.metric("Total Detections", len(boxes))
                    with col_metrics2:
                        st.metric("Diseased Areas", diseased_count)
                    with col_metrics3:
                        st.metric("Healthy Areas", healthy_count)
                    
                else:
                    with col2:
                        st.subheader("🎯 Segmentation Results")
                        st.image(image, caption="No Diseases Detected")
                    
                    # DSI Display Box for no detections
                    st.markdown(f"""
                    <div class="dsi-box">
                        <h3>🌾 Disease Severity Index (DSI)</h3>
                        <h2 style="color: green;">0.00%</h2>
                        <p>Severity Level: Healthy</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info(f"✅ No diseases detected with confidence ≥ {conf_threshold:.0%}")
        
        else:
            st.error("❌ Failed to load the model. Please check the model file.")
    
    else:
        st.error(f"❌ Model file not found at: {model_path}")

if __name__ == "__main__":
    main()
