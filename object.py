import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import time

# Function to load a specific model
def load_model(model_name):
    try:
        if model_name == "YOLOv5":
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        elif model_name == "Faster R-CNN":
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            model = fasterrcnn_resnet50_fpn(pretrained=True)
            model.eval()  # Set model to evaluation mode
        elif model_name == "SSD":
            model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
        elif model_name == "EfficientDet":
            model = torch.hub.load('rwightman/efficientdet', 'efficientdet_d0')
        elif model_name == "RetinaNet":
            model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_retinanet')
        else:
            st.error("Model not supported")
            return None
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None
    return model

# Preprocessing function for each model
def preprocess_image(image, model_name):
    if model_name in ["YOLOv5", "Faster R-CNN", "SSD", "EfficientDet", "RetinaNet"]:
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    return None

# Perform detection and draw bounding boxes
def detect_and_draw(model, image, model_name):
    img = preprocess_image(image, model_name)
    if model_name == "YOLOv5":
        results = model(img)
        results.show()
        return results.render()[0]
    elif model_name == "Faster R-CNN":
        input_tensor = [torch.tensor(np.transpose(img, (2, 0, 1)) / 255.).float()]
        with torch.no_grad():
            predictions = model(input_tensor)
        for prediction in predictions:
            boxes = prediction['boxes'].cpu().numpy().astype(int)
            for box in boxes:
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        return img


# Streamlit user interface
st.title("Real-Time Object Detection")

# Sidebar for selecting the model
model_name = st.sidebar.selectbox(
    "Choose Object Detection Model",
    ["YOLOv5", "Faster R-CNN", "SSD", "EfficientDet", "RetinaNet"]
)

# Button to start detection
start_detection = st.button("Start Detection")

# Load the selected model
model = load_model(model_name)

# Start video stream using OpenCV
if start_detection and model is not None:
    cap = cv2.VideoCapture(0)  # Open webcam
    
    if not cap.isOpened():
        st.error("Could not open webcam.")
    else:
        stframe = st.empty()  # Streamlit placeholder for video feed
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break
            
            # Convert the frame to PIL image
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Perform detection and display result
            result_img = detect_and_draw(model, img, model_name)
            
            # Update the video stream frame in Streamlit
            stframe.image(result_img, channels="BGR", use_column_width=True)
            
            time.sleep(0.03)  # Add slight delay to avoid too fast refresh rate

    cap.release()
