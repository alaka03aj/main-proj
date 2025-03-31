import torch
import gradio as gr
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import random

# Set up device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the pre-trained model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# Function to process image and return crowd count (number of objects detected)
# Threshold for confidence scores is set to a default value of 0.8
def process_image(image, threshold=0.8):
    # Convert the image to a tensor
    image_tensor = F.to_tensor(image)
    
    # Run the model
    with torch.no_grad():
        output = model([image_tensor])[0]
    
    # Get bounding boxes, scores, and masks (we don't use masks here, just for bounding boxes)
    boxes = output['boxes']
    scores = output['scores']
    
    # Count how many objects have a score greater than the threshold
    count = sum(score >= threshold for score in scores)
    
    return count

# Gradio interface function
def gradio_interface(image):
    crowd_count = process_image(image)
    return f"Crowd count: {crowd_count}"

# Set up Gradio Interface with the new API
demo = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type="pil"),
    outputs="text",  # Changed to text since we want to return the count
    title="Object Detection for Crowd Count",
    description="Upload an image for crowd detection and get the count of objects.",
)

# Launch the Gradio app
demo.launch(share=True)