import streamlit as st
import os
import clip
import torch
from torchvision.datasets import CIFAR100
from PIL import Image
from utils import icon


# Set the environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# UI configurations
st.set_page_config(page_title="Replicate Image Generator",
                   page_icon=":desktop_computer:",
                   #layout="wide"
                   
                   )
icon.show_icon(":desktop_computer:")
st.markdown("# :blue[Image Analysis]")

# Placeholders for images and gallery
generated_images_placeholder = st.empty()
gallery_placeholder = st.empty()

# Function to configure sidebar and get user input
def configure_sidebar() -> (bool, int, int, Image.Image):
    """
    Setup and display the sidebar elements.
    """
    with st.sidebar:
        with st.form("my_form"):
            st.info("**Welcome! Start here ↓**", icon="👋")
            with st.expander(":blue[**Refine your output here**]"):
                # Advanced Settings
                uploaded_file = st.file_uploader(
                    "Upload an image", type=["jpg", "png", "jpeg"])
                image_width = st.number_input(
                    "Width of output image", value=350)
                image_height = st.number_input(
                    "Height of output image", value=350)

            # Submit Button
            submitted = st.form_submit_button(
                "Upload Image", type="primary", use_container_width=True)

    return submitted, image_width, image_height, uploaded_file

# Function to configure prompt and get user input
def configure_prompt() -> (str, bool):
    """
    Setup and display the prompt elements.
    """
    with st.form("my_form_analyse"):
        prompt = st.text_area(
            ":blue[**Enter prompt: ✍️**]",
            value="For example: A photo of a rainy day in New York City")
        # Submit Button
        submitted_analyse = st.form_submit_button(
            "Analyse Image", type="primary", use_container_width=True)

    return prompt, submitted_analyse

# Configure the sidebar and get user input
submitted, image_width, image_height, uploaded_file = configure_sidebar()

# Initialize image variable
image = None

# Check if the form is submitted and an image is uploaded
if submitted and uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    st.write("Original Image :camera:")
    st.image(image, caption='Uploaded Image', width=image_width)

# Show the uploaded image placeholder
uploaded_image_placeholder = st.empty()

# Configure the prompt and get user input
prompt, submitted_analyse = configure_prompt()

# Check if the form is submitted and an image is uploaded
if submitted_analyse:
    # Check if the image is uploaded
    if image is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load('ViT-B/32', device)

        # Download the dataset
        cifar100 = CIFAR100(
            root=os.path.expanduser("~/.cache"), download=True, train=False)

        # Prepare the inputs
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = torch.cat(
            [clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

        # Perform image analysis here
        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        values, indices = similarity[0].topk(5)
        top_probs, top_labels = similarity.topk(5, dim=-1)

        # Print the result
        print("\nTop predictions:\n")
        for value, index in zip(values, indices):
            print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")