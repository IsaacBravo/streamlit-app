import streamlit as st
import os
import clip
import torch
from PIL import Image
from torchvision.datasets import CIFAR100
import matplotlib.pyplot as plt
import pandas as pd
from utils import icon
import tempfile

# Set the environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

def process_image(image_path):
    # Load the image
    image = Image.open(image_path)
    image_width = 450

    # Preprocess the image
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

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

    # Empty lists to store the results
    class_names = []
    percentages = []

    # Iterate over values and indices
    for value, index in zip(values, indices):
        # Get the class name and percentage
        class_name = cifar100.classes[index]
        percentage = 100 * value.item()

        # Append to the lists
        class_names.append(class_name)
        percentages.append(percentage)

    # Create the DataFrame
    df = pd.DataFrame({
        'Class Name': class_names,
        'Percentage': percentages
    })

    # Sort the DataFrame by Percentage
    df = df.sort_values(by='Percentage')

    return df


# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

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

# Prepare the inputs -- ADD THE PATH WHERE IS THE IMAGE
image = Image.open('climate_image.jpeg')
image_path = 'climate_image.jpeg'
image_width = 450

result_df = process_image(image_path)


grid_image, grid_space, grid_predictions = st.columns([3,3,3])

with grid_image:
    st.write("Original Image :camera:")
    st.image(image, caption='Uploaded Image', width=image_width)

with grid_predictions:
    st.write("Model Predictions :dart:")
    st.dataframe(result_df)

grid_image, grid_space, grid_predictions = st.columns([3,3,3])


st.write("Original Image :camera:")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image_user = Image.open(uploaded_file)
    # Create a temporary directory to save the uploaded file
    temp_dir = tempfile.mkdtemp()

    # Save the uploaded file to the temporary directory
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    with grid_image:
        st.image(image_user, caption='Uploaded Image', width=image_width)
    with grid_predictions:
        result = process_image(file_path)
        st.write("Model Predictions :dart:")
        st.dataframe(result)
else:
    st.write("Please upload an image.")

