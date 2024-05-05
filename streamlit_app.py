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
# from streamlit_option_menu import option_menu

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
    df = df.sort_values(by='Percentage', ascending=False)

    return df

def process_image_labels(image_path, labels):
    # Load the image
    image = Image.open(image_path)
    image_width = 450

    # Preprocess the image
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in labels]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    values, indices = similarity[0].topk(len(labels))
    top_probs, top_labels = similarity.topk(len(labels), dim=-1)

    # Print the result
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{labels[index]:>16s}: {100 * value.item():.2f}%")

    # Empty lists to store the results
    class_names = []
    percentages = []

    # Iterate over values and indices
    for value, index in zip(values, indices):
        # Get the class name and percentage
        class_name = labels[index]
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
    df = df.sort_values(by='Percentage', ascending=False)

    return df

def process_image_labels_binary(image_path, labels):
    # Load the image
    image = Image.open(image_path)
    image_width = 450

    # Preprocess the image
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in labels]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    values, indices = similarity[0].topk(len(labels))
    top_probs, top_labels = similarity.topk(len(labels), dim=-1)

    # Print the result
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{labels[index]:>16s}: {100 * value.item():.2f}%")

    # Empty lists to store the results
    class_names = []
    percentages = []

    # Iterate over values and indices
    for value, index in zip(values, indices):
        # Get the class name and percentage
        class_name = labels[index]
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
    df = df.sort_values(by='Percentage', ascending=False)

    return df

############################################################################################################
# Load the CIFAR-100 dataset
############################################################################################################

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)


############################################################################################################
# UI configurations
############################################################################################################

st.set_page_config(page_title="Home - Clip Model Prototype",
                   page_icon=":desktop_computer:",
                   initial_sidebar_state="auto",
                   layout="wide"             
                   )

############################################################################################################
# UI sidebar - Menu
############################################################################################################

with st.sidebar:
     st.title(":blue[Image Classification Prototype: Clip Model]")
     st.divider()
     st.markdown('<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.2;"><strong>Contact us: <br></br>Isaac Bravo:</strong> <a href="https://www.linkedin.com/in/isaac-bravo/"><img src="https://openvisualfx.com/wp-content/uploads/2019/10/linkedin-icon-logo-png-transparent.png" width="20" height="20"></a><a href="https://github.com/IsaacBravo"><img src="https://www.pngarts.com/files/8/Github-Logo-Transparent-Background-PNG-420x236.png" width="35" height="20"></a><br></br><strong>Katharina Prasse:</strong> <a href="https://www.linkedin.com/in/katharina-prasse/"><img src="https://openvisualfx.com/wp-content/uploads/2019/10/linkedin-icon-logo-png-transparent.png" width="20" height="20"></a><a href="https://github.com/KathPra"><img src="https://www.pngarts.com/files/8/Github-Logo-Transparent-Background-PNG-420x236.png" width="35" height="20"></a></p>', 
                 unsafe_allow_html=True)
     st.divider()
     sidebar_title = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 15px; letter-spacing: -0.005em; line-height: 1.5;">This page is part of the ClimateVision project, academic initiative between the Technical University of Munich and the University of Mannheim. This project is founding by the Bundesministerium für Bildung und Forschung and the European Union. If you want to know more about the project, please check our website <a href="https://web.informatik.uni-mannheim.de/climatevisions/">here.</a></p>'
     st.markdown(sidebar_title, unsafe_allow_html=True)


icon.show_icon(":desktop_computer:")

st.header(":blue[Welcome to ClimateVision Project! 👋]")
original_header = '<p style="font-family:Source Sans Pro; text-align:justify; color:#1F66CB; font-size: 17px; letter-spacing: -0.005em; line-height: 1.5; background-color:#EBF2FC; padding:25px; border-radius:10px; border:1px solid graylight;">This page provides users with the ability to upload an image and receive predictions from the CLIP model developed by OpenAI. The CLIP model, short for "Contrastive Language-Image Pre-training," is a powerful artificial intelligence model capable of understanding both images and text. Using this model, the application predicts the class or content depicted in the uploaded image based on its visual features and any accompanying text description. By leveraging the CLIP model`s unique ability to analyze images in conjunction with text, users can gain insights into what the model perceives from both modalities, offering a richer understanding of the image content.</p>'
st.markdown(original_header, unsafe_allow_html=True)


#st.info("""
#This page allows the user to upload an image and get predictions from the model. 
#The model is based on the CLIP model from OpenAI, which predicts the class of the image based on its text description. 
#""")
st.markdown('<br></br>', unsafe_allow_html=True)

original_title = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 25px; font-weight: 600; letter-spacing: -0.005em; line-height: 1.2;">Model example 1: A photo of a climate change</p>'
st.markdown(original_title, unsafe_allow_html=True)

original_title_text_1 = '<p style="font-family:Source Sans Pro; color:black; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">The following example provide a glance on how CLIP Model works to label a climate change image, based on its trained data.</p>'
st.markdown(original_title_text_1, unsafe_allow_html=True)
st.divider()

# Placeholders for images and gallery
generated_images_placeholder = st.empty()
gallery_placeholder = st.empty()

############################################################################################################
# style
############################################################################################################
th_props = [
  ('font-size', '20px'),
  ('font-weight', 'bold'),
  ('color', '#fff'),
  ('text-align', 'center'),
  ('text-shadow', '0 1px 0 #000'),
  ('background-color', 'blue'),
  ('padding', '5px 10px'),
  ('box-shadow', '0 0 20px rgba(0, 0, 0, 0.15)')
  ]
                               
td_props = [
  ('font-size', '17px'),
  ('text-align', 'center')
  ]

table_props = [
  ('border', '1px solid #6d6d6d'),
  ('border-radius', '30px'),
  ('overflow', 'hidden')
  ]
                                 
styles_dict = [
  dict(selector="th", props=th_props),
  dict(selector="td", props=td_props),
  dict(selector="table", props=table_props)
  ]

############################################################################################################
# Prepare the inputs -- EXAMPLE 1
############################################################################################################

image = Image.open('climate_image.jpeg')
image_path = 'climate_image.jpeg'
image_width = 450

result_df = process_image(image_path)

# UI - Original image and predictions (pre-loaded image)
grid_image, grid_predictions = st.columns([3,3])

with grid_image:
    example_text_1 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Original Image &#128247;</p>'
    st.markdown(example_text_1, unsafe_allow_html=True)
    st.image(image, caption='Pre-loaded Image', width=image_width)

with grid_predictions:
    example_text_2 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Model Predictions &#127919;</p>'
    st.markdown(example_text_2, unsafe_allow_html=True)
    # st.table(result_df.style.set_properties(**{'border-radius': '30px'}).set_table_styles(styles_dict))
    st.markdown("""
        <style>
        table {border-radius: 60px;}
        </style>
        """, unsafe_allow_html=True)
    st.dataframe(result_df.style.background_gradient(cmap='Blues'))

st.divider()

############################################################################################################
# UI - User input and predictions # USER EXAMPLE 1
############################################################################################################

user_example_text_1 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Upload your own Image! &#128247;</p>'
st.markdown(user_example_text_1, unsafe_allow_html=True)
user_example_text_2 = '<p style="font-family:Source Sans Pro; color:black; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Now you can test the model using your own image data, and see how the model detect the different elements on your image.</p>'
st.markdown(user_example_text_2, unsafe_allow_html=True)

uploaded_file_example_1  = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="1")
st.warning("""
Please consider that analizing an image may take a few seconds.
""")

grid_image, grid_predictions = st.columns([3,3])

if uploaded_file_example_1 is not None:
    image_user = Image.open(uploaded_file_example_1 )
    # Create a temporary directory to save the uploaded file
    temp_dir = tempfile.mkdtemp()

    # Save the uploaded file to the temporary directory
    file_path = os.path.join(temp_dir, uploaded_file_example_1 .name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file_example_1 .getvalue())

    with grid_image:
        example_text_1 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Original Image &#128247;</p>'
        st.markdown(example_text_1, unsafe_allow_html=True)
        st.image(image_user, caption='Uploaded Image', width=image_width)
    with grid_predictions:
        result = process_image(file_path)
        example_text_2 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Model Predictions &#127919;</p>'
        st.markdown(example_text_2, unsafe_allow_html=True)
        st.dataframe(result.style.background_gradient(cmap='Blues'))
    st.divider()
else:
    st.write("Please upload an image. :point_up:")

st.divider()

############################################################################################################
# Prepare the inputs -- EXAMPLE 2
############################################################################################################

image_2 = Image.open('climate_image_2.jpeg')
image_path_2 = 'climate_image_2.jpeg'

original_title_2 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 25px; font-weight: 600; letter-spacing: -0.005em; line-height: 1.2;">Model example 2: A photo of a climate change (Defining input labels)</p>'
st.markdown(original_title_2, unsafe_allow_html=True)

original_title_text_2 = '<p style="font-family:Source Sans Pro; color:black; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">The following example provide a glance on how CLIP Model works to label a climate change image, based on labels defined by the researcher.</p>'
st.markdown(original_title_text_2, unsafe_allow_html=True)
st.divider()

# UI - Original image and predictions (pre-loaded image)
grid_image, grid_space, grid_predictions_1, grid_predictions_2 = st.columns([3,1,3,3])

result_df_labels_1 = process_image_labels(image_path_2, labels=['wildfires', 'drought', 'pollution', 'deforestation', 'flood'])
result_df_labels_2 = process_image_labels_binary(image_path_2, labels=['Yes', 'No'])

with grid_image:
    example_text_1 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Original Image &#128247;</p>'
    st.markdown(example_text_1, unsafe_allow_html=True)
    st.image(image_2, caption='Pre-loaded Image', width=image_width)

with grid_predictions_1:
    example_text_2 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Model Predictions (Multi label) &#127919;</p>'
    st.markdown(example_text_2, unsafe_allow_html=True)
    st.dataframe(result_df_labels_1.style.background_gradient(cmap='Blues'))
    st.info("""For this example we defined the following labels: wildfires, drought, pollution, deforestation, and flood.""")

with grid_predictions_2:
    example_text_2 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Model Predictions (One label) &#127919;</p>'
    st.markdown(example_text_2, unsafe_allow_html=True)
    st.dataframe(result_df_labels_2.style.background_gradient(cmap='Blues'))
    st.info("""For this example we ask the following question: Does the image represent a flood?""")
st.divider()


############################################################################################################
# UI - User input and predictions # USER EXAMPLE 2
############################################################################################################

# UI - User input and predictions
user_example_text_1 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Upload your own Image! &#128247;</p>'
st.markdown(user_example_text_1, unsafe_allow_html=True)
user_example_text_2 = '<p style="font-family:Source Sans Pro; color:black; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Now you can test the model using your own image data, and see how the model detect the different elements on your image.</p>'
st.markdown(user_example_text_2, unsafe_allow_html=True)

uploaded_file_example_2 = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="2")
st.warning("""
Please consider that analizing an image may take a few seconds.
""")
st.write("Please upload an image. :point_up:")

grid_text_1, grid_text_2 = st.columns([3,3])
with grid_text_1:
    user_example_text_3 = '<p style="font-family:Source Sans Pro; color:black; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Now you can define either one or a set of labels to classify your image: &#128073;</p>'
    st.markdown(user_example_text_3, unsafe_allow_html=True)
with grid_text_2:
    labels_user = st.text_input('Enter one or multiple labels (separated by comma).')

grid_image, grid_predictions = st.columns([3,3])

if uploaded_file_example_2 is not None:
    image_user_example_2 = Image.open(uploaded_file_example_2)
    # Create a temporary directory to save the uploaded file
    temp_dir = tempfile.mkdtemp()

    # Save the uploaded file to the temporary directory
    file_path = os.path.join(temp_dir, uploaded_file_example_2.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file_example_2.getvalue())

    with grid_image:
        example_text_1 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Original Image &#128247;</p>'
        st.markdown(example_text_1, unsafe_allow_html=True)
        st.image(image_user_example_2, caption='Uploaded Image', width=image_width)
    with grid_predictions:
        if labels_user:
            example_text_3 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Model Predictions &#127919;</p>'
            st.markdown(example_text_3, unsafe_allow_html=True)
            labels_user_list = [label.strip() for label in labels_user.split(',')]
            result_df_labels_3 = process_image_labels(file_path, labels=labels_user_list)
            st.dataframe(result_df_labels_3.style.background_gradient(cmap='Blues'))
        else:
            st.info("Please enter one or multiple labels (separated by comma).")
else:
    st.write("")

st.divider()
