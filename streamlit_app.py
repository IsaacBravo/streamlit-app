import streamlit as st
from PIL import Image
from utils import icon

# UI configurations
st.set_page_config(page_title="Replicate Image Generator",
                   page_icon=":desktop_computer:",
                   #layout="wide"
                   
                   )
icon.show_icon(":desktop_computer:")
st.markdown("# :blue[Image Uploader and Display]")

# Placeholders for images and gallery
generated_images_placeholder = st.empty()
gallery_placeholder = st.empty()

def configure_sidebar() -> None:
    """
    Setup and display the sidebar elements.

    This function configures the sidebar of the Streamlit application, 
    including the form for user inputs and the resources section.
    """
    with st.sidebar:
        with st.form("my_form"):
            st.info("**Welcome! Start here ↓**", icon="👋")
            with st.expander(":blue[**Refine your output here**]"):
                # Advanced Settings (for the curious minds!)
                uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
                image_width = st.number_input("Width of output image", value=500)
                image_height = st.number_input("Height of output image", value=500)
                # num_outputs = st.slider("Number of images to output", value=1, min_value=1, max_value=4)
            prompt = st.text_area(
                ":blue[**Enter prompt: ✍️**]",
                value="An astronaut riding a rainbow unicorn, cinematic, dramatic")

            # The Big Red "Submit" Button!
            submitted = st.form_submit_button(
                "Submit", type="primary", use_container_width=True)

        st.divider()

        return submitted, image_width, image_height, prompt, uploaded_file

# Configure the sidebar and get user input
submitted, image_width, image_height, prompt, uploaded_file = configure_sidebar()

# Check if the form is submitted and an image is uploaded
if submitted and uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    # Display the uploaded image with the specified width
    st.write("Original Image :camera:")
    st.image(image, caption='Uploaded Image', width=image_width)

# Show the uploaded image placeholder
uploaded_image_placeholder = st.empty()

# Check if the form is submitted
# if submitted:
    # Display the uploaded image
    # uploaded_image_placeholder.image(image, caption='Uploaded Image', width=image_width)