try:
    import clip
    print("Package is installed.")
except ImportError:
    print("Package is not installed.")



# Configure the sidebar and get user input
submitted, image_width, image_height, uploaded_file = configure_sidebar()

# Show the uploaded image placeholder
uploaded_image_placeholder = st.empty()

# Initialize image variable
image = None

# Check if the form is submitted and an image is uploaded
if submitted and uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    st.write("Original Image :camera:")
    st.image(image, caption='Uploaded Image', width=image_width)

# Configure the prompt and get user input
prompt, submitted_analyse = configure_prompt()


# Three columns with different widths
col1, col2, col3 = st.columns([3,3,3])

with col1:
    st.write('This is column 1')
with col2:
    st.write('This is column 2')
with col3:
    st.write('This is column 3')



# Proceed if an image is uploaded
if image is not None:
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # Download the dataset
    cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

    # Prepare the inputs
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

