# Streamlit Image Classification App

This Streamlit application allows users to upload images and get predictions from the CLIP model developed by OpenAI. The CLIP model, which stands for "Contrastive Language-Image Pre-training," is a state-of-the-art artificial intelligence model capable of understanding both images and text.

## Features

- Upload images: Users can upload images directly to the app interface.
- Get predictions: The app uses the CLIP model to predict the class or content depicted in the uploaded image based on its visual features and any accompanying text description.

## How to Use

1. Clone this repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Streamlit app using `streamlit run app.py`.
4. Once the app is running, upload an image using the provided file uploader.
5. Wait for the app to process the image and display the predictions.

## About CLIP Model

The CLIP model is a powerful AI model developed by OpenAI that can understand both images and text. It achieves this by training on a large dataset of image-text pairs using a technique called contrastive learning. This enables the model to learn a joint representation space where images and text are semantically similar if they describe the same concept.

## Credits

This app was created by Isaac Bravo, Katharina Prasse, and Hsien-Yi Wang. It uses the CLIP model developed by OpenAI:

Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G. &amp; Sutskever, I.. (2021). Learning Transferable Visual Models From Natural Language Supervision. <i>Proceedings of the 38th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 139:8748-8763 Available from https://proceedings.mlr.press/v139/radford21a.html.



