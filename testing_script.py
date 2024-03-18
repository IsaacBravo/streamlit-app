# Install the packages
import os
import clip
import torch
from PIL import Image
from torchvision.datasets import CIFAR100
import matplotlib.pyplot as plt
import pandas as pd

# Set the environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs -- ADD THE PATH WHERE IS THE IMAGE
image = Image.open('climate_image.jpeg')

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

# empty lists to store the results
class_names = []
percentages = []

# iterate over values and indices
for value, index in zip(values, indices):
    # get the class name and percentage
    class_name = cifar100.classes[index]
    percentage = 100 * value.item()

    # append to the lists
    class_names.append(class_name)
    percentages.append(percentage)

# create the dataframe
df = pd.DataFrame({
    'Class Name': class_names,
    'Percentage': percentages
})

df = df.sort_values(by='Percentage')

# print the dataframe
# print(df)

# create a figure with two subplots, one for the bar plot and one for the image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# plot the bar plot in the first subplot
ax1.barh(df['Class Name'], df['Percentage'], color='blue')
ax1.set_xlabel('Percentage')
ax1.set_ylabel('Class Name')

ax2.imshow(image)
ax2.axis('off')

# adjust the spacing between the two subplots
fig.subplots_adjust(wspace=0.1)

# show the plot
plt.show()

