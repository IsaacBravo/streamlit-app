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