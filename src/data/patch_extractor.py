import os # navigate file paths
import cv2 # OpenCV for reading/cropping images
import numpy as np # Numerical operations on image arrays
from pathlib import Path # Cleanly handle file paths

# Tells scripts where to find inputs/outputs
ACNE04_IMAGES = Path("data/acne04/train/images")
ACNE04_LABELS = Path("data/acne04/train/labels")
OUTPUT_DIR = Path("data/patches") 

# Patches are small cropped regions of skin (224x224). 
# Positive patches contain acne (from bounding boxes)
# Negative patches are random clear skin crops from the same image

def create_output_dirs():
    for split in ["train", "val"]:
        for label in ["acne", "clear"]:
            os.makedirs(OUTPUT_DIR / split / label, exist_ok=True)
