import os 
import cv2 
import numpy as np 
from pathlib import Path 
# Tells scripts where to find inputs/outputs
ACNE04_IMAGES = Path("data/acne04/train/images")
ACNE04_LABELS = Path("data/acne04/train/labels")
OUTPUT_DIR = Path("data/patches") 

# Patches are small cropped regions of skin (224x224). 
# Positive patches contain acne (from bounding boxes)
# Negative patches are random clear skin crops from the same image

# Make output folders in data/patches, if they dont already exist
def create_output_dirs():
    for split in ["train", "val"]:
        for label in ["acne", "clear"]:
            os.makedirs(OUTPUT_DIR / split / label, exist_ok=True)

# Each image has a txt file listing every acne bounding box. Each line is a box written as class id, x center, y center, width, height, as fractions between 0 and 1. 
# Read every line and return them as list of boxes, these are the yolo coordinates

def read_yolo_labels(label_path):
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            boxes.append((class_id, x_center, y_center, width, height))
    return boxes


# Convert YOLO coordinates to pixel coordinates
def yolo_to_pixels(x_center, y_center, width, height, img_w, img_h):
    # Convert normalized coordinates to actual pixel values
    x_center = int(x_center * img_w)
    y_center = int(y_center * img_h)
    width = int(width * img_w)
    height = int(height * img_h)
    
    # Calculate the top-left and bottom-right corners of the box
    x1 = max(0, x_center - width // 2)
    y1 = max(0, y_center - height // 2)
    x2 = min(img_w, x_center + width // 2)
    y2 = min(img_h, y_center + height // 2)
    
    return x1, y1, x2, y2  


# Crop and save a patch
def save_patch(image, x1, y1, x2, y2, output_path, patch_size=(224, 224)):
    # Crop the region from the image using the pixel coordinates
    patch = image[y1:y2, x1:x2]
    
    # Skip if the crop is empty (can happen with bad labels)
    if patch.size == 0:
        return
    
    # Resize to a fixed size so all patches are the same dimensions
    patch = cv2.resize(patch, patch_size)
    
    # Save the patch to disk
    cv2.imwrite(str(output_path), patch)

def extract_negative_patch(image, boxes, img_w, img_h, patch_size=(224, 224)):
    # Convert all boxes to pixel coordinates
    pixel_boxes = []
    for _, x_center, y_center, width, height in boxes:
        x1, y1, x2, y2 = yolo_to_pixels(x_center, y_center, width, height, img_w, img_h)
        pixel_boxes.append((x1, y1, x2, y2))
    
    # Try random crops until we find one that doesn't overlap with any acne box
    for _ in range(50):
        # Pick a random top-left corner
        rx1 = np.random.randint(0, img_w - patch_size[0])
        ry1 = np.random.randint(0, img_h - patch_size[1])
        rx2 = rx1 + patch_size[0]
        ry2 = ry1 + patch_size[1]
        
        # Check if this random crop overlaps with any acne bounding box
        overlap = False
        for (x1, y1, x2, y2) in pixel_boxes:
            if rx1 < x2 and rx2 > x1 and ry1 < y2 and ry2 > y1:
                overlap = True
                break
        
        # If no overlap found, crop and return it
        if not overlap:
            patch = image[ry1:ry2, rx1:rx2]
            return cv2.resize(patch, patch_size)
    
    # If we couldn't find a clean crop after 50 tries, return None
    return None


def process_image(image_path, label_path, split, patch_idx):
    # Load the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return patch_idx
    
    # Get image dimensions
    img_h, img_w = image.shape[:2]
    
    # Read the bounding boxes from the label file
    boxes = read_yolo_labels(label_path)
    
    # Extract one positive patch per bounding box
    for box in boxes:
        _, x_center, y_center, width, height = box
        x1, y1, x2, y2 = yolo_to_pixels(x_center, y_center, width, height, img_w, img_h)
        output_path = OUTPUT_DIR / split / "acne" / f"patch_{patch_idx:05d}.jpg"
        save_patch(image, x1, y1, x2, y2, output_path)
        patch_idx += 1
    
    # Extract one negative patch per image

    for i in range(3):
        neg_patch = extract_negative_patch(image, boxes, img_w, img_h)
        if neg_patch is not None:
            output_path = OUTPUT_DIR / split / "clear" / f"patch_{patch_idx:05d}.jpg"
            cv2.imwrite(str(output_path), neg_patch)
            patch_idx += 1
        
    return patch_idx

def main():
    create_output_dirs()
    
    # Get all image files
    image_files = sorted(ACNE04_IMAGES.glob("*.jpg"))
    
    # Split into train and val (80/20)
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    patch_idx = 0
    
    # Process training images
    print(f"Processing {len(train_files)} training images...")
    for image_path in train_files:
        label_path = ACNE04_LABELS / image_path.with_suffix(".txt").name
        if not label_path.exists():
            continue
        patch_idx = process_image(image_path, label_path, "train", patch_idx)
    
    # Process validation images
    print(f"Processing {len(val_files)} validation images...")
    for image_path in val_files:
        label_path = ACNE04_LABELS / image_path.with_suffix(".txt").name
        if not label_path.exists():
            continue
        patch_idx = process_image(image_path, label_path, "val", patch_idx)
    
    print(f"Done. {patch_idx} total patches saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()