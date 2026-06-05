"""
setup.py

Run this script once after cloning the repo to download all datasets
and set up the data directory structure automatically.

Usage: python setup.py
"""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")



# check if any required env variables are missing
def check_env():
    missing = []

    # Check each key — if os.getenv() returned None, the key is missing
    if not ROBOFLOW_API_KEY:
        missing.append("ROBOFLOW_API_KEY")
    if not KAGGLE_USERNAME:
        missing.append("KAGGLE_USERNAME")
    if not KAGGLE_KEY:
        missing.append("KAGGLE_KEY")

    # If anything is missing, stop immediately with a helpful message
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Add them to a .env file in the repo root. See README.md for details."
        )

    print("API keys loaded.\n")

# create locations for data to be added to
def create_directories():
    
    print("Creating directory structure...")

    dirs = [
        "data/acne04",           # raw ACNE04 images + labels in YOLOv5 format
        "data/acne04_coco",      # same images, COCO format for Faster R-CNN
        "data/dermnet",          # raw DermNet images
        "data/patches/train/acne",   # positive training patches (acne crops)
        "data/patches/train/clear",  # negative training patches (clear skin crops)
        "data/patches/val/acne",     # positive validation patches
        "data/patches/val/clear",    # negative validation patches
        "outputs/checkpoints",   # saved model weights
        "outputs/predictions",   # sample output images with boxes/grad-cam
        "outputs/metrics",       # CSVs or JSONs with eval results
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)

    print("Done.\n")

# Downloads ACNE04 in YOLOv5 format from Roboflow
def download_acne04():
    """
    Downloads ACNE04 in YOLOv5 format from Roboflow.
    YOLOv5 format gives us one .txt label file per image, each line being
    one bounding box in normalized [class x_center y_center width height] format.
    This is used for Part 1 (detection) and as the source for patch extraction.

    Roboflow sometimes nests the download inside an extra folder
    (acne04-detection-1/). The cleanup block moves everything up one level
    so the structure is always data/acne04/train/, data/acne04/valid/, etc.
    """
  
    print("Downloading ACNE04 (YOLOv5 format)...")
    from roboflow import Roboflow
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("acne-vulgaris-detection").project("acne04-detection")
    version = project.version(1)
    dataset = version.download("yolov5", location="data/acne04", overwrite=True)
    print(f"Downloaded to: {dataset.location}")
    print("Done.\n")

# Downloads ACNE04 in COCO format from Roboflow.
def download_acne04_coco():
    """
    Downloads ACNE04 in COCO format from Roboflow.
    COCO format gives us a single annotations.json file with absolute pixel
    coordinates for all bounding boxes. This is required for Faster R-CNN
    and DINO-DETR which don't accept YOLOv5's .txt format.
    Same images, different annotation format.
    """
    print("Downloading ACNE04 (COCO format)...")

    from roboflow import Roboflow
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("acne-vulgaris-detection").project("acne04-detection")
    version = project.version(1)
    version.download("coco", location="data/acne04_coco", overwrite=True)

    print("Done.\n")

# Downloads DermNet from Kaggle using kagglehub.
def download_dermnet():
    print("Downloading DermNet...")
    
    import json
    import subprocess
    
    # Write credentials to kaggle.json
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    with open(kaggle_dir / "kaggle.json", "w") as f:
        json.dump({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}, f)
    
    # Use kaggle CLI directly
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "shubhamgoel27/dermnet",
        "-p", "data/dermnet",
        "--unzip"
    ], check=True)
    
    print("Done.\n")

def clone_yolov5():
    print("Cloning YOLOv5...")
    import subprocess
    if not Path("yolov5").exists():
        subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5"], check=True)
        subprocess.run(["pip", "install", "-r", "yolov5/requirements.txt"], check=True)
    else:
        print("YOLOv5 already cloned, skipping...")
    print("Done.\n")


def create_yolo_config():
    print("Creating YOLOv5 config...")
    import yaml
    
    repo_root = Path.cwd()
    config = {
        "train": str(repo_root / "data/acne04/train/images"),
        "val": str(repo_root / "data/acne04/valid/images"),
        "test": str(repo_root / "data/acne04/test/images"),
        "nc": 4,
        "names": ["nodules and cysts", "papules", "pustules", "whitehead and blackhead"]
    }
    os.makedirs("configs", exist_ok=True)
    with open("configs/acne04.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print("Done.\n")


if __name__ == "__main__":
    check_env()
    create_directories()

    print("Which datasets would you like to download?")
    print("  1. ACNE04 (YOLOv5)")
    print("  2. ACNE04 (COCO)")
    print("  3. DermNet")
    print("  4. All")
    print("  5. Skip downloads")

    choice = input("\nEnter your choice (1-5): ").strip()

    if choice in ("1", "4"):
        download_acne04()
    if choice in ("2", "4"):
        download_acne04_coco()
    if choice in ("3", "4"):
        download_dermnet()
    if choice == "5":
        print("Skipping downloads.\n")

    clone_yolov5()
    create_yolo_config()
    print("Setup complete. You are ready to go.")