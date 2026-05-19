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

def install_dependencies():
    print("Installing dependencies...")
    import subprocess
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
    print("Done.\n")


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
    version.download("yolov5", location="data/acne04")

    # Fix nested folder if Roboflow creates one
    nested = Path("data/acne04/acne04-detection-1")
    if nested.exists():
        for f in nested.iterdir():
            shutil.move(str(f), "data/acne04/")  # move contents up one level
        nested.rmdir()                            # remove the now-empty nested folder

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
    version.download("coco", location="data/acne04_coco")

    print("Done.\n")

# Downloads DermNet from Kaggle using kagglehub.
def download_dermnet():
    """
    Downloads DermNet from Kaggle using kagglehub.
    kagglehub downloads to a global cache (~/.cache/kagglehub/) rather than
    the current directory, so we move it into data/dermnet/ afterward.
    Kaggle requires authentication — we pass the credentials from .env
    into the environment so kagglehub can pick them up automatically.
    """
    print("Downloading DermNet...")

    # Pass Kaggle credentials into the environment so kagglehub can find them
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY

    import kagglehub
    path = kagglehub.dataset_download("shubhamgoel27/dermnet")

    # Move from kaggle cache into the project data folder
    dest = Path("data/dermnet")
    src = Path(path)
    for f in src.iterdir():
        shutil.move(str(f), str(dest))

    print("Done.\n")


if __name__ == "__main__":
    # This block only runs when you execute `python setup.py` directly.
    # If another script imports from setup.py, these lines won't fire.
    # Check keys first, create folders second, then download.
    install_dependencies()
    check_env()
    create_directories()
    download_acne04()
    download_acne04_coco()
    download_dermnet()
    print("Setup complete. You are ready to go.")