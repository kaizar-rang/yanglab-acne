"""
setup.py
Run this script once after cloning the repo to download all datasets and set up the data directory structure.
Usage: python setup.py
"""

import os
import shutil
from pathlib import Path


def create_directories():
    print("Creating directory structure...")
    dirs = [
        "data/acne04",
        "data/acne04_coco",
        "data/dermnet",
        "data/patches/train/acne",
        "data/patches/train/clear",
        "data/patches/val/acne",
        "data/patches/val/clear",
        "outputs/checkpoints",
        "outputs/predictions",
        "outputs/metrics",
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)

    print("Done.\n")


def download_acne04():
    print("Downloading ACNE04 (YOLOv5 format)...")
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")  # replace with your key
    project = rf.workspace("acne-vulgaris-detection").project("acne04-detection")
    version = project.version(1)
    version.download("yolov5", location="data/acne04")

    # Fix nested folder if present
    nested = Path("data/acne04/acne04-detection-1")
    if nested.exists():
        for f in nested.iterdir():
            shutil.move(str(f), "data/acne04/")
        nested.rmdir()

    print("Done.\n")


def download_acne04_coco():
    print("Downloading ACNE04 (COCO format)...")
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")  # replace with your key
    project = rf.workspace("acne-vulgaris-detection").project("acne04-detection")
    version = project.version(1)
    version.download("coco", location="data/acne04_coco")
    print("Done.\n")


def download_dermnet():
    print("Downloading DermNet...")
    import kagglehub
    path = kagglehub.dataset_download("shubhamgoel27/dermnet")

    # Move from kaggle cache to project data folder
    dest = Path("data/dermnet")
    src = Path(path)
    for f in src.iterdir():
        shutil.move(str(f), str(dest))

    print("Done.\n")


if __name__ == "__main__":
    create_directories()
    download_acne04()
    download_acne04_coco()
    download_dermnet()
    print("Setup complete. You are ready to go.")