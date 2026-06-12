# yanglab-acne

ML research project for acne detection and cross-domain classification using ACNE04 and DermNet datasets.

## Project Structure

```
yanglab-acne/
├── setup.py                      # Run once after cloning
├── data/                         # Downloaded datasets (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_training_yolo.ipynb    # YOLOv5 training (Colab GPU)
│   ├── 03_training_frcnn.ipynb   # Faster R-CNN training (Colab GPU)
│   ├── 04_evaluation.ipynb       # Evaluation and metrics
│   └── 05_visualization.ipynb    # Detection and Grad-CAM visualizations
├── src/
│   ├── data/patch_extractor.py   # Generates training patches for Part 2
│   ├── models/
│   ├── eval/
│   └── viz/
├── configs/                      # Generated config files (gitignored)
└── outputs/                      # Weights, predictions, metrics (gitignored)
```

---

## Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/kaizar-rang/yanglab-acne.git
cd yanglab-acne
```

### 2. Create and activate the conda environment

**Mac/Linux:**
```bash
conda create -n yanglab python=3.10
conda activate yanglab
```

**Windows — use Anaconda Prompt, not PowerShell:**
```bash
conda create -n yanglab python=3.10
conda activate yanglab
```

If conda is not recognized, install Miniconda from https://docs.conda.io/en/latest/miniconda.html and restart your terminal.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API keys
Create a `.env` file in the repo root (never commit this file):
```
ROBOFLOW_API_KEY=your_key_here
KAGGLE_USERNAME=your_username_here
KAGGLE_KEY=your_kaggle_key_here
```

See `.env.example` for reference. Your Roboflow key is at app.roboflow.com → account settings. Your Kaggle credentials are at kaggle.com/settings → API.

### 5. Run setup
```bash
python setup.py
```

Prompts you to select which datasets to download, then clones YOLOv5 and generates `configs/acne04.yaml`.

### 6. Run patch extraction
```bash
python src/data/patch_extractor.py
```

Populates `data/patches/` with ~52k labeled patches for Part 2 classifier training.

Expected output:
- `data/patches/train/acne/` — ~39,713 patches
- `data/patches/train/clear/` — ~8,919 patches
- `data/patches/val/acne/` — ~3,423 patches
- `data/patches/val/clear/` — ~849 patches

---

## Training Detection Models

Training is designed to run on a GPU. Open the relevant notebook in Google Colab / Kaggle Notebook with a T4 GPU runtime:

- **YOLOv5:** `notebooks/02_training_yolo.ipynb`
- **Faster R-CNN:** `notebooks/03_training_frcnn.ipynb`
- **YOLOv8s & YOLOv8-P2:** `notebooks/04_training_yolov8.ipynb`

Each notebook is self-contained — it clones the repo, downloads data, trains the model, and downloads results to your local machine. Run the download cell immediately after training finishes before closing the session.

---

## Training Detection Models

Training is designed to run on a GPU. Open the relevant notebook in Google Colab / Kaggle Notebook with a T4 GPU runtime:

- **EfficientNetB0 Classifier on DermNet** `notebooks/05_classification.ipynb`

Each notebook is self-contained — it clones the repo, downloads data, trains the model, and downloads results to your local machine. Run the download cell immediately after training finishes before closing the session.

---



