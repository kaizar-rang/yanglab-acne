## Setup

### 1. Clone the repo
```bash
git clone https://github.com/kaizar-rang/yanglab-acne.git
cd yanglab-acne
```

### 2. Create and activate the conda environment
**Mac/Linux — Terminal:**
```bash
conda create -n yanglab python=3.10
conda activate yanglab
```

**Windows — use Anaconda Prompt, not PowerShell:**
```bash
conda create -n yanglab python=3.10
conda activate yanglab
```

If conda is not recognized in your terminal, install Miniconda from https://docs.conda.io/en/latest/miniconda.html and restart your terminal.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API keys
Create a `.env` file in the repo root (never commit this file):

```bash
touch .env
```

### 5. Setup

### 6. Preprocessing

After setup, run the patch extractor to generate training patches for Part 2:

```bash
python src/data/patch_extractor.py
```

This will populate `data/patches/` with ~48k labeled patches. Expected output:
- `data/patches/train/acne/` — ~39,713 patches
- `data/patches/train/clear/` — ~8,919 patches
- `data/patches/val/acne/` — ~3,423 patches
- `data/patches/val/clear/` — ~849 patches