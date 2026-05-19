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