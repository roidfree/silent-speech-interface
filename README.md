# EMG Yes/No Classifier

Small scaffold for an EMG-based binary classifier ("yes" vs "no").

Project layout

```
emg-yes-no-classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── yes/
│   │   └── no/
│   └── processed/
│       └── (your .mat files go here after export)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   └── train.py
├── results/
│   ├── plots/
│   └── models/
└── tests/
```

Quick start

1. Create a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Place your exported `.mat` files into `data/processed/` (organized or labelled as you prefer).
3. Run `src/train.py` to run a minimal training script (example in the file).

Notes

- The scaffold includes minimal helper modules for loading .mat files, preprocessing, and creating a simple scikit-learn classifier.
- Extend preprocessing and model definitions to match your dataset and goals.
