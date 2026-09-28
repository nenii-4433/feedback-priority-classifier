# Feedback Priority Classifier

A Python command-line tool that classifies feedback as `high`, `medium`, or `low` priority. It includes a TF-IDF model for immediate use and can optionally train/use a sentence-embedding model.

## Requirements

- Python 3.10 or newer
- Dependencies in `requirements.txt`

## Setup

In PowerShell, from this project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, run the commands below using `.\.venv\Scripts\python.exe` instead of `python`.

## Classify feedback

Classify one message:

```powershell
python predict.py "The checkout page is down and customers cannot pay"
```

Start interactive mode:

```powershell
python predict.py
```

Classify a CSV containing a `text` column:

```powershell
python predict.py --file feedback.csv
```

Batch mode writes a sibling file named `feedback_predicted.csv` with predicted priority and confidence columns.

## Train the models

```powershell
python train_model.py
```

Training always attempts to create the TF-IDF model. Sentence-embedding training is optional and downloads the pretrained `all-MiniLM-L6-v2` model on its first run (about 90 MB):

```powershell
pip install sentence-transformers
```

To include additional labeled examples, create `real_feedback.csv` with `text` and `priority` columns. Valid priorities are `high`, `medium`, and `low`. This local file is ignored by Git.

To regenerate the synthetic training dataset:

```powershell
python generate_data.py
```

This overwrites `feedback_dataset.csv`.

## Tests

```powershell
python -m pytest test_model.py -v
python test_cases.py
```

The committed `.joblib` files provide the ready-to-use models for prediction. Retrain with `train_model.py` to replace them.