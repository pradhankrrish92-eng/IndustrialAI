# IndustrialAI

Predictive maintenance and fault diagnosis on the **synthetic** UCI AI4I 2020 dataset.

> **Status: work in progress (through Milestone 1 - data download and validation).**
> No model has been trained yet and no results are reported. The dataset is synthetic;
> nothing here is validated for real industrial use.

## Dataset

UCI AI4I 2020 Predictive Maintenance Dataset (synthetic):
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Check the license and the citation requested on that page before redistributing anything
derived from it. The raw data is not stored in this repository; it is downloaded on demand.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Get and validate the data

```powershell
python scripts/download_data.py
```

If the automatic download fails, download the zip from the dataset page and run:

```powershell
python scripts/download_data.py --from-file C:\path\to\the\downloaded.zip
```

## Checks

```powershell
pytest
ruff check .
ruff format --check .
```

## Training

Train the baseline Random Forest predictive maintenance model:

```powershell
python scripts/train.py