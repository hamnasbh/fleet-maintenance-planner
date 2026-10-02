# Fleet Maintenance Planner

Predicting the remaining useful life (RUL) of aircraft turbofan engines, with a dashboard to help plan maintenance.

## Project status

Work in progress: data exploration.

## Data

This project uses the NASA C-MAPSS turbofan engine degradation dataset.
Download it from the NASA Open Data Portal (or Kaggle), extract it, and place the `.txt` files in the `data/` folder.

Currently used: `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook
```

## Project structure