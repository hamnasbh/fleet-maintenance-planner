# Fleet Maintenance Planner

Predicting the remaining useful life (RUL) of aircraft turbofan engines, with a dashboard to help plan maintenance.

## Project status

Work in progress: modelling on FD001 is complete; the maintenance planning dashboard is next.

## The problem

Aircraft engines wear out with every flight. Servicing them too early wastes money and grounds aircraft for nothing; servicing them too late risks a failure. Predictive maintenance aims to estimate, for each engine, how many cycles (roughly, flights) it has left before failure, so maintenance can be planned at the right time.

## Data

This project uses the NASA C-MAPSS turbofan engine degradation dataset. Each engine is recorded cycle by cycle with 3 operational settings and 21 sensors (temperatures, pressures, rotation speeds, etc.).

The current work uses subset **FD001** (one operating condition, one fault mode):

- `train_FD001.txt`: 100 engines run until failure.
- `test_FD001.txt`: 100 engines whose records stop before failure.
- `RUL_FD001.txt`: the true remaining life of each test engine at its last recorded cycle.

Download the dataset from the NASA Open Data Portal (or Kaggle), extract it, and place the `.txt` files in the `data/` folder.

## Approach

1. **Target**: RUL = cycles left before failure, capped at 125 because wear is not detectable early in an engine's life.
2. **Sensor selection**: removed sensors that never vary, then kept those with an absolute correlation with RUL above 0.5.
3. **Features**: rolling mean and standard deviation of each selected sensor over the last 80 cycles, to capture wear trends rather than noisy single readings.
4. **Validation**: 5-fold cross-validation **grouped by engine**, so cycles from the same engine never appear in both training and validation. The test set was used only for the final evaluation.
5. **Model**: gradient boosting (`HistGradientBoostingRegressor` from scikit-learn), compared against a linear regression baseline.

## Results

Two metrics are used:

- **RMSE**: average prediction error, in cycles.
- **NASA score**: penalizes late predictions (overestimating remaining life, which is dangerous) more heavily than early ones. Lower is better.

| Model | CV RMSE | Test RMSE | Test NASA score |
|---|---|---|---|
| Linear regression on raw sensors (baseline) | 22.9 | 21.3 | 1145 |
| Gradient boosting, 80-cycle rolling features | 15.0 | 15.0 | 446 |

The final model reduces the error by about 30% and the NASA score by about 60% compared to the baseline. Cross-validation and test RMSE match, which indicates the model generalizes well to unseen engines.

Key findings from model selection:

- Gradient boosting outperformed linear regression for every feature configuration tested.
- Longer rolling windows consistently helped, with cross-validation RMSE dropping from 19.4 (10 cycles) to 15.0 (80 cycles) and plateauing beyond that.
- Errors were not larger for test engines with short histories, despite the long window.

## Error analysis

On average, predictions are slightly optimistic (+1.6 cycles). A few engines remain hard to predict, with errors of up to about ±45 cycles.

The largest optimistic errors occur in mid-life, when the engines still had 57 to 84 cycles left: their sensors show little or no sign of wear yet, so the model keeps predicting a healthy engine. Since sensor signals become clearer as failure approaches, re-computing predictions at every cycle should let them catch up, and a safety margin in maintenance planning can absorb the remaining optimism.

## Limitations

- The RUL cap at 125 assumes wear becomes visible at the same point for every engine, which is a simplification: in reality, this point likely varies from one engine to another.
- Predictions fluctuate by about ±10 cycles from one cycle to the next; the dashboard will smooth them over recent cycles.
- Only FD001 has been modelled so far. Subsets FD002–FD004 include multiple operating conditions and fault modes, and would require normalizing sensors per operating condition.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook
```

## Project structure

```
data/        Dataset files (not tracked by Git)
notebooks/   Exploration and modelling (01_exploration.ipynb)
src/         Reusable code
app/         Dashboard
```

## Roadmap

- [x] Data exploration
- [x] RUL target and baseline model
- [x] Feature engineering, model comparison and cross-validation
- [x] Final evaluation and error analysis on FD001
- [ ] Move reusable code into `src/`
- [ ] Maintenance planning dashboard
- [ ] Extend to FD002–FD004