# Freight Rate Prediction

A machine learning pipeline that predicts freight (trucking) load rates from route, equipment, and market-condition data. The model is trained with **LightGBM** on historical posted rates and used to generate predictions for a held-out validation set and a forward-looking "December scenario" set.

## Overview

Given a load's pickup/delivery locations, distance, equipment type, weight, and market signals, the pipeline predicts `posted_rate` (the price the load should command). It includes:

- Feature engineering derived from exploratory data analysis (EDA) — geographic distance, seasonality, market conditions, and route-level statistics
- A LightGBM regression model trained on log-transformed rates
- A time-based local validation split to estimate out-of-sample performance before generating final predictions
- A scoring script that validates prediction files and produces a chart comparing predicted vs. expected December rates

## Project Structure

```
.
├── data/                              # Input datasets (train/test, validation, December scenario)
├── scorer_results/                    # Output of score.py (e.g. candidate_december.png)
├── freight_rate_prediction.ipynb      # Exploratory data analysis notebook
├── train_predict.py                   # Feature engineering, model training, and prediction generation
├── score.py                           # Validates predictions and scores/plots results
├── requirements.py                    # Scorer dependencies (matplotlib, numpy, pandas)
└── validation_predictions.csv         # Generated predictions for the validation set
```

> Note: `requirements.py` despite its extension is a plain pip requirements list used by `score.py` (matplotlib, numpy, pandas). It does **not** cover the packages needed to run `train_predict.py` — see [Setup](#setup) below.

## How It Works

### 1. Feature Engineering (`preprocess_and_engineer`)

Applied consistently across the training, validation, and December datasets:

- **Coordinate imputation** — fills missing pickup/delivery latitude/longitude for known cities (e.g. Lexington, Fort Wayne)
- **Date features** — day of week, day of year, days since a reference start date, and cyclical (sine/cosine) encodings of day-of-year to capture seasonality
- **Spatial features** — haversine (great-circle) distance between pickup and delivery, plus raw latitude/longitude differences
- **`distance_bin`** — bucketed haul length (short / medium / long / very long), based on EDA showing distinct rate distributions per bucket
- **`market_hot`** — binary flag for whether `market_index` is above its neutral value of 1.0
- **`route_frequency`** — how often a pickup–delivery pair appears in the training data, computed only from the training set to avoid leakage
- **Missing value imputation** — weight imputed by per-equipment median; `market_index` and `quote_signal` imputed by global median
- **Categorical typing** — `pickup`, `delivery`, and `equipment` cast to `category` dtype for native LightGBM handling

### 2. Model Training (`train_predict.py`)

- Target: `posted_rate`, log-transformed (`log1p`) before training and inverse-transformed (`expm1`) at prediction time
- Model: LightGBM regressor (`objective: regression`, `metric: mae`), gradient-boosted trees with 1,200 boosting rounds
- **Local validation**: a time-based split — train on data before 2025-09-01, validate on data from 2025-09-01 onward — reported using MAE, RMSE, MAPE, and R²
- **Final model**: retrained on the full training dataset for generating submission predictions
- Predictions are clipped to a strictly positive minimum

### 3. Predictions Generated

- `validation_predictions.csv` — predicted rates for every `load_id` in `data/validation.csv`
- `data/december-chart-inputs.csv` — predicted rates for the December scenario rows, with daily `market_index`/`quote_signal` backfilled from validation-set December statistics

### 4. Scoring (`score.py`)

Validates both prediction files and produces `scorer_results/candidate_december.png`, a chart comparing predicted December rates against expectations.

## Run Instructions

```bash
git clone https://github.com/Krdhirendra/frieght_rate_prediction.git
cd frieght_rate_prediction

# Core modeling dependencies (used by train_predict.py)
pip install pandas numpy scikit-learn lightgbm

# Scorer dependencies (used by score.py)
pip install -r requirements.py
```



### Train the model and generate predictions

```bash
python train_predict.py
```

This will:
1. Load `data/train-test.csv`, `data/validation.csv`, and `data/december-chart-inputs.csv`
2. Engineer features and run a time-based local validation, printing MAE / RMSE / MAPE / R²
3. Retrain on the full dataset
4. Write `validation_predictions.csv`
5. Write predictions back into `data/december-chart-inputs.csv`

### Score the predictions

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv
```

This validates the prediction files and saves `scorer_results/candidate_december.png`.

## Data

| File | Description |
|---|---|
| `data/train-test.csv` | Historical loads with known `posted_rate`, used for training and local validation |
| `data/validation.csv` | Loads requiring a `predicted_rate` prediction, identified by `load_id` |
| `data/december-chart-inputs.csv` | A forward-looking December scenario set requiring predicted rates |

Key columns include `pickup`, `delivery`, `distance`, `equipment`, `weight`, `date`, `market_index`, `quote_signal`, and (for training data) `posted_rate`.



## Results

Local time-based validation (train: Jan–Aug 2025, validation: Sep–Oct 2025) reports MAE, RMSE, MAPE, and R² to the console when `train_predict.py` is run. See `freight_rate_prediction.ipynb` for the full exploratory analysis behind the feature choices, and `scorer_results/candidate_december.png` for the December scenario chart.

## Notebook

`freight_rate_prediction.ipynb` contains the exploratory data analysis that motivated the engineered features above, including the rate distributions by distance bucket, the market index time series, and route volume analysis.