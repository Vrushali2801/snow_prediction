# Code Challenge- Snow Prediction for Tomorrow
Goal: Predict whether it will snow tomorrow for each station using time-aware splits, trailing features, and a tuned classifier that outputs both probabilities and labels.

## Repository Layout:

data/interim/ — contains cleaned extracts and stations table.
data/processed/ — train/validation/test splits, predictions, and metrics.
notebook/ — stepwise notebooks for setup, Part 1 (data extraction), and Part 2 (modeling: LightGBM and Logistic Regression).

## Data Scope:

Stations: subset within 725300–725330 with full coverage from 2000–2005 (stations_ok.csv).
Target: y_tomorrow indicates if snow occurs the next calendar day at the same station. Created by sorting by station and date, and then shifting the snow flag by -1 day.

## Features Used:

Numeric: total_precipitation, mean_temp, max_temperature.
Boolean flags: rain, fog, hail, thunder, tornado.
Time context: per-station lag1, lag3, lag7 and lag14 and 3-day rolling mean for numeric features.
Seasonality: month and day-of-year encoded as sine/cosine.
Excluded: min_temperature 100% missing, snow_depth ~96% missing.

## Models used:

LightGBM: gradient‑boosted decision trees using LGBMClassifier, handles non-linear interactions efficiently and robustly for tabular data.
Logistic Regression: fast, interpretable linear classifier with regularization, serves as a baseline

## Threshold Tuning:

The decision threshold for each model is tuned on the validation set to optimize F1.
This is important because when classes are imbalanced, the default 0.50 cutoff may not be optimal.
Threshold tuning is implemented using a simple grid search over probabilities.

## Training Flow

Fit the Pipeline on train only.
Evaluate and tune the threshold on validation.
Refit on train+valid with the chosen threshold.
Score the fixed test split to produce probabilities and labels per station.

## Outputs

data/processed/train.csv, valid.csv, test.csv: final modeling splits aligned to the test date.
data/processed/test_predictions_lightgbm.csv, test_metrics_lightgbm.csv: LightGBM predictions and metrics.
data/processed/test_predictions_logreg.csv, test_metrics_logreg.csv: Logistic Regression predictions and metrics.


## Interpreting Results
AUC (Area Under the ROC Curve): Measure of how well the model ranks positives above negatives across all possible thresholds. A higher AUC means the model’s probability estimates are good at separating “snow tomorrow” vs. “no snow tomorrow,” regardless of the cutoff you eventually pick.

F1 Score: This balances precision (how many predicted snows were actually snow) and recall (how many actual snows the model caught)

Threshold: By default, classifiers use 0.5 as the cutoff, but that’s rarely optimal—especially if the classes are imbalanced. We tune the threshold on the validation set to get the best F1, then lock that threshold before scoring the test set. This way, test results are a fair reflection of how the model will behave in the real world.

## Model comparison

Both models predicted the same labels: one snow day caught, one missed.
Logistic Regression ranked probabilities better (higher AUC), but after thresholding, both produced the same F1, accuracy, and recall.
Perfect precision (no false positives) but limited recall (missed half of true snow events).