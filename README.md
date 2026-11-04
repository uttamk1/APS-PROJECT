# Applying Probability and Statistics to Evaluate a Machine Learning Model

**Question:** Does an improved model (tuned Random Forest) really beat a baseline (Logistic Regression), or could the gap be due to variation in the test sample?

**Dataset:** [Kaggle - Weather Type Classification](https://www.kaggle.com/datasets/nikhil7280/weather-type-classification) (13,200 rows, 4 balanced classes)

## What the code does
1. Loads and cleans the data (duplicates removed, impossible percentages set to missing, median/mode imputation inside the pipeline)
2. Descriptive statistics and EDA plots
3. One stratified 80/20 train-test split, shared by both models
4. Baseline Logistic Regression; improved model = better of a tuned Random Forest and a tuned Gradient Boosting model, picked by 5-fold CV on the training set only (Random Forest won)
5. Evaluation on the same held-out test set
6. Sampling: bootstrap resampling of the test set (B = 2000)
7. 95% confidence intervals for each accuracy and the difference (normal approximation)
8. Two-proportion z-test, alpha = 0.05
9. Robustness check over 10 random splits

## Run
```bash
pip install -r requirements.txt
python weather_ml_stats.py                       # uses weather_classification_data.csv or downloads it via kagglehub
python weather_ml_stats.py --data path/to/weather_classification_data.csv
```
All tables and figures are written to `outputs/`. Everything is seeded (SEED = 42).

## Result
| | Logistic Regression | Random Forest | Difference |
|---|---|---|---|
| Accuracy | 0.8667 | 0.9121 | +0.0455 |
| 95% CI | [0.8537, 0.8796] | [0.9013, 0.9229] | [+0.0286, +0.0623] |

Two-proportion z-test: z = 5.27, p < 0.001 -> **reject H0**. The Random Forest is genuinely better; it also won on 10 of 10 extra random splits.

## Files
- `weather_ml_stats.py` - full reproducible implementation
- `Weather_Type_Classification_ML_Stats.ipynb` - earlier exploratory notebook (Decision Tree baseline, includes the Weather Check demo app)
- `ML_Statistics_Report_Weather_Classification.docx` / `.pdf` - report
