# Applying Probability and Statistics to Evaluate a Machine Learning Model

**Authors:** Abhay Burye, Jalesh Kumar, Uttam K

**Question:** Does an improved model (tuned Random Forest) really beat a baseline (Decision Tree), or could the gap be due to variation in the test sample?

**Dataset:** [Kaggle - Weather Type Classification](https://www.kaggle.com/datasets/nikhil7280/weather-type-classification) (13,200 rows, 4 balanced classes)

## What the code does
1. Loads and cleans the data (duplicates removed, impossible percentages set to missing, median/mode imputation inside the pipeline)
2. Descriptive statistics and EDA plots
3. One stratified 80/20 train-test split, shared by both models
4. Baseline Decision Tree and a Random Forest tuned with 5-fold CV on the training set only
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
All tables and figures are written to `outputs/`. Run `python eda.py` for the exploratory analysis. Everything is seeded (SEED = 42).

## Result
| | Decision Tree | Random Forest | Difference |
|---|---|---|---|
| Accuracy | 0.9083 | 0.9121 | +0.0038 |
| 95% CI | [0.8973, 0.9193] | [0.9013, 0.9229] | [-0.0116, +0.0192] |

Two-proportion z-test: z = 0.48, p = 0.63 -> **fail to reject H0**. The Random Forest's small lead could be due to variation in the test sample.

## Files
- `weather_ml_stats.py` - full reproducible implementation
- `eda.py` - exploratory data analysis (black-and-white figures and tables in `eda_outputs/`)
- `Weather_Type_Classification_ML_Stats.ipynb` - notebook version (includes the Weather Check demo app)
- `ML_Statistics_Report_Weather_Classification.docx` / `.pdf` - report
