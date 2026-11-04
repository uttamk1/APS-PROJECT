"""
Applying Probability and Statistics to Evaluate a Machine Learning Model
=========================================================================
Dataset : Kaggle - Weather Type Classification (nikhil7280/weather-type-classification)
Question: Does the improved model (Random Forest) perform better than the baseline
          (Decision Tree), or could the observed difference be due to variation in
          the test sample?

Pipeline
  1. Load data (local CSV -> kagglehub download)
  2. Clean: remove duplicates, set impossible percentages (>100 or <0) to missing
  3. Exploratory statistics (mean, median, variance, sd, skewness, correlation)
  4. Stratified 80/20 train-test split (ONE held-out test set shared by both models)
  5. Baseline Decision Tree, improved Random Forest (tuned by 5-fold CV on train only)
  6. Evaluation on the same test set (accuracy, balanced accuracy, precision, recall, F1)
  7. Sampling: bootstrap resampling of the test set, plus normal-approximation 95% CIs
  8. Hypothesis test: two-proportion z-test on the two accuracies
  9. Robustness check over 10 different random splits
 10. Results table + written conclusion

Usage
  pip install -r requirements.txt
  python weather_ml_stats.py                       # looks for weather_classification_data.csv
  python weather_ml_stats.py --data path/to/file.csv --boot 2000 --out outputs

Everything is seeded (SEED = 42), so re-running gives identical numbers.
"""

import argparse
import glob
import os
import warnings

import matplotlib
matplotlib.use("Agg")                      # write figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

SEED = 42            # global random seed: split, models and bootstrap are reproducible
ALPHA = 0.05         # significance level for the hypothesis tests
TARGET = "Weather Type"
CSV_NAME = "weather_classification_data.csv"
KAGGLE_ID = "nikhil7280/weather-type-classification"


# --------------------------------------------------------------------------------------
# 1. Data loading
# --------------------------------------------------------------------------------------
def load_dataset(path=None):
    """Load the CSV from --data, the working folder, or Kaggle (kagglehub)."""
    candidates = [path] if path else glob.glob(CSV_NAME) + glob.glob("*weather*.csv")
    for p in candidates:
        if p and os.path.exists(p):
            print(f"Loaded local file: {p}")
            return pd.read_csv(p)
    try:
        import kagglehub
        folder = kagglehub.dataset_download(KAGGLE_ID)
        files = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
        print(f"Downloaded from Kaggle: {files[0]}")
        return pd.read_csv(files[0])
    except Exception as e:
        raise SystemExit(
            f"Could not find or download the dataset ({e}).\n"
            f"Download {CSV_NAME} from https://www.kaggle.com/datasets/{KAGGLE_ID} and either put it "
            f"next to this script or pass --data <path>.")


# --------------------------------------------------------------------------------------
# 2. Cleaning
# --------------------------------------------------------------------------------------
def clean(df_raw):
    print(f"\nRaw shape: {df_raw.shape}")
    print("Missing values per column:\n", df_raw.isna().sum().to_string())
    df = df_raw.drop_duplicates().reset_index(drop=True)
    print(f"\nRows after removing duplicates: {len(df)} (removed {len(df_raw) - len(df)})")

    # Humidity and precipitation are percentages: values outside 0-100 are impossible.
    # They become NaN and are later imputed with the TRAINING median inside the pipeline.
    pct_cols = [c for c in df.columns if "Humidity" in c or "Precipitation" in c]
    for c in pct_cols:
        bad = (df[c] > 100) | (df[c] < 0)
        print(f"{c}: {int(bad.sum())} implausible values (outside 0-100) -> set to NaN")
        df.loc[bad, c] = np.nan
    return df, pct_cols


# --------------------------------------------------------------------------------------
# 3. Exploratory statistics
# --------------------------------------------------------------------------------------
def describe(df, y, num_cols, out):
    desc = pd.DataFrame({
        "mean": df[num_cols].mean(), "median": df[num_cols].median(),
        "variance": df[num_cols].var(), "std": df[num_cols].std(),
        "min": df[num_cols].min(), "max": df[num_cols].max(),
        "skewness": df[num_cols].skew()}).round(3)
    desc.to_csv(f"{out}/descriptive_statistics.csv")
    print("\nDescriptive statistics:\n", desc.to_string())
    print("\nClass counts:\n", y.value_counts().to_string())

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = y.value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="Set2")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({v / len(y):.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_title("Class distribution"); ax.set_ylabel("Count"); ax.set_ylim(0, counts.max() * 1.2)
    plt.tight_layout(); plt.savefig(f"{out}/class_distribution.png", dpi=150); plt.close()

    n, cols = len(num_cols), 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    for ax, c in zip(axes.ravel(), num_cols):
        sns.histplot(df[c].dropna(), kde=True, ax=ax, color="steelblue"); ax.set_title(c)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(f"{out}/histograms.png", dpi=150); plt.close()

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    for ax, c in zip(axes.ravel(), num_cols):
        sns.boxplot(data=df, x=TARGET, y=c, ax=ax, palette="Set2"); ax.set_title(f"{c} by weather type")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(f"{out}/boxplots_by_class.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Pearson correlation between numeric features")
    plt.tight_layout(); plt.savefig(f"{out}/correlation_heatmap.png", dpi=150); plt.close()


# --------------------------------------------------------------------------------------
# 4-5. Models
# --------------------------------------------------------------------------------------
def make_preprocessor(num_cols, cat_cols):
    """Median imputation for numbers, mode imputation + one-hot for categories.
    Fitted inside each pipeline on training data only, so the test set never leaks in."""
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


def make_dt(num_cols, cat_cols):
    return Pipeline([("prep", make_preprocessor(num_cols, cat_cols)),
                     ("clf", DecisionTreeClassifier(random_state=SEED))])


def make_rf(num_cols, cat_cols, **params):
    return Pipeline([("prep", make_preprocessor(num_cols, cat_cols)),
                     ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1, **params))])


def metrics(y_true, y_pred):
    return {"Accuracy": accuracy_score(y_true, y_pred),
            "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
            "Macro precision": precision_score(y_true, y_pred, average="macro"),
            "Macro recall": recall_score(y_true, y_pred, average="macro"),
            "Macro F1": f1_score(y_true, y_pred, average="macro")}




# --------------------------------------------------------------------------------------
# 7. Confidence intervals and sampling
# --------------------------------------------------------------------------------------
Z95 = stats.norm.ppf(0.975)   # 1.96


def normal_ci(p, n):
    """95% CI for a proportion (accuracy): p +/- 1.96 * sqrt(p(1-p)/n)."""
    se = np.sqrt(p * (1 - p) / n)
    return p - Z95 * se, p + Z95 * se, se


def diff_ci(p1, p2, n1, n2):
    """95% CI for the difference of two proportions p2 - p1."""
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p2 - p1
    return d - Z95 * se, d + Z95 * se, se


def bootstrap_accuracy(correct_dt, correct_rf, n_boot, rng):
    """Draw n_boot samples of the test set WITH replacement and record both accuracies."""
    n = len(correct_dt)
    acc_dt, acc_rf = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        acc_dt[b] = correct_dt[idx].mean()
        acc_rf[b] = correct_rf[idx].mean()
    return acc_dt, acc_rf


# --------------------------------------------------------------------------------------
# 8. Hypothesis test: two-proportion z-test
# --------------------------------------------------------------------------------------
def two_proportion_z_test(x1, n1, x2, n2):
    """H0: p1 = p2 (both models have the same accuracy).  H1: p1 != p2.
    z = (p2 - p1) / sqrt(p(1-p)(1/n1 + 1/n2)), where p is the pooled proportion."""
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se
    p_two = 2 * stats.norm.sf(abs(z))
    p_one = stats.norm.sf(z)          # H1: p2 > p1 (improved model better)
    return z, p_two, p_one


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="path to weather_classification_data.csv")
    ap.add_argument("--boot", type=int, default=2000, help="number of bootstrap resamples (default 2000)")
    ap.add_argument("--splits", type=int, default=10, help="extra random splits for the robustness check")
    ap.add_argument("--out", default="outputs", help="folder for tables and figures")
    args = ap.parse_args()

    np.random.seed(SEED)
    sns.set_theme(style="whitegrid")
    out = args.out
    os.makedirs(out, exist_ok=True)

    # 1-2. Load and clean ---------------------------------------------------------------
    df, pct_cols = clean(load_dataset(args.data))
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    print("\nNumeric features:", num_cols, "\nCategorical features:", cat_cols)

    # 3. EDA ----------------------------------------------------------------------------
    describe(df, y, num_cols, out)

    # 4. One stratified held-out test set used by BOTH models ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")
    print(pd.DataFrame({"train %": y_train.value_counts(normalize=True).round(3),
                        "test %": y_test.value_counts(normalize=True).round(3)}).to_string())

    # 5A. Baseline: default Decision Tree -------------------------------------------------
    dt_model = make_dt(num_cols, cat_cols).fit(X_train, y_train)
    tree = dt_model.named_steps["clf"]
    print(f"\nDecision Tree trained. Depth: {tree.get_depth()} | Leaves: {tree.get_n_leaves()}")

    # 5B. Improved: Random Forest tuned with 5-fold stratified CV on the training set only
    param_grid = {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 20],
                  "clf__min_samples_leaf": [1, 2]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(make_rf(num_cols, cat_cols), param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    search.fit(X_train, y_train)
    rf_model = search.best_estimator_
    best_rf_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    print(f"Best RF parameters: {best_rf_params} | mean 5-fold CV macro F1 = {search.best_score_:.4f}")

    # 6. Evaluation on the same test set -----------------------------------------------
    y_pred_dt, y_pred_rf = dt_model.predict(X_test), rf_model.predict(X_test)
    labels = sorted(y.unique())
    point = pd.DataFrame({"Decision Tree": metrics(y_test, y_pred_dt),
                          "Random Forest": metrics(y_test, y_pred_rf)})
    point["Difference (RF - DT)"] = point["Random Forest"] - point["Decision Tree"]
    point.round(4).to_csv(f"{out}/point_metrics.csv")
    print("\nPoint estimates on the held-out test set:\n", point.round(4).to_string())
    print("\n=== Decision Tree ===\n", classification_report(y_test, y_pred_dt, digits=4))
    print("=== Random Forest ===\n", classification_report(y_test, y_pred_rf, digits=4))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, pred, name in zip(axes, [y_pred_dt, y_pred_rf], ["Decision Tree", "Random Forest"]):
        ConfusionMatrixDisplay(confusion_matrix(y_test, pred, labels=labels),
                               display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name} - confusion matrix")
    plt.tight_layout(); plt.savefig(f"{out}/confusion_matrices.png", dpi=150); plt.close()

    # 7. Sampling and confidence intervals ----------------------------------------------
    correct_dt = (y_pred_dt == y_test.to_numpy())
    correct_rf = (y_pred_rf == y_test.to_numpy())
    n = len(y_test)
    x_dt, x_rf = int(correct_dt.sum()), int(correct_rf.sum())
    acc_dt, acc_rf = x_dt / n, x_rf / n
    diff = acc_rf - acc_dt

    lo_dt, hi_dt, se_dt = normal_ci(acc_dt, n)
    lo_rf, hi_rf, se_rf = normal_ci(acc_rf, n)
    lo_d, hi_d, se_d = diff_ci(acc_dt, acc_rf, n, n)
    print(f"\nTest set size n = {n}. Correct: Decision Tree {x_dt}, Random Forest {x_rf}")
    print(f"Decision Tree accuracy {acc_dt:.4f}  SE {se_dt:.4f}  95% CI [{lo_dt:.4f}, {hi_dt:.4f}]")
    print(f"Random Forest accuracy {acc_rf:.4f}  SE {se_rf:.4f}  95% CI [{lo_rf:.4f}, {hi_rf:.4f}]")
    print(f"Difference (RF - DT)   {diff:+.4f}  SE {se_d:.4f}  95% CI [{lo_d:+.4f}, {hi_d:+.4f}]")

    # Bootstrap sampling: how much would accuracy change with a different test sample?
    rng = np.random.default_rng(SEED)
    b_dt, b_rf = bootstrap_accuracy(correct_dt, correct_rf, args.boot, rng)
    b_diff = b_rf - b_dt
    bci = {k: np.percentile(v, [2.5, 97.5]) for k, v in
           {"dt": b_dt, "rf": b_rf, "diff": b_diff}.items()}
    print(f"\nBootstrap (B = {args.boot}) 95% CIs: DT [{bci['dt'][0]:.4f}, {bci['dt'][1]:.4f}] | "
          f"RF [{bci['rf'][0]:.4f}, {bci['rf'][1]:.4f}] | "
          f"difference [{bci['diff'][0]:+.4f}, {bci['diff'][1]:+.4f}]")

    ci_table = pd.DataFrame([
        {"Measure": "Accuracy - Decision Tree", "Estimate": acc_dt, "SE": se_dt,
         "Normal 95% CI": f"[{lo_dt:.4f}, {hi_dt:.4f}]", "Bootstrap 95% CI": f"[{bci['dt'][0]:.4f}, {bci['dt'][1]:.4f}]"},
        {"Measure": "Accuracy - Random Forest", "Estimate": acc_rf, "SE": se_rf,
         "Normal 95% CI": f"[{lo_rf:.4f}, {hi_rf:.4f}]", "Bootstrap 95% CI": f"[{bci['rf'][0]:.4f}, {bci['rf'][1]:.4f}]"},
        {"Measure": "Difference (RF - DT)", "Estimate": diff, "SE": se_d,
         "Normal 95% CI": f"[{lo_d:+.4f}, {hi_d:+.4f}]", "Bootstrap 95% CI": f"[{bci['diff'][0]:+.4f}, {bci['diff'][1]:+.4f}]"},
    ]).set_index("Measure").round(4)
    ci_table.to_csv(f"{out}/confidence_intervals.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    sns.histplot(b_dt, ax=axes[0], color="tab:orange", label="Decision Tree", kde=True, stat="density")
    sns.histplot(b_rf, ax=axes[0], color="tab:blue", label="Random Forest", kde=True, stat="density")
    axes[0].set_title(f"Bootstrap distribution of accuracy (B = {args.boot})"); axes[0].legend()
    sns.histplot(b_diff, ax=axes[1], kde=True, color="tab:green", stat="density")
    axes[1].axvline(0, color="red", ls="--", label="No difference (0)")
    for v in bci["diff"]:
        axes[1].axvline(v, color="black", ls=":")
    axes[1].set_title("Bootstrap distribution of accuracy difference (RF - DT)"); axes[1].legend()
    plt.tight_layout(); plt.savefig(f"{out}/bootstrap_accuracy.png", dpi=150); plt.close()

    # 8. Hypothesis test ------------------------------------------------------------------
    # H0: p_RF = p_DT (same accuracy)   H1: p_RF != p_DT   alpha = 0.05
    z, p_two, p_one = two_proportion_z_test(x_dt, n, x_rf, n)
    decision = "Reject H0" if p_two < ALPHA else "Fail to reject H0"
    print(f"\nTwo-proportion z-test: z = {z:.3f}, two-sided p = {p_two:.3g}, one-sided p = {p_one:.3g}")
    print(f"Critical value at alpha = {ALPHA}: |z| > {Z95:.2f}  ->  {decision}")
    pd.DataFrame([{"Test": "Two-proportion z-test", "z": z, "p (two-sided)": p_two,
                   "p (one-sided, RF better)": p_one, "Decision": decision}]
                 ).round(4).to_csv(f"{out}/hypothesis_test.csv", index=False)

    # 9. Robustness: repeat on different random splits (descriptive only) ------------------
    rep = []
    for s in range(args.splits):
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, stratify=y, random_state=100 + s)
        dt_s = make_dt(num_cols, cat_cols).fit(Xa, ya)
        rf_s = make_rf(num_cols, cat_cols, **best_rf_params).fit(Xa, ya)
        rep.append({"split": s, "DT accuracy": accuracy_score(yb, dt_s.predict(Xb)),
                    "RF accuracy": accuracy_score(yb, rf_s.predict(Xb))})
    rep = pd.DataFrame(rep)
    if len(rep):
        rep["RF - DT"] = rep["RF accuracy"] - rep["DT accuracy"]
        rep.round(4).to_csv(f"{out}/repeated_splits.csv", index=False)
        print(f"\nRF better in {(rep['RF - DT'] > 0).sum()} of {len(rep)} splits; mean difference = "
              f"{rep['RF - DT'].mean():+.4f}")

    # Feature importance
    feat_names = rf_model.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(rf_model.named_steps["clf"].feature_importances_, index=feat_names).sort_values(ascending=False)
    imp.round(4).to_csv(f"{out}/feature_importance.csv")
    plt.figure(figsize=(8, 6)); imp.head(15)[::-1].plot.barh(color="tab:blue")
    plt.title("Top 15 Random Forest feature importances")
    plt.tight_layout(); plt.savefig(f"{out}/feature_importance.png", dpi=150); plt.close()

    # 10. Final table and conclusion -----------------------------------------------------
    results = pd.DataFrame({
        "Decision Tree": [f"{acc_dt:.4f}", f"[{lo_dt:.4f}, {hi_dt:.4f}]"],
        "Random Forest": [f"{acc_rf:.4f}", f"[{lo_rf:.4f}, {hi_rf:.4f}]"],
        "Difference (RF - DT)": [f"{diff:+.4f}", f"[{lo_d:+.4f}, {hi_d:+.4f}]"]},
        index=["Accuracy", "95% CI"])
    results.to_csv(f"{out}/final_results_table.csv")
    print("\nFinal results:\n", results.to_string())

    lines = [f"Random Forest accuracy {acc_rf:.4f} vs Decision Tree {acc_dt:.4f} "
             f"(difference {diff:+.4f}, 95% CI [{lo_d:+.4f}, {hi_d:+.4f}]).",
             f"Two-proportion z-test: z = {z:.3f}, p = {p_two:.3g} -> {decision} at alpha = {ALPHA}."]
    if decision == "Reject H0" and diff > 0:
        lines.append("CONCLUSION: The Random Forest is significantly better; the difference is unlikely "
                     "to be due to variation in the test sample.")
    elif decision == "Reject H0":
        lines.append("CONCLUSION: The models differ significantly, in favour of the Decision Tree.")
    else:
        lines.append("CONCLUSION: We fail to reject H0. The difference is small enough that it could be due to "
                     "variation in the test sample, so we cannot say the Random Forest is really better.")
    conclusion = "\n".join(lines)
    print("\n" + conclusion)
    with open(f"{out}/conclusion.txt", "w", encoding="utf-8") as f:
        f.write(conclusion)
    print(f"\nAll tables and figures saved in '{out}/'.")


if __name__ == "__main__":
    main()
