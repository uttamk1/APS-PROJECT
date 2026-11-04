"""
Exploratory Data Analysis - Kaggle Weather Type Classification dataset
=======================================================================
Produces black-and-white figures and CSV tables in eda_outputs/.

Usage
  python eda.py                                   # uses weather_classification_data.csv
  python eda.py --data path/to/file.csv --out eda_outputs
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

TARGET = "Weather Type"
ORDER = ["Sunny", "Cloudy", "Rainy", "Snowy"]
GREYS = ["0.9", "0.65", "0.4", "0.15"]           # one grey per class
HATCH = ["", "//", "..", "xx"]
STYLES = ["-", "--", "-.", ":"]


def fmt_p(p):
    return "< 1e-300" if p == 0 else f"{p:.1e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="weather_classification_data.csv")
    ap.add_argument("--out", default="eda_outputs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sns.set_theme(style="whitegrid", palette=["0.2"])
    plt.rcParams.update({"axes.edgecolor": "0.3", "font.size": 10})

    df = pd.read_csv(a.data)
    num = df.select_dtypes("number").columns.tolist()
    cat = [c for c in df.columns if c not in num and c != TARGET]

    # 1. Structure and data quality ------------------------------------------------------
    print("Shape:", df.shape)
    print("Missing values:", int(df.isna().sum().sum()), "| duplicate rows:", int(df.duplicated().sum()))
    quality = pd.DataFrame({
        "Check": ["Humidity > 100%", "Precipitation > 100%", "Temperature > 50 °C",
                  "Atmospheric pressure < 950 hPa", "Atmospheric pressure > 1050 hPa"],
        "Rows": [(df["Humidity"] > 100).sum(), (df["Precipitation (%)"] > 100).sum(),
                 (df["Temperature"] > 50).sum(), (df["Atmospheric Pressure"] < 950).sum(),
                 (df["Atmospheric Pressure"] > 1050).sum()]})
    quality["% of data"] = (quality["Rows"] / len(df) * 100).round(1)
    quality.to_csv(f"{a.out}/data_quality.csv", index=False)
    print(quality.to_string(index=False))

    # 2. Descriptive statistics and outliers (1.5 x IQR rule) ------------------------------
    rows = []
    for c in num:
        s = df[c]; q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
        rows.append({"Feature": c, "Mean": s.mean(), "Median": s.median(), "Std": s.std(),
                     "Min": s.min(), "Q1": q1, "Q3": q3, "Max": s.max(), "Skewness": s.skew(),
                     "Kurtosis": s.kurt(), "IQR outliers": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())})
    desc = pd.DataFrame(rows).set_index("Feature").round(2)
    desc.to_csv(f"{a.out}/descriptive_statistics.csv")
    print(desc.to_string())

    # 3. Class balance --------------------------------------------------------------------
    counts = df[TARGET].value_counts().reindex(ORDER)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(counts.index, counts.values, color="0.6", edgecolor="black")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 40, f"{v} ({v / len(df):.0%})", ha="center", fontsize=9)
    ax.set_ylabel("Rows"); ax.set_title("Class distribution"); ax.set_ylim(0, counts.max() * 1.15)
    plt.tight_layout(); plt.savefig(f"{a.out}/class_distribution.png", dpi=150); plt.close()

    # 4. Distributions of numeric features (raw), impossible zones hatched ------------------
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, c in zip(axes.ravel(), num):
        ax.hist(df[c], bins=40, color="0.55", edgecolor="0.25", linewidth=0.4)
        ax.axvline(df[c].mean(), color="black", ls="-", lw=1.2, label="mean")
        ax.axvline(df[c].median(), color="black", ls="--", lw=1.2, label="median")
        if c in ("Humidity", "Precipitation (%)"):
            ax.axvspan(100, df[c].max() + 1, facecolor="none", edgecolor="black", hatch="///", lw=0, label="impossible (>100%)")
        if c == "Atmospheric Pressure":
            ax.set_yscale("log")                     # extreme values are rare; log scale shows them
        ax.set_title(f"{c}  (skew {df[c].skew():.2f})"); ax.legend(fontsize=7)
    for ax in axes.ravel()[len(num):]:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(f"{a.out}/histograms.png", dpi=150); plt.close()

    # 5. Numeric features by weather type -----------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, c in zip(axes.ravel(), num):
        sns.boxplot(data=df, x=TARGET, y=c, order=ORDER, ax=ax, color="0.85", linecolor="black",
                    flierprops={"marker": ".", "markersize": 2, "markerfacecolor": "0.3"})
        ax.set_title(c); ax.set_xlabel("")
    for ax in axes.ravel()[len(num):]:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(f"{a.out}/boxplots_by_class.png", dpi=150); plt.close()

    means = df.groupby(TARGET)[num].mean().reindex(ORDER).T.round(2)
    means.to_csv(f"{a.out}/class_means.csv")
    print("\nMean of each feature by weather type:\n", means.to_string())

    # Temperature density by class (key separator between Rainy and Snowy)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for cls, ls in zip(ORDER, STYLES):
        sns.kdeplot(df.loc[df[TARGET] == cls, "Temperature"], ax=ax, color="black", ls=ls, label=cls, lw=1.6)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_title("Temperature distribution by weather type"); ax.legend()
    plt.tight_layout(); plt.savefig(f"{a.out}/temperature_by_class.png", dpi=150); plt.close()

    # 6. Statistical tests: does each feature differ between weather types? ---------------
    tests = []
    for c in num:
        groups = [g[c].values for _, g in df.groupby(TARGET)]
        f = stats.f_oneway(*groups); h = stats.kruskal(*groups)
        ss_between = sum(len(g) * (g.mean() - df[c].mean()) ** 2 for g in groups)
        eta2 = ss_between / ((df[c] - df[c].mean()) ** 2).sum()      # share of variance explained by the class
        tests.append({"Feature": c, "Test": "ANOVA F / Kruskal-Wallis H", "Statistic": f"{f.statistic:.0f} / {h.statistic:.0f}",
                      "p-value": fmt_p(max(f.pvalue, h.pvalue)), "Effect size": f"eta^2 = {eta2:.2f}"})
    for c in cat:
        ct = pd.crosstab(df[c], df[TARGET])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        tests.append({"Feature": c, "Test": f"Chi-square (df = {dof})", "Statistic": f"{chi2:.0f}",
                      "p-value": fmt_p(p), "Effect size": f"Cramer's V = {v:.2f}"})
    pd.DataFrame(tests).to_csv(f"{a.out}/association_tests.csv", index=False)
    print("\n", pd.DataFrame(tests).to_string(index=False))

    # 7. Categorical features vs weather type (row percentages) ----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, c in zip(axes, cat):
        ct = pd.crosstab(df[c], df[TARGET], normalize="index")[ORDER] * 100
        ct.to_csv(f"{a.out}/crosstab_{c.replace(' ', '_').lower()}.csv")
        left = np.zeros(len(ct))
        for cls, g, h in zip(ORDER, GREYS, HATCH):
            ax.barh(ct.index, ct[cls], left=left, color=g, edgecolor="black", hatch=h, label=cls, lw=0.5)
            left += ct[cls].values
        ax.set_title(f"{c}: share of each weather type (%)"); ax.set_xlim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", frameon=False)
    plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(f"{a.out}/categorical_vs_class.png", dpi=150); plt.close()

    # 8. Correlation between numeric features --------------------------------------------
    corr = df[num].corr()
    corr.round(2).to_csv(f"{a.out}/correlation.csv")
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Greys", vmin=-1, vmax=1, linewidths=0.5, linecolor="white")
    plt.title("Pearson correlation between numeric features")
    plt.tight_layout(); plt.savefig(f"{a.out}/correlation_heatmap.png", dpi=150); plt.close()

    print(f"\nFigures and tables saved in '{a.out}/'.")


if __name__ == "__main__":
    main()
