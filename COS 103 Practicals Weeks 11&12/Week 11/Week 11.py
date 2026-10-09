"""
Week 11: Data Analysis — Iris Dataset Exploratory Data Analysis
Requirements: pandas, matplotlib, seaborn (optional; not used), numpy
Run: python iris_eda.py
Place Iris.csv in the same folder as this script.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the dataset
df = pd.read_csv("Iris.csv")

# 2. Inspect the data
print("First five rows:")
print(df.head())
print("\nDataset shape (rows, columns):", df.shape)
print("\nColumn information:")
print(df.info())
print("\nMissing values in each column:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# 3. Summary statistics for numerical variables
numeric_columns = [
    "SepalLengthCm", "SepalWidthCm",
    "PetalLengthCm", "PetalWidthCm"
]
print("\nSummary statistics:")
print(df[numeric_columns].describe().round(2))

# 4. Frequency distribution of species
print("\nSpecies frequency:")
print(df["Species"].value_counts())

# 5. Mean measurements for each species
print("\nMean measurements by species:")
print(df.groupby("Species")[numeric_columns].mean().round(2))

# 6. Histograms: distributions of the four measurements
df[numeric_columns].hist(bins=12, figsize=(10, 7), edgecolor="black")
plt.suptitle("Distributions of Iris Measurements")
plt.tight_layout()
plt.show()

# 7. Bar chart: number of observations in each species
df["Species"].value_counts().sort_index().plot(
    kind="bar", figsize=(7, 4), edgecolor="black"
)
plt.title("Number of Flowers by Species")
plt.xlabel("Species")
plt.ylabel("Count")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 8. Boxplots: compare measurements across species
for column in numeric_columns:
    df.boxplot(column=column, by="Species", figsize=(7, 4))
    plt.title(f"{column} by Species")
    plt.suptitle("")
    plt.xlabel("Species")
    plt.ylabel("Centimetres")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

# 9. Scatter plot: petal length against petal width
for species, group in df.groupby("Species"):
    plt.scatter(
        group["PetalLengthCm"], group["PetalWidthCm"], label=species, alpha=0.8
    )
plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# 10. Correlation matrix and heatmap
correlation = df[numeric_columns].corr()
print("\nCorrelation matrix:")
print(correlation.round(2))
plt.imshow(correlation, vmin=-1, vmax=1)
plt.xticks(range(len(numeric_columns)), numeric_columns, rotation=35, ha="right")
plt.yticks(range(len(numeric_columns)), numeric_columns)
for i in range(len(numeric_columns)):
    for j in range(len(numeric_columns)):
        plt.text(j, i, f"{correlation.iloc[i, j]:.2f}", ha="center", va="center")
plt.colorbar(label="Correlation coefficient")
plt.title("Correlation Between Measurements")
plt.tight_layout()
plt.show()
