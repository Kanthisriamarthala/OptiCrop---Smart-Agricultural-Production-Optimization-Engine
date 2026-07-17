import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import sys
import logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
from config import IMAGES_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(IMAGES_DIR, exist_ok=True)

PCA_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "pca_data.pkl")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

CHART_FILES = [
    "crop_distribution.png",
    "npk_distribution.png",
    "temperature_distribution.png",
    "humidity_distribution.png",
    "rainfall_distribution.png",
    "ph_distribution.png",
    "correlation_heatmap.png",
    "boxplots.png",
    "pairplot.png",
    "kmeans_clusters.png",
]


def charts_exist_on_disk():
    for fname in CHART_FILES:
        path = os.path.join(IMAGES_DIR, fname)
        if not os.path.exists(path) or os.path.getsize(path) < 100:
            return False
    return True


def get_existing_chart_paths():
    paths = {}
    for fname in CHART_FILES:
        path = os.path.join(IMAGES_DIR, fname)
        if os.path.exists(path) and os.path.getsize(path) >= 100:
            key = fname.replace(".png", "")
            paths[key] = f"images/{fname}"
    return paths


def save_plot(filename):
    path = os.path.join(IMAGES_DIR, filename)
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return f"images/{filename}"


def safe_plot(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Chart generation failed in {func.__name__}: {e}")
            plt.close("all")
            return None
    return wrapper


@safe_plot
def plot_crop_distribution(df):
    plt.figure()
    counts = df["label"].value_counts()
    colors = sns.color_palette("Set2", len(counts))
    plt.bar(counts.index, counts.values, color=colors)
    plt.title("Crop Distribution in Dataset", fontsize=16, fontweight="bold")
    plt.xlabel("Crop", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 1, str(v), ha="center", fontsize=10)
    return save_plot("crop_distribution.png")


@safe_plot
def plot_nutrient_distribution(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    nutrients = ["N", "P", "K"]
    colors = ["#4CAF50", "#FF9800", "#2196F3"]
    for i, nutrient in enumerate(nutrients):
        axes[i].hist(df[nutrient], bins=20, color=colors[i], edgecolor="black", alpha=0.7)
        axes[i].set_title(f"{nutrient} Distribution", fontsize=14, fontweight="bold")
        axes[i].set_xlabel(nutrient, fontsize=12)
        axes[i].set_ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    return save_plot("npk_distribution.png")


@safe_plot
def plot_temperature_distribution(df):
    plt.figure()
    plt.hist(df["temperature"], bins=20, color="#FF5722", edgecolor="black", alpha=0.7)
    plt.title("Temperature Distribution", fontsize=16, fontweight="bold")
    plt.xlabel("Temperature (°C)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.axvline(df["temperature"].mean(), color="red", linestyle="--", label=f"Mean: {df['temperature'].mean():.1f}°C")
    plt.legend()
    return save_plot("temperature_distribution.png")


@safe_plot
def plot_humidity_distribution(df):
    plt.figure()
    plt.hist(df["humidity"], bins=20, color="#2196F3", edgecolor="black", alpha=0.7)
    plt.title("Humidity Distribution", fontsize=16, fontweight="bold")
    plt.xlabel("Humidity (%)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.axvline(df["humidity"].mean(), color="red", linestyle="--", label=f"Mean: {df['humidity'].mean():.1f}%")
    plt.legend()
    return save_plot("humidity_distribution.png")


@safe_plot
def plot_rainfall_distribution(df):
    plt.figure()
    plt.hist(df["rainfall"], bins=20, color="#9C27B0", edgecolor="black", alpha=0.7)
    plt.title("Rainfall Distribution", fontsize=16, fontweight="bold")
    plt.xlabel("Rainfall (mm)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.axvline(df["rainfall"].mean(), color="red", linestyle="--", label=f"Mean: {df['rainfall'].mean():.1f} mm")
    plt.legend()
    return save_plot("rainfall_distribution.png")


@safe_plot
def plot_ph_distribution(df):
    plt.figure()
    plt.hist(df["ph"], bins=20, color="#4CAF50", edgecolor="black", alpha=0.7)
    plt.title("pH Distribution", fontsize=16, fontweight="bold")
    plt.xlabel("pH", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.axvline(df["ph"].mean(), color="red", linestyle="--", label=f"Mean: {df['ph'].mean():.2f}")
    plt.legend()
    return save_plot("ph_distribution.png")


@safe_plot
def plot_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix Heatmap", fontsize=16, fontweight="bold")
    return save_plot("correlation_heatmap.png")


@safe_plot
def plot_boxplots(df):
    numeric_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        df.boxplot(column=col, by="label", ax=axes[i])
        axes[i].set_title(f"{col} by Crop", fontsize=10)
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=45, labelsize=8)
    if len(numeric_cols) < len(axes):
        for j in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[j])
    plt.suptitle("")
    plt.tight_layout()
    return save_plot("boxplots.png")


@safe_plot
def plot_pairplot(df):
    numeric_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    sample = df.sample(min(500, len(df)), random_state=42)
    g = sns.pairplot(sample[numeric_cols], diag_kind="kde", plot_kws={"alpha": 0.5})
    g.fig.suptitle("Pairplot of Numerical Features", y=1.02, fontsize=16, fontweight="bold")
    return save_plot("pairplot.png")


@safe_plot
def plot_kmeans_clusters(df):
    if not os.path.exists(PCA_DATA_PATH):
        logger.warning("pca_data.pkl not found. Skipping K-Means cluster plot.")
        return None
    pca_data = joblib.load(PCA_DATA_PATH)
    X_pca = np.array(pca_data["X_pca"])
    cluster_labels = np.array(pca_data["cluster_labels"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    scatter1 = axes[0].scatter(
        X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="tab20",
        alpha=0.7, s=30, edgecolors="k", linewidth=0.3
    )
    axes[0].set_title("K-Means Clusters (PCA-reduced)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Principal Component 1", fontsize=11)
    axes[0].set_ylabel("Principal Component 2", fontsize=11)
    cbar1 = fig.colorbar(scatter1, ax=axes[0], ticks=range(int(cluster_labels.max()) + 1))
    cbar1.set_label("Cluster", fontsize=10)

    encoder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "label_encoder.pkl")
    if os.path.exists(encoder_path):
        encoder = joblib.load(encoder_path)
        labels_encoded = encoder.transform(df["label"])
        n_crops = len(encoder.classes_)
        scatter2 = axes[1].scatter(
            X_pca[:, 0], X_pca[:, 1], c=labels_encoded, cmap="tab20",
            alpha=0.7, s=30, edgecolors="k", linewidth=0.3
        )
        axes[1].set_title("Actual Crop Labels (PCA-reduced)", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Principal Component 1", fontsize=11)
        axes[1].set_ylabel("Principal Component 2", fontsize=11)
        cbar2 = fig.colorbar(scatter2, ax=axes[1], ticks=range(n_crops))
        cbar2.set_label("Crop", fontsize=10)
    else:
        axes[1].text(0.5, 0.5, "Encoder not available", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Actual Crop Labels", fontsize=14, fontweight="bold")

    plt.suptitle("K-Means Exploratory Clustering (for visualization only)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    return save_plot("kmeans_clusters.png")


def generate_all_charts(df):
    if charts_exist_on_disk():
        paths = get_existing_chart_paths()
        if len(paths) == len(CHART_FILES):
            logger.info("All charts already exist on disk, skipping regeneration.")
            return paths
    paths = {}
    chart_funcs = [
        ("crop_distribution", plot_crop_distribution),
        ("npk_distribution", plot_nutrient_distribution),
        ("temperature_distribution", plot_temperature_distribution),
        ("humidity_distribution", plot_humidity_distribution),
        ("rainfall_distribution", plot_rainfall_distribution),
        ("ph_distribution", plot_ph_distribution),
        ("correlation_heatmap", plot_correlation_heatmap),
        ("boxplots", plot_boxplots),
        ("pairplot", plot_pairplot),
        ("kmeans_clusters", plot_kmeans_clusters),
    ]
    for key, func in chart_funcs:
        result = func(df)
        if result is not None:
            paths[key] = result
    return paths
