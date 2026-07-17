import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MODEL_PATH, METRICS_PATH, ENCODER_PATH
from utils.preprocessing import preprocess_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score


def train_kmeans_exploratory(X, encoder, n_clusters=22):
    print(f"\n{'-' * 40}")
    print("Exploratory: K-Means Clustering")
    print(f"{'-' * 40}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    explained_var = pca.explained_variance_ratio_.sum()

    cluster_info = {
        "n_clusters": n_clusters,
        "inertia": round(kmeans.inertia_, 2),
        "pca_explained_variance": round(explained_var, 4),
        "n_features": X.shape[1],
        "cluster_centers": kmeans.cluster_centers_.tolist(),
        "pca_components": pca.components_.tolist(),
    }

    print(f"  Number of clusters: {n_clusters}")
    print(f"  Inertia: {kmeans.inertia_:.2f}")
    print(f"  PCA 2D explained variance: {explained_var:.2%}")
    print(f"  Cluster label distribution: {np.bincount(cluster_labels).tolist()}")

    return kmeans, pca, X_pca, cluster_labels, cluster_info


def train_and_evaluate_models():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, encoder, df = preprocess_pipeline()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    best_model = None
    best_accuracy = 0
    best_name = ""

    print("\n" + "=" * 60)
    print("SUPERVISED MODEL TRAINING AND EVALUATION")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n{'-' * 40}")
        print(f"Training: {name}")
        print(f"{'-' * 40}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        results[name] = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "cv_mean": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  CV Mean:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name

    full_X = pd.concat([X_train, X_test])
    kmeans, pca, X_pca, cluster_labels, cluster_info = train_kmeans_exploratory(full_X, encoder)

    print(f"\n{'=' * 60}")
    print(f"BEST SUPERVISED MODEL: {best_name} with Accuracy: {best_accuracy:.4f}")
    print(f"(K-Means is for exploratory clustering only; not used for prediction)")
    print(f"{'=' * 60}")

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    pca_data = {
        "X_pca": X_pca.tolist(),
        "cluster_labels": cluster_labels.tolist(),
    }
    joblib.dump(pca_data, os.path.join(os.path.dirname(MODEL_PATH), "pca_data.pkl"))

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Encoder saved to: {ENCODER_PATH}")

    results["best_model"] = best_name
    results["kmeans"] = cluster_info
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Metrics saved to: {METRICS_PATH}")

    return results, best_model, encoder, df


if __name__ == "__main__":
    train_and_evaluate_models()
