"""
Exploratory PCA Analysis (Standalone)
Loads raw data, applies preprocessing, then runs PCA analysis.
Run with:   python src/pca_analysis.py
Output:     models/pca_scree.png, models/pca_2d.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from preprocessing import split_data, impute_missing, treat_outliers, encode_categorical, add_features, scale_data


def run_pca_analysis():
    """Load data, preprocess, and run PCA analysis."""
    
    # Load data
    print("Loading dataset...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "dataset.csv")
    
    df = pd.read_csv(data_path, header=None)
    df.columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income"
    ]
    
    df["income"] = (df["income"].str.strip() == ">50K").astype(int)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    # Preprocessing (same as model_main.py)
    print("Preprocessing...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, "income")
    X_train, X_val, X_test, num_cols, cat_cols = impute_missing(X_train, X_val, X_test)
    X_train, X_val, X_test, _ = treat_outliers(X_train, X_val, X_test, num_cols)
    X_train, X_val, X_test = encode_categorical(X_train, X_val, X_test, cat_cols)
    X_train, X_val, X_test = add_features(X_train, X_val, X_test)
    X_train, X_val, X_test, scaler = scale_data(X_train, X_val, X_test)
    
    X_train, X_val, X_test = X_train.fillna(0), X_val.fillna(0), X_test.fillna(0)
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}\n")
    
    # PCA Analysis
    print("Running PCA analysis...\n")
    os.makedirs(os.path.join(base_dir, "..", "models"), exist_ok=True)
    models_dir = os.path.join(base_dir, "..", "models")
    
    pca_full = PCA(random_state=42)
    pca_full.fit(X_train)
    
    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    
    # Scree plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, len(explained) + 1), explained, alpha=0.6, label='Per component')
    ax.plot(range(1, len(explained) + 1), cumulative, marker='o', color='red', label='Cumulative')
    ax.axhline(0.90, color='gray', linestyle='--', label='90% threshold')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA Scree Plot')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'pca_scree.png'), bbox_inches='tight')
    plt.close()
    print("Scree plot saved to models/pca_scree.png")
    
    # 2D projection
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_train)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, marker, color in [(0, 'o', 'steelblue'), (1, '^', 'tomato')]:
        mask = y_train.values == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   label=f'Income {"<=50K" if label == 0 else ">50K"}',
                   alpha=0.3, s=10, marker=marker, color=color)
    ax.set_xlabel(f'PC1 ({pca_full.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca_full.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title('PCA 2D Projection by Income Class')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'pca_2d.png'), bbox_inches='tight')
    plt.close()
    print("2D projection saved to models/pca_2d.png")
    
    # PCA Loadings (Top features per component)
    feature_names = X_train.columns.tolist()
    print("\n" + "*"*60)
    print("TOP FEATURES BY PRINCIPAL COMPONENT")
    print("*"*60)
    
    for pc_idx in range(3):
        loadings = pd.Series(
            np.abs(pca_full.components_[pc_idx]),
            index=feature_names
        ).sort_values(ascending=False).head(5)
        print(f"\nPC{pc_idx + 1} (explains {explained[pc_idx]:.1%} variance):")
        print(loadings.round(3).to_string())
    
    n_90 = sum(cumulative < 0.90) + 1
    print(f"\n" + "*"*60)
    print(f"Components needed for 90% variance: {n_90}")
    print("*"*60 + "\n")
    
    print("PCA analysis complete!")


if __name__ == "__main__":
    run_pca_analysis()