"""
Model Development Pipeline
Trains classical ML and neural network models, compares them, and saves the best.
Run with:   python model_main.py
Output:     models/best_model.pkl, models/scaler.pkl, visualizations as PNG
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.preprocessing import split_data, impute_missing, treat_outliers, encode_categorical, add_features, scale_data
from src.train_classical import compare_models
from src.train_neural import train_neural_network, save_model, compare_activations
from src.evaluate import run_evaluation

def run_pca(X_train, y_train):
    """Perform PCA analysis and generate visualizations."""
    os.makedirs('models', exist_ok=True)
    
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
    plt.savefig('models/pca_scree.png', bbox_inches='tight')
    plt.close()
    
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
    plt.savefig('models/pca_2d.png', bbox_inches='tight')
    plt.close()
    
    print(f"PCA: {sum(cumulative < 0.90) + 1} components needed for 90% variance")


def main():
    """Main pipeline: load → preprocess → train → evaluate → save."""
    
    # Load data
    print("Loading dataset...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "dataset.csv")
    
    df = pd.read_csv(data_path, header=None)
    df.columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income"
    ]
    
    df["income"] = (df["income"].str.strip() == ">50K").astype(int)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    # Preprocessing
    print("Preprocessing...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, "income")
    X_train, X_val, X_test, num_cols, cat_cols = impute_missing(X_train, X_val, X_test)
    X_train, X_val, X_test, _ = treat_outliers(X_train, X_val, X_test, num_cols)
    X_train, X_val, X_test = encode_categorical(X_train, X_val, X_test, cat_cols)
    X_train, X_val, X_test = add_features(X_train, X_val, X_test)
    X_train, X_val, X_test, scaler = scale_data(X_train, X_val, X_test)
    
    X_train, X_val, X_test = X_train.fillna(0), X_val.fillna(0), X_test.fillna(0)
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}\n")
    
    # PCA analysis
    print("Running PCA...")
    run_pca(X_train, y_train)
    print()
    
    # Convert to numpy
    feature_names = X_train.columns.tolist()
    X_train_np, X_val_np, X_test_np = X_train.values, X_val.values, X_test.values
    y_train_np, y_val_np, y_test_np = y_train.values, y_val.values, y_test.values
    
    # Train models
    print("Training classical models...")
    ml_model1, ml_model2, best_classical, best_classical_name, feature_names = compare_models(
        X_train_np, X_val_np, X_test_np,
        y_train_np, y_val_np, y_test_np,
        feature_names=feature_names
    )
    print()
    
    print("Training neural network...")
    nn_model, _ = train_neural_network(
        X_train_np, y_train_np,
        X_val_np, y_val_np,
        X_test_np, y_test_np,
        epochs=100, lr=0.001
    )
    print()
    
    # Compare activation functions
    print("Comparing activation functions...")
    compare_activations(X_train_np, y_train_np, X_val_np, y_val_np, epochs=100, lr=0.001)
    print()
    
    # Save models
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_classical, 'models/classical_model.pkl')
    joblib.dump(best_classical, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    save_model(nn_model, 'models/neural_network.pt')
    print(f"Saved: classical_model.pkl, best_model.pkl ({best_classical_name}), scaler.pkl, neural_network.pt\n")
    
    # Evaluate
    print("Evaluating models...")
    models = {best_classical_name: best_classical, "Neural Network": nn_model}
    run_evaluation(models, X_test_np, y_test_np, feature_names=feature_names)
    
    print("\nDone! Ready for agent deployment.")


if __name__ == "__main__":
    main()