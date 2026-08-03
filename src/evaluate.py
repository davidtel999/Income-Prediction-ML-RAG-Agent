import numpy as np
import pandas as pd
import torch
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

def run_evaluation(models_dict, X_test, y_test, feature_names=None):

    results = []
    os.makedirs('models', exist_ok=True)

    print("\n" + "*"*60)
    print("EVALUATION RESULTS")
    print("*"*60)

    for name, model in models_dict.items():
        if isinstance(model, torch.nn.Module):
            model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X_test.copy())
                y_prob = model(X_t).numpy().flatten()
                y_pred = (y_prob >= 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall":    recall_score(y_test, y_pred, zero_division=0),
            "F1":        f1_score(y_test, y_pred, zero_division=0),
            "AUC":       roc_auc_score(y_test, y_prob)
        }
        results.append(metrics)

        cm = confusion_matrix(y_test, y_pred)
        print(f"\n{name}:")
        print(f"  Accuracy:  {metrics['Accuracy']:.3f} | Precision: {metrics['Precision']:.3f} | "
              f"Recall: {metrics['Recall']:.3f} | F1: {metrics['F1']:.3f} | AUC: {metrics['AUC']:.3f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
        print(f"    FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")

        # Save confusion matrix as PNG
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=16)
        
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(f"Confusion Matrix - {name}", fontsize=14, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["≤50K", ">50K"])
        ax.set_yticklabels(["≤50K", ">50K"])
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(f'models/confusion_matrix_{name.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Confusion matrix saved to models/confusion_matrix_{name.lower().replace(' ', '_')}.png")

    df = pd.DataFrame(results).set_index("Model")
    print("\n" + "*"*60)
    print("COMPARISON TABLE")
    print("*"*60)
    print(df.round(3).to_string())

    if feature_names is not None:
        print("\n" + "*"*60)
        print("FEATURE IMPORTANCES")
        print("*"*60)
        for name, model in models_dict.items():
            if hasattr(model, 'feature_importances_'):
                scores = pd.Series(model.feature_importances_, index=feature_names)
                print(f"\n{name} — Top 5 Features (importance):")
                print(scores.sort_values(ascending=False).head(5).round(3).to_string())
            elif hasattr(model, 'coef_'):
                scores = pd.Series(np.abs(model.coef_[0]), index=feature_names)
                print(f"\n{name} — Top 5 Features (absolute coefficient):")
                print(scores.sort_values(ascending=False).head(5).round(3).to_string())

    best_name = df["AUC"].idxmax()
    best_model = models_dict[best_name]

    print("\n" + "*"*60)
    print(f"BEST MODEL: {best_name} (AUC = {df.loc[best_name, 'AUC']:.3f})")
    print("*"*60)

    os.makedirs('models', exist_ok=True)
    if isinstance(best_model, torch.nn.Module):
        torch.save(best_model.state_dict(), 'models/best_model.pt')
        print("Saved to models/best_model.pt")
    else:
        joblib.dump(best_model, 'models/best_model.pkl')
        print("Saved to models/best_model.pkl")

    return df, best_name


def plot_loss_curves(train_losses, val_losses, model_name="Neural Network"):
    """Plot and save training/validation loss curves as PNG"""
    
    os.makedirs('models', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label='Training Loss', marker='o', markersize=3)
    ax.plot(val_losses, label='Validation Loss', marker='s', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{model_name} - Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'models/loss_curves_{model_name.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to models/loss_curves_{model_name.lower().replace(' ', '_')}.png")