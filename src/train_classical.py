import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

def compare_models(X_train, X_val, X_test, y_train, y_val, y_test, feature_names=None):
   
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    results = []

    ml_model1 = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    ml_model1.fit(X_train, y_train)
    val_auc1 = roc_auc_score(y_val, ml_model1.predict_proba(X_val)[:, 1])
    results.append({
        "Model": "Logistic Regression",
        "Training Accuracy": round(accuracy_score(y_train, ml_model1.predict(X_train)), 3),
        "Validation Accuracy": round(accuracy_score(y_val, ml_model1.predict(X_val)), 3),
        "Validation AUC": round(val_auc1, 3),
        "Test Accuracy": round(accuracy_score(y_test, ml_model1.predict(X_test)), 3)
    })

    ml_model2 = RandomForestClassifier(n_estimators=100, max_depth=10,
                                    class_weight='balanced', random_state=42)
    ml_model2.fit(X_train, y_train)
    val_auc2 = roc_auc_score(y_val, ml_model2.predict_proba(X_val)[:, 1])
    results.append({
        "Model": "Random Forest",
        "Training Accuracy": round(accuracy_score(y_train, ml_model2.predict(X_train)), 3),
        "Validation Accuracy": round(accuracy_score(y_val, ml_model2.predict(X_val)), 3),
        "Validation AUC": round(val_auc2, 3),
        "Test Accuracy": round(accuracy_score(y_test, ml_model2.predict(X_test)), 3)
    })

    print("\n" + "*"*60)
    print("CLASSICAL ML COMPARISON" + "\n")
    print(pd.DataFrame(results).to_string(index=False))

    os.makedirs('models', exist_ok=True)
    if val_auc2 > val_auc1:
        best_model, best_name = ml_model2, "Random Forest"
    else:
        best_model, best_name = ml_model1, "Logistic Regression"

    joblib.dump(best_model, 'models/classical_model.pkl')
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"\nBest classical model ({best_name}) saved to models/classical_model.pkl")
    print(f"Also saved to models/best_model.pkl (pending neural network comparison)")

    return ml_model1, ml_model2, best_model, best_name, feature_names

if __name__ == "__main__":
    print("This file is meant to be imported by main.py")