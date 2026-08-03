import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. TRAIN / VALIDATION / TEST SPLIT

def split_data(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    # 10% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=42
    )

    # 80% train, 10% val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=10/90, stratify=y_temp, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# 2. DATA CLEANING / MISSING VALUES

def impute_missing(X_train, X_val, X_test):

    X_train = X_train.replace(["?","n/a","N/A","", " ", "NA", "null", "NULL"], np.nan)
    X_val   = X_val.replace(["?","n/a","N/A","", " ", "NA", "null", "NULL"], np.nan)
    X_test  = X_test.replace(["?","n/a","N/A","", " ", "NA", "null", "NULL"], np.nan)

   # Identify categorical columns before conversion
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Only coerce the numeric columns - categorical columns stay as strings
    # so encode_categorical() has real values to one-hot encode later
    X_train[num_cols] = X_train[num_cols].apply(pd.to_numeric, errors="coerce")
    X_val[num_cols]   = X_val[num_cols].apply(pd.to_numeric, errors="coerce")
    X_test[num_cols]  = X_test[num_cols].apply(pd.to_numeric, errors="coerce")

    # Fill numeric with medians (computed ONLY from the training set)
    for col in num_cols:
        median_val = X_train[col].median()
        if pd.isna(median_val):
            median_val = 0
        X_train[col] = X_train[col].fillna(median_val)
        X_val[col]   = X_val[col].fillna(median_val)
        X_test[col]  = X_test[col].fillna(median_val)

    # Fill categorical with mode (computed ONLY from the training set)
    for col in cat_cols:
        mode_series = X_train[col].mode()
        mode_val = mode_series[0] if len(mode_series) > 0 else "Unknown"
        X_train[col] = X_train[col].fillna(mode_val)
        X_val[col]   = X_val[col].fillna(mode_val)
        X_test[col]  = X_test[col].fillna(mode_val)

    return X_train, X_val, X_test, num_cols, cat_cols

# 3. IQR

def treat_outliers(X_train, X_val, X_test, num_cols):

    for col in num_cols:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        X_train[col] = X_train[col].clip(lower, upper)
        X_val[col]   = X_val[col].clip(lower, upper)
        X_test[col]  = X_test[col].clip(lower, upper)

    return X_train, X_val, X_test, None

# 4. CATEGORICAL ENCODING

def encode_categorical(X_train, X_val, X_test, cat_cols):
    """Apply one-hot encoding to categorical variables (fit on training set only)"""
    
    if len(cat_cols) == 0:
        return X_train, X_val, X_test
    
    # One-hot encode categorical columns (drop_first to avoid multicollinearity)
    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True, dtype=int)
    X_val   = pd.get_dummies(X_val, columns=cat_cols, drop_first=True, dtype=int)
    X_test  = pd.get_dummies(X_test, columns=cat_cols, drop_first=True, dtype=int)

    # Ensure all three sets have the same columns (handles unseen categories in val/test)
    train_cols = set(X_train.columns)
    val_cols = set(X_val.columns)
    test_cols = set(X_test.columns)
    
    all_cols = train_cols | val_cols | test_cols
    
    for col in all_cols:
        if col not in X_train.columns:
            X_train[col] = 0
        if col not in X_val.columns:
            X_val[col] = 0
        if col not in X_test.columns:
            X_test[col] = 0
    
    # Reorder to match training set column order
    X_val = X_val[X_train.columns]
    X_test = X_test[X_train.columns]
    
    return X_train, X_val, X_test

# 5. FEATURE ENGINEERING

def add_features(X_train, X_val, X_test):

    X_train["capital_diff"] = X_train["capital_gain"] - X_train["capital_loss"]
    X_val["capital_diff"]   = X_val["capital_gain"] - X_val["capital_loss"]
    X_test["capital_diff"]  = X_test["capital_gain"] - X_test["capital_loss"]

    # Handle zeros before log transformation
    X_train["hours_log"] = np.log1p(X_train["hours_per_week"].clip(lower=0))
    X_val["hours_log"]   = np.log1p(X_val["hours_per_week"].clip(lower=0))
    X_test["hours_log"]  = np.log1p(X_test["hours_per_week"].clip(lower=0))

    return X_train, X_val, X_test

# 6. SCALING

def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), 
        columns=X_val.columns, 
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns, 
        index=X_test.index
    )

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler