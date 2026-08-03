import os
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib

from langchain_core.tools import Tool

# PATHS INITIALIZATION #
MODEL_PATH  = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# ORIGINAL FEATURE LIST (in the same order as the HW1 training data) #
FEATURE_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country"
]

# LOAD THE MODEL AND SCALER #
_model  = joblib.load(MODEL_PATH)
_scaler = joblib.load(SCALER_PATH)

print(f">> Model was loaded from:  {MODEL_PATH}")
print(f">> Scaler was loaded from: {SCALER_PATH}")

# PREPROCESSING #
def _preprocess(input_dict: dict) -> np.ndarray:
    row = {col: input_dict.get(col, np.nan) for col in FEATURE_COLUMNS}
    df = pd.DataFrame([row])
    
    # Identify categorical columns BEFORE conversion
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Convert to numeric (categorical strings → NaN)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Impute missing numeric values using scaler means
    for i, col in enumerate(FEATURE_COLUMNS):
        if df[col].isna().any():
            df[col] = df[col].fillna(_scaler.mean_[i])

    # Feature engineering
    df["capital_diff"] = df["capital_gain"] - df["capital_loss"]
    df["hours_log"]    = np.log1p(df["hours_per_week"].clip(lower=0))

    # One-hot encode categorical columns (to match training)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    
    # Align columns with training data
    # The scaler expects features in a specific order
    # We ensure missing columns are added as 0
    n_features_expected = _scaler.n_features_in_
    if df.shape[1] < n_features_expected:
        # Pad with zeros if needed
        missing_cols = n_features_expected - df.shape[1]
        for _ in range(missing_cols):
            df[f'padding_{_}'] = 0
    elif df.shape[1] > n_features_expected:
        # Take only the first n features
        df = df.iloc[:, :n_features_expected]
    
    return _scaler.transform(df)

# PREDICTION FUNCTION AND TOOL #
def predict_income(input_data: str) -> str:
    try:
        input_dict  = json.loads(input_data) if isinstance(input_data, str) else input_data
        X           = _preprocess(input_dict)
        prediction  = _model.predict(X)[0]
        probability = _model.predict_proba(X)[0][1]
        label = ">50K" if prediction == 1 else "<=50K"
        return f"Prediction: {label} (probability of earning >50K: {probability * 100:.1f}%)"
    except json.JSONDecodeError:
        return "Error: Invalid JSON input."
    except Exception as e:
        return f"Error during prediction: {str(e)}"

prediction_tool = Tool(
    name="IncomePrediction",
    func=predict_income,
    description=(
        "Predicts whether a person earns more or less than $50K/year. "
        "Input must be a JSON string with any of these fields: "
        "age, workclass, fnlwgt, education, education_num, marital_status, "
        "occupation, relationship, race, sex, capital_gain, capital_loss, "
        "hours_per_week, native_country. "
        "Missing fields are imputed automatically. "
        "Output: prediction label and probability of earning >50K."
    ),
)

# TASK'S 5 (BONUS) - (1) DATASET STATISTICS AND (2) CSV LOOKUP TOOLS #
DATASET_PATH = "data/dataset.csv"

DATASET_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

try:
    _df = pd.read_csv(DATASET_PATH, header=None, names=DATASET_COLUMNS)
    _df["income"] = _df["income"].astype(str).str.strip()
    print(f">> Dataset loaded: {len(_df)} rows")
except (FileNotFoundError, PermissionError, IsADirectoryError):
    _df = None
    print(f">> Dataset not found at '{DATASET_PATH}' - bonus tools disabled.")

# BONUS TOOL 1: DATASET STATISTICS #
def dataset_stats(column: str) -> str:
    if _df is None:
        return f"Error: Dataset not found at '{DATASET_PATH}'."
    column = column.strip().lower()
    if column not in _df.columns:
        return f"Column '{column}' not found. Available: {', '.join(_df.columns)}"
    series = _df[column]
    if pd.api.types.is_numeric_dtype(series):
        stats = series.describe()
        return (
            f"Statistics for '{column}':\n"
            f"  Count:  {int(stats['count']):,}\n"
            f"  Mean:   {stats['mean']:.2f}\n"
            f"  Median: {series.median():.2f}\n"
            f"  Std:    {stats['std']:.2f}\n"
            f"  Min:    {stats['min']:.2f}\n"
            f"  Max:    {stats['max']:.2f}"
        )
    else:
        counts = series.value_counts()
        lines = [f"Statistics for '{column}' (categorical):"]
        lines.append(f"  Unique values: {series.nunique()}")
        lines.append(f"  Most common: '{counts.index[0].strip()}' ({counts.iloc[0]:,} times)")
        lines.append(f"  Distribution:")
        for val, cnt in counts.head(5).items():
            pct = cnt / len(series) * 100
            lines.append(f"    {str(val).strip():<30} {cnt:>6,} ({pct:.1f}%)")
        return "\n".join(lines)

dataset_stats_tool = Tool(
    name="DatasetStatistics",
    func=dataset_stats,
    description=(
        "Returns summary statistics for any column in the UCI Adult Census dataset. "
        "Use this when the user asks numerical or distributional questions about the dataset. "
        "Input: a single column name as a string. "
        "Available columns: age, workclass, fnlwgt, education, education_num, "
        "marital_status, occupation, relationship, race, sex, "
        "capital_gain, capital_loss, hours_per_week, native_country, income."
    ),
)

# BONUS TOOL 2: CSV LOOKUP #
def csv_lookup(query: str) -> str:
    if _df is None:
        return f"Error: Dataset not found at '{DATASET_PATH}'."
 
    try:
        # Parse the query into column, operator, value
        import re
        match = re.match(r"([\w_]+)\s*(>=|<=|==|>|<)\s*(.+)", query.strip())
        if not match:
            return (
                "Invalid query format. Use: 'column operator value'\n"
                "Examples: 'age > 60', 'education_num == 13', 'hours_per_week >= 50'\n"
                f"Available columns: {', '.join(_df.columns)}"
            )
 
        column, operator, value = match.groups()
        column = column.strip()
        value  = value.strip()
 
        if column not in _df.columns:
            return f"Column '{column}' not found. Available: {', '.join(_df.columns)}"
 
        # Convert value to numeric if possible
        try:
            value = float(value)
        except ValueError:
            pass  # keep as string for categorical columns
 
        # Apply filter
        ops = {">": "__gt__", "<": "__lt__", ">=": "__ge__", "<=": "__le__", "==": "__eq__"}
        filtered = _df[getattr(_df[column], ops[operator])(value)]
 
        if filtered.empty:
            return f"No rows found where {column} {operator} {value}."
 
        # Return a concise summary
        total = len(filtered)
        income_dist = filtered["income"].value_counts().to_dict()
        avg_age     = filtered["age"].mean()
        avg_edu     = filtered["education_num"].mean()
        avg_hours   = filtered["hours_per_week"].mean()
 
        return (
            f"Filter: {column} {operator} {value}\n"
            f"Matching rows: {total:,} out of {len(_df):,} ({total/len(_df)*100:.1f}%)\n"
            f"Income distribution: {income_dist}\n"
            f"Average age: {avg_age:.1f}\n"
            f"Average education (years): {avg_edu:.1f}\n"
            f"Average hours/week: {avg_hours:.1f}"
        )
 
    except Exception as e:
        return f"Error processing query: {str(e)}"
 
# LangChain Tool
csv_lookup_tool = Tool(
    name="CSVLookup",
    func=csv_lookup,
    description=(
        "Filters the UCI Adult Census dataset and returns a summary of matching rows. "
        "Use this when the user wants to explore a specific subset of the data, "
        "such as people above a certain age, with a specific education level, etc. "
        "Input format: 'column operator value' where operator is one of: >, <, >=, <=, ==. "
        "Examples: 'age > 60', 'education_num >= 13', 'hours_per_week == 40'. "
        "Available columns: age, fnlwgt, education_num, capital_gain, capital_loss, "
        "hours_per_week, workclass, education, marital_status, occupation, "
        "relationship, race, sex, native_country, income."
    ),
)

# TESTING THE TOOL INDEPENDENTLY #
if __name__ == "__main__":
    test_cases = [
        # Likely >50K: older, highly educated, high capital gain
        '{"age": 52, "education_num": 16, "hours_per_week": 50, "capital_gain": 15000, "capital_loss": 0}',
        # Likely <=50K: young, less educated, no capital activity
        '{"age": 25, "education_num": 9, "hours_per_week": 35, "capital_gain": 0, "capital_loss": 0}',
    ]

    print("\n=============================")
    print("  Task 2 - Prediction Tool Test")
    print("===============================\n")

    for case in test_cases:
        print(f"Input:  {case}")
        print(f"Output: {predict_income(case)}\n")