# Income Prediction: ML Pipeline to RAG-Powered Conversational Agent

A complete end-to-end machine learning and AI system. **Stage 1** trains and compares classical ML and neural network models for income prediction. **Stage 2** wraps the best of these models in a conversational AI agent that answers domain questions, makes predictions, and explores the dataset.

---

## Table of Contents

1. [Problem Description](#1-problem-description)
2. [Dataset Description](#2-dataset-description)
3. [Preprocessing Approach](#3-preprocessing-approach)
4. [Feature Engineering](#4-feature-engineering)
5. [PCA Insights](#5-pca-insights)
6. [Model Comparison](#6-model-comparison)
7. [Best Model Designation](#7-best-model-designation)
8. [Agent System Overview](#8-agent-system-overview)
9. [Architecture](#9-architecture)
10. [Knowledge Base](#10-knowledge-base)
11. [Model Integration](#11-model-integration)
12. [Example Conversations](#12-example-conversations)
13. [Installation & Execution](#13-installation--execution)
14. [Example API Calls](#14-example-api-calls)
15. [Project Structure](#15-project-structure)
16. [Known Limitations & Future Work](#16-known-limitations--future-work)
17. [Summary](#17-summary)

---

## 1. Problem Description

The dataset used in this project is called _UCI Adult Census Data_. It falls within the domain of Social-Demographic data and concerns the average income of adult individuals. This is a **binary classification** problem: determine whether someone earns more than $50,000/year or not. The target variable is _income_.

**Why this problem matters:** Income prediction using personal data helps understand the factors behind income accumulation, which is valuable for governmental decisions and policy making.

**Problem Type:** Binary Classification

**Target Variable:** `income` (=<50K or >50K)

---

## 2. Dataset Description

**Source:** UCI Machine Learning Repository via Kaggle  
**URL:** https://www.kaggle.com/datasets/sagnikpatra/uci-adult-census-data-dataset

**Size:** 32,561 rows × 15 columns (14 features + 1 target)

**Features:**

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `age` | Continuous | Age in years |
| 2 | `workclass` | Categorical | Employment type (8 categories) |
| 3 | `fnlwgt` | Continuous | Census sampling weight |
| 4 | `education` | Categorical | Education level (16 categories) |
| 5 | `education_num` | Continuous | Years of education (1-16) |
| 6 | `marital_status` | Categorical | Marital status (7 categories) |
| 7 | `occupation` | Categorical | Job type (14 categories) |
| 8 | `relationship` | Categorical | Family role (6 categories) |
| 9 | `race` | Categorical | Race (5 categories) |
| 10 | `sex` | Categorical | Gender (Male/Female) |
| 11 | `capital_gain` | Continuous | Capital gains income |
| 12 | `capital_loss` | Continuous | Capital losses |
| 13 | `hours_per_week` | Continuous | Weekly work hours |
| 14 | `native_country` | Categorical | Country of origin (42 categories) |
| 15 | `income` | **Target** | =<50K or >50K |

**Target Distribution:**

| Class | Count | Percentage |
|-------|-------|------------|
| =<50K | 24,720 | 75.9% |
| >50K | 7,841 | 24.1% |
| Total | 32,561 | 100% |

The dataset is imbalanced (75.9% negative class), requiring stratified splitting and careful metric selection.

---

## 3. Preprocessing Approach

All preprocessing steps were applied strictly in the order: **Split -> Impute -> Treat Outliers -> Encode -> Feature Engineer -> Scale**. All statistics (medians, IQR bounds, scaler parameters, mode values) were computed **exclusively on the training set** and then applied to validation and test sets to prevent data leakage.

### STEP 1: Data Splitting and Stratification

The dataset was split 80/10/10 (train/validation/test) using stratified splitting to preserve class distribution:
- **Training set (80%):** 26,048 rows – used to fit preprocessing and train models
- **Validation set (10%):** 3,256 rows – used for early stopping and model selection
- **Test set (10%):** 3,257 rows – used only for final evaluation

Stratification ensures all three partitions maintain the 75.9%/24.1% class ratio.

### STEP 2: Standardized Cleaning and Missing Value Imputation

Raw data contained multiple missing value representations: "?", "n/a", "N/A", "", " ", "NA", "null", "NULL". These were standardized to NaN.

**Numeric columns:** Filled with the **median** of the training set (robust to outliers).  
**Categorical columns:** Filled with the **mode** of the training set. If all values missing, filled with "Unknown".

**Note:** An earlier implementation converted the entire dataframe (numeric and categorical columns alike) using `pd.to_numeric(errors='coerce')`, which silently turned all 8 categorical columns into NaN before they ever reached the encoding step. This version scopes that conversion to numeric columns only, so categorical values survive intact and get properly one-hot encoded in Step 4.

### STEP 3: Outlier Handling Using IQR Clipping

Outliers detected using the Interquartile Range (IQR) method:
- Lower bound = Q1 − 1.5 * IQR
- Upper bound = Q3 + 1.5 * IQR

Values outside these bounds were **clipped** (Winsorized) rather than removed, preserving all data while reducing extreme value influence. Bounds computed on training set only.

### STEP 4: Categorical Encoding

**Critical Step:** All categorical variables were **one-hot encoded** (drop_first=True to avoid multicollinearity):
- `workclass` (8 categories) → 7 binary columns
- `education` (16 categories) → 15 binary columns
- `marital_status` (7 categories) → 6 binary columns
- `occupation` (14 categories) → 13 binary columns
- `relationship` (6 categories) → 5 binary columns
- `race` (5 categories) → 4 binary columns
- `native_country` (42 categories) → 41 binary columns
- `sex` (binary) → kept as 0/1

After encoding, the feature set grows from 14 raw columns to **102 columns**. This preserves categorical information that a naϊve numeric conversion would otherwise discard. The encoding was fit on the training set; unseen categories in validation/test sets are represented as all zeros.

### STEP 5: Feature Scaling

StandardScaler applied to all numeric and one-hot encoded features, transforming to zero mean and unit variance:
- Fitted on training set only
- Applied identically to validation and test sets
- Scaler saved as `models/scaler.pkl` for use in Stage 2

---

## 4. Feature Engineering

Two new features were created and added alongside original columns:

| # | Feature | Type | Formula |
|---|---------|------|---------|
| 103 | `capital_diff` | Continuous | `capital_gain` - `capital_loss` |
| 104 | `hours_log` | Continuous | `log(1 + hours_per_week)` |

**`capital_diff`:** Combines capital gains and losses into net capital income. A person with $15,000 gain and $5,000 loss differs meaningfully from one with the same gain and $0 loss; a single feature captures this relationship better than two separate features.

**`hours_log`:** Most people work ~40 hours/week, but a small group works 60+ hours. Without transformation, high values dominate. Log scaling compresses the range and makes the distribution more amenable to model learning. The `+1` prevents log(0). As it turns out, this ends up being the single most important feature to our best model (see Section 7) - so this transformation mattered more than we expected going in.

---

## 5. PCA Insights

In an early version of the pipeline, the categorical encoding was problematic, keeping only the cleaned-up numeric features and losing potential information from the rest. This actually led to another model being assigned the label of "best model", which was Random Forest - with relevant details being shown in _Part 6_. After one-hot encoding all 8 categorical variables, we ended up with the 102 columns that were mentioned before.

The scree plot shows that with this many features, no few components dominate the variance anymore. We now need **76 components** to reach the 90% variance threshold, which is a big jump from the few components we needed before the encoding fix. This actually makes sense: instead of a few continuous variables like `hours_per_week` and `education_num` carrying most of the signal, we now have dozens of one-hot columns (individual occupation types, native countries, etc), and each one only contributes a tiny sliver of variance on its own.

This doesn't mean PCA became useless - it just means the categorical features carry a lot of small, spread-out signal rather than a few dominant directions. The 2D scatter plot (PC1 vs PC2) still shows the =<50K and >50K classes overlapping a bit, which confirms what we already suspected: this isn't something a simple linear division on two axes can separate cleanly.

**2D Scatter Plot (PC1 vs PC2):**
- **Class Overlap:** Blue (=<50K) and red (>50K) points heavily overlap, especially in the center.
- **Vertical Banding:** Distinct stripes visible, reflecting the categorical feature structure (education level, occupation, etc. create discrete groupings).

_Note:_ With 102 columns, individual PCA loadings stop being very interpretable one by one, so we didn't try to pick out "top 5 features per component" this time the way we could before. For that, the model's own feature importances (see Section 7) give a much clearer picture of what's actually driving predictions.

---

## 6. Model Comparison

### Classical ML vs Neural Network (Test Set Results)

| Model | Training Accuracy | Validation Accuracy | Validation AUC | Test Accuracy |
|-------|-------------------|---------------------|----------------|---------------|
| Logistic Regression | 0.792 | 0.802 | 0.884 | 0.795 |
| Random Forest | 0.788 | 0.787 | 0.883 | 0.783 |

With the categorical features actually being used this time, both classical models jumped noticeably in performance. And this time, Logistic Regression narrowly edges out Random Forest on Validation AUC (**0.884** vs 0.883), so it's the one selected as the best classical model and saved to `models/classical_model.pkl`.

We now compare the winning classical model against the neural network on the test set:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.795 | 0.548 | 0.843 | 0.664 | 0.894 |
| Neural Network | 0.844 | 0.710 | 0.598 | 0.649 | 0.892 |

_Note #1:_ **Why is Logistic Regression winning this time, and not Random Forest or the Neural Network?** Once the categorical features (especially `marital_status`) were actually available to the model, a simpler linear model turned out to separate the classes almost as well as the more complex ones - and slightly better on AUC. The Neural Network still wins on raw Accuracy (0.844 vs 0.795) and Precision (0.710 vs 0.548), but its Recall is much lower (0.598 vs 0.843), meaning it misses a lot more of the actual >50K earners. Logistic Regression catches far more of them, which matters more for a task like this.

_Note #2:_ This result flips what we found in our earlier run, where Random Forest came out on top. It's a good reminder of how much a preprocessing bug can quietly skew which model looks "best" - the ranking of models isn't fixed, it depends heavily on whether the features actually going into them are meaningful in the first place.

**Does the loss curve suggest the neural network had enough data to learn well?** The training history shows a steady decline in both training and validation loss, with the most rapid improvement happening before epoch 40, and validation accuracy climbing to around 0.82-0.83 by epoch 100. The curves don't show obvious divergence, so the model isn't wildly overfitting, but it also isn't clearly outperforming the much simpler Logistic Regression - which is really the more interesting finding here.

As we see, the added complexity of the Neural Network still isn't clearly justified - Logistic Regression achieves a marginally higher AUC (0.894 vs 0.892) with a much simpler, more interpretable model, and a considerably better Recall.

---

## 7. Best Model Designation

The best model selected is **Logistic Regression**, saved in `models/best_model.pkl`.

**_Justification summary_**:

- Highest ROC-AUC (0.894), narrowly ahead of both Random Forest (0.883) and the Neural Network (0.892)
- Highest Recall (0.843), meaning it correctly identifies ~84% of high-income individuals - by far the best of the three models on this metric
- Simple, interpretable, and considerably cheaper to train and deploy than the Neural Network

Looking at the model's own feature importances (absolute coefficient magnitude), the top 5 features are:

| Feature | Importance (abs. coefficient) |
|---------|-------------------------------|
| `hours_log` | 2.416 |
| `hours_per_week` | 1.883 |
| `marital_status_Married-civ-spouse` | 1.174 |
| `education_num` | 0.591 |
| `age` | 0.464 |

It's worth pointing out that `marital_status_Married-civ-spouse` shows up as the 3rd most important feature here - this is a categorical feature that didn't even exist in the model before the encoding fix, so this is fairly direct evidence that the fix mattered, and not just a number moving around on a table.

This model is carried forward to Stage 2.

---

## 8. Agent System Overview

The conversational AI agent is built on top of the best model from Stage 1. It can:
- **Answer domain questions** using a knowledge base (RAG system)
- **Make income predictions** using the Logistic Regression model
- **Explore the dataset** with statistical and CSV lookup tools
- **Maintain conversation memory** within sessions

**LLM Backbone:** OpenAI GPT-4o-mini  
**Agent Framework:** LangGraph with `create_react_agent`  
**API Framework:** FastAPI with Pydantic validation  
**Session Management:** Conversation history persisted per `session_id`

---

## 9. Architecture

### Tool Definitions

| Tool | Name | Triggered When |
|------|------|----------------|
| RAG Retrieval | `KnowledgeBaseRetrieval` | User asks factual/conceptual questions about income factors |
| Income Prediction | `IncomePrediction` | User provides personal attributes and requests a prediction |
| Dataset Statistics | `DatasetStatistics` | User asks for numerical summaries (mean, median, distribution) |
| CSV Lookup | `CSVLookup` | User wants to filter and explore dataset subsets |

### LangGraph Flow

```
User Message 
    ->
LLM Decides Which Tool (if any)
    ->
Tool Executes (RAG/Prediction/Stats/Lookup)
    ->
LLM Formats Response
    ->
Response Sent to User
    ->
Conversation Saved to Memory (per session_id)
```

Conversation memory is handled by `MemorySaver`, which persists chat history per `thread_id` (mapped from `session_id`). Memory exists only within a session; it resets on application restart.

---

## 10. Knowledge Base

Ten domain-specific documents stored in `data/documents/`, chunked and embedded into ChromaDB at `data/vector_store/`:

| # | Document | Type | Why Chosen |
|---|----------|------|------------|
| 1 | Statistical Approach to Adult Census Income Level Prediction | PDF | Description of the relevant dataset and ML logic |
| 2 | Comparative Study of ML Algorithms on Census Income Data | PDF | Model comparison for income classification |
| 3 | Causes and Consequences of Income Inequality | PDF | Socio-economic context |
| 4 | Chartbook of Economic Inequality | PDF | Income distribution statistics |
| 5 | Income Inequality in the United States | PDF | US-specific income factors |
| 6 | Income in the United States | PDF | Census Bureau statistics |
| 7 | Minimum Wages, Inequality and the Informal Sector | PDF | Labor market factors |
| 8 | US Income 2024 (U.S. Census Bureau) | PDF | Current income benchmarks |
| 9 | Dictionary of Economics | PDF | Domain terminology |
| 10 | Model Development Summary | TXT | Stage 1 methodology and results |

**Chunking & Retrieval:**
- Chunk size: 800 characters
- Chunk overlap: 150 characters
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: ChromaDB
- Retrieval method: MMR (Maximal Marginal Relevance)

**MMR Parameters:**
- `k=5`: Return 5 final chunks
- `fetch_k=15`: Consider 15 candidates before selecting
- `lambda_mult=0.4`: Bias toward diversity (0.4 = 40% diversity, 60% relevance)

MMR prevents the agent from retrieving repetitive paragraphs from a single document. It balances semantic relevance with information diversity.

---

## 11. Model Integration

The Logistic Regression classifier from Stage 1 (ROC-AUC: 0.894, Recall: 0.843) is wrapped as a LangChain tool. The preprocessing pipeline replicates Stage 1 exactly:

1. Build single-row DataFrame from input JSON, fill missing keys with NaN
2. Coerce numeric fields to numeric (categorical strings are left as-is, not converted)
3. Apply one-hot encoding for categorical variables
4. Impute any remaining missing numeric values
5. Apply feature engineering (`capital_diff`, `hours_log`)
6. Reindex columns against the saved training-time column order (`models/feature_columns.pkl`), filling any missing one-hot columns with 0
7. Scale using saved `scaler.pkl` and predict using `best_model.pkl`

**Tool Input:** JSON string with any subset of 14 UCI Adult Census features  
**Tool Output:** Prediction label (>50K or =<50K) + probability

**Categorical Feature Handling:** The LLM acts as an intelligent translator, mapping conversational descriptions (e.g., "electrical engineer" → `occupation_Exec-managerial`) into correct numeric features before calling the prediction tool.

---

## 12. Example Conversations

### (a) RAG Retrieval Response

```
User:  "What factors typically influence whether a person earns more than $50K?"

Agent: Several factors typically influence whether a person earns more than $50K
       per year, including education level, age, occupation, hours worked per week,
       and capital gains. Research confirms that higher education and more years of
       work experience are among the strongest predictors of higher income.
```

### (b) Prediction Response

```
User:  "Would a 52-year-old with 16 years of education and a capital gain
        of 15000 earn more than $50K?"

Agent: Based on the trained Logistic Regression model, this person is predicted to
       earn more than $50K.
```

### (c) Memory and Follow-up Turn

```
User:  "What if they only had 9 years of education instead?"

Agent: With only 9 years of education (and keeping all other attributes the same),
       the model now predicts a lower probability of earning above $50K.
```

### (d) Dataset Statistics Response

```
User:  "What is the average age of people in the dataset?"

Agent: The average age of individuals in the UCI Adult Census dataset is
       approximately 38.6 years, with a median of 37 years, a minimum of
       17 and a maximum of 90.
```

### (e) CSV Lookup Response

```
User:  "How many people work more than 60 hours per week?"

Agent: There are 1,120 individuals in the dataset who work more than 60 hours
       per week. Among this subset, approximately 42.5% earn more than $50K/year.
```

---

## 13. Installation & Execution

### Requirements

- Python 3.8+
- OpenAI API key (for Stage 2 only)
- Git

### Stage 1: Model Training

**STEP 1: Clone the Repository**

```bash
git clone https://github.com/davidtel999/Income-Prediction-ML-RAG-Agent.git
cd Income-Prediction-ML-RAG-Agent
```

**STEP 2: Create and activate virtual environment**

```bash
python -m venv .venv
```

**Windows:**
```bash
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

**STEP 3: Install dependencies**

```bash
pip install -r requirements.txt
```

**STEP 4: Train the model**

```bash
python model_main.py
```

This generates:
- `models/best_model.pkl` (all-around best classifier)
- `models/scaler.pkl` (preprocessing scaler)
- `models/classical_model.pkl` (best classical model)
- `models/feature_columns.pkl` (training-time column order, used for consistent predictions)
- `models/neural_network.pt` (neural network for comparison)
- `models/confusion_matrix_*.png` (visualization)
- `models/loss_curves_neural_network.png` (neural network training curves)
- `models/pca_scree.png` (PCA variance plot)
- `models/pca_2d.png` (PCA 2D projection)

### Stage 2: Conversational Agent

**STEP 1: Set your OpenAI API key**

**Windows:**
```powershell
$env:OPENAI_API_KEY = "sk-proj-..."
```

**Mac/Linux:**
```bash
export OPENAI_API_KEY="sk-proj-..."
```

**STEP 2: Build the vector store (one-time)**

This loads your documents, splits them into chunks, and creates embeddings. The first run also downloads the embedding model, so it can take **5-10 minutes** depending on your hardware and the number of documents. You won't see much output while it works.

```bash
python -m src.rag
```

When it finishes, a `data/vector_store/` folder appears in the project. You only need to run this again if you add, remove or edit documents in `data/documents/`. Each rerun replaces the old store, so nothing is duplicated.

**STEP 3: Start the agent**

```bash
uvicorn agent_main:app --reload
```

Visit: **http://127.0.0.1:8000/** (redirects to Swagger UI at `/docs`)

---

## 14. Example API Calls

### Using `curl` - Windows:

```powershell
curl -X POST "http://127.0.0.1:8000/chat" `
     -H "Content-Type: application/json" `
     -d '{"message": "Would a 52-year-old with 16 years of education and a capital gain of 15000 earn more than 50K?", "session_id": "user_001"}'
```

### Using `curl` - Mac/Linux:

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "Would a 52-year-old with 16 years of education and a capital gain of 15000 earn more than 50K?", "session_id": "user_001"}'
```

### Using Python `requests`:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/chat",
    json={
        "message": "Would a 52-year-old with 16 years of education and a capital gain of 15000 earn more than 50K?",
        "session_id": "user_001"
    }
)

print(response.json()["response"])
```

### Expected Response:

```json
{
  "response": "Based on the trained Logistic Regression model, this person is predicted to earn more than $50K."
}
```

---

## 15. Project Structure

```
├── data/
│   ├── dataset.csv                    (UCI Adult Census raw data)
│   ├── documents/                     (RAG knowledge base)
│   └── vector_store/                  (ChromaDB embeddings)
├── models/
│   ├── best_model.pkl                 (The winning classifier: Logistic Regression)
│   ├── scaler.pkl                     (StandardScaler)
│   ├── classical_model.pkl            (The winning classical model: Logistic Regression)
│   ├── feature_columns.pkl            (Training-time columns' order)
│   ├── neural_network.pt              (Neural network)
│   ├── confusion_matrix_*.png         (Evaluation visuals)
│   ├── loss_curves_neural_network.png (Training curves)
│   ├── pca_scree.png                  (PCA variance plot)
│   └── pca_2d.png                     (PCA 2D projection)
├── src/
│   ├── preprocessing.py               (Cleaning, encoding, scaling)
│   ├── train_classical.py             (Logistic Regression, Random Forest)
│   ├── train_neural.py                (Neural network + activation comparison)
│   ├── evaluate.py                    (Metrics, confusion matrices, loss curves)
│   ├── pca_analysis.py                (Standalone PCA exploratory script)
│   ├── rag.py                         (Knowledge base & retrieval)
│   ├── agent.py                       (LangGraph agent logic)
│   ├── tools.py                       (Prediction, stats, lookup tools)
│   └── api.py                         (FastAPI endpoints)
├── model_main.py                      (Stage 1: Training pipeline)
├── agent_main.py                      (Stage 2: Agent server)
├── requirements.txt                   (Dependencies)
└── README.md                          (Current file)
```

---

## 16. Known Limitations & Future Work

1. **Recursion Limit:** The agent uses LangGraph's default recursion limit. For complex multi-step reasoning, consider setting `max_iterations` explicitly.

2. **Hyperparameter Tuning:** Logistic Regression and Random Forest both use fixed hyperparameters. RandomizedSearchCV could explore optimal combinations and potentially push ROC-AUC beyond the current 0.894.

3. **Neural Network Depth:** The current architecture (2 hidden layers, 64→32 units) is intentionally shallow. Given that Logistic Regression already matches or beats the Neural Network on AUC after the categorical encoding fix, a deeper architecture wasn't pursued - the added complexity doesn't seem justified by the data at this scale.

4. **PCA Interpretability:** With 102 features after one-hot encoding, individual component loadings are harder to interpret one by one compared to the smaller numeric-only feature set. Feature importances from the trained model (Section 7) are a more direct way to understand what's driving predictions at this feature count.

---

## 17. Summary

**Stage 1** builds a complete ML pipeline with proper preprocessing (including working categorical encoding), feature engineering, and model comparison. Logistic Regression emerges as the best model for this income prediction task - outmatching Random Forest and a shallow Neural Network.

**Stage 2** uses the previous model for a conversational AI agent, combining it with a knowledge base (RAG) to create an interactive system that answers questions, makes predictions, and explores the dataset.

Together, these stages demonstrate how classical ML and modern LLM-driven agents can be integrated to build sophisticated systems.
