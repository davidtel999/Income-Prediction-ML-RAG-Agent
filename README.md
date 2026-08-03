# Income Prediction: ML Pipeline to RAG-Powered Conversational Agent

A complete end-to-end machine learning and AI system. **Stage 1** trains and compares classical ML and neural network models for income prediction. **Stage 2** wraps the best model in a conversational AI agent that answers domain questions, makes predictions, and explores the dataset.

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

---

## 1. Problem Description

The dataset used in this project is called _UCI Adult Census Data_. It falls within the domain of Social-Demographic data and concerns the average income of adult individuals. This is a **binary classification** problem: determine whether someone earns more than $50,000/year or not. The target variable is _income_.

**Why this problem matters:** Income prediction using personal data helps understand the factors behind income accumulation, which is valuable for governmental decisions and policy making.

**Problem Type:** Binary Classification

**Target Variable:** `income` (≤50K or >50K)

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
| 15 | `income` | **Target** | ≤50K or >50K |

**Target Distribution:**

| Class | Count | Percentage |
|-------|-------|------------|
| ≤50K | 24,720 | 75.9% |
| >50K | 7,841 | 24.1% |
| Total | 32,561 | 100% |

The dataset is imbalanced (75.9% negative class), requiring stratified splitting and careful metric selection.

---

## 3. Preprocessing Approach

All preprocessing steps were applied strictly in the order: **Split -> Impute -> Treat Outliers -> Encode -> Feature Engineer -> Scale**. All statistics (medians, IQR bounds, scaler parameters, mode values) were computed **exclusively on the training set** and then applied to validation and test sets to prevent data leakage.

### STEP 1: Data Splitting and Stratification

The dataset was split 80/10/10 (train/validation/test) using stratified splitting to preserve class distribution:
- **Training set (80%):** 26,048 rows — used to fit preprocessing and train models
- **Validation set (10%):** 3,256 rows — used for early stopping and model selection
- **Test set (10%):** 3,257 rows — used only for final evaluation

Stratification ensures all three partitions maintain the 75.9%/24.1% class ratio.

### STEP 2: Standardized Cleaning and Missing Value Imputation

Raw data contained multiple missing value representations: "?", "n/a", "N/A", "", " ", "NA", "null", "NULL". These were standardized to NaN.

**Numeric columns:** Filled with the **median** of the training set (robust to outliers).  
**Categorical columns:** Filled with the **mode** of the training set. If all values missing, filled with "Unknown".
**Critical fix:** Previous implementation used `pd.to_numeric(errors='coerce')` which converted all categorical columns to NaN and filled with 0, effectively **discarding 8 categorical features**. This version preserves categorical data and encodes it properly (see Step 4).

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

This **preserves all categorical information** that was previously lost. The encoding was fit on the training set; unseen categories in validation/test sets are represented as all zeros.

### STEP 5: Feature Scaling

StandardScaler applied to all numeric and one-hot encoded features, transforming to zero mean and unit variance:
- Fitted on training set only
- Applied identically to validation and test sets
- Scaler saved as `models/scaler.pkl` for use in production (Homework 2)

---

## 4. Feature Engineering

Two new features were created and added alongside original columns:

| # | Feature | Type | Formula |
|---|---------|------|---------|
| 16 | `capital_diff` | Continuous | `capital_gain` - `capital_loss` |
| 17 | `hours_log` | Continuous | `log(1 + hours_per_week)` |

**`capital_diff`:** Combines capital gains and losses into net capital income. A person with $15,000 gain and $5,000 loss differs meaningfully from one with the same gain and $0 loss; a single feature captures this relationship better than two separate features.

**`hours_log`:** Most people work ~40 hours/week, but a small group works 60+ hours. Without transformation, high values dominate. Log scaling compresses the range and makes the distribution more amenable to model learning. The `+1` prevents log(0).

---

## 5. PCA Insights

PCA was run on scaled training data as a mandatory exploratory step.

**Scree Plot Findings:**
- PC1 explains ~25% of variance
- PC1 + PC2 explain ~40% of variance
- First 4 components explain ~80% of variance
- First 6 components explain ~90% of variance

**PC1 Loadings (Top Features):**
The first principal component is dominated by:
1. `hours_per_week` (high positive loading)
2. `hours_log` (high positive loading)
3. `education_num` (high positive loading)

**Interpretation:** Work hours and education are the strongest signals in the data. People who work more hours and have higher education tend to cluster together in PC1 space.

**PC2 Loadings (Top Features):**
The second principal component is dominated by:
1. `fnlwgt` (census weight)
2. `age`
3. `education_num`

**Interpretation:** Demographic factors (age) and sampling characteristics drive PC2 variance.

**2D Scatter Plot (PC1 vs PC2):**
- **Class Overlap:** Blue (≤50K) and red (>50K) points heavily overlap, especially in the center. Clean separation is impossible with linear boundaries.
- **Vertical Banding:** Distinct stripes visible, reflecting the categorical feature structure (education level, occupation, etc. create discrete groupings).
- **Implication:** Non-linear decision boundaries (Random Forest, Neural Networks) are justified to find complex relationships in the data.

---

## 6. Model Comparison

### Classical ML vs Neural Network (Test Set Results)

| Metric | Logistic Regression | Random Forest | Neural Network |
|--------|---------------------|---------------|----------------|
| Accuracy | 0.730 | 0.722 | 0.794 |
| Precision | 0.459 | 0.453 | 0.645 |
| Recall | 0.719 | 0.753 | 0.324 |
| F1-Score | 0.564 | 0.565 | 0.431 |
| ROC-AUC | 0.799 | 0.815 | 0.812 |

### Analysis

**Classical Models Comparison:**
Random Forest and Logistic Regression achieved nearly identical test accuracies (0.722 vs 0.730). However, Random Forest achieved higher validation AUC (0.807 vs 0.799), demonstrating superior class separation. Random Forest is selected as the best classical model.

**Classical vs Neural Network:**
The Neural Network achieved higher accuracy (0.794 vs 0.722) but with critically different trade-offs:
- **Neural Network:** High precision (0.645) but very low recall (0.324). It "plays safe" — when it predicts >50K, it's often correct, but it misses 67.6% of high-income individuals.
- **Random Forest:** Balanced metrics. Catches 75.3% of high-income individuals (recall) while maintaining reasonable precision (0.453).

**Validation Loss Behavior:**
The neural network's validation loss plateaued around epoch 34, after which further training provided diminishing returns. The training set of 26,048 rows was sufficient for basic convergence but insufficient for the model to learn complex non-linear patterns that would justify its architectural complexity.

**Why Random Forest Wins:**
On tabular datasets of this size, classical algorithms like Random Forest consistently outperform simple neural networks. This is well-established empirically. The added complexity of deep learning is not justified by measurable performance gains. Random Forest achieves the best balance of recall, precision, and F1-score.

---

## 7. Best Model Designation

**Best Model:** Random Forest  
**Location:** `models/best_model.pkl`

**Justification:**
- **Highest ROC-AUC (0.815):** Best class separation and discrimination
- **Highest Recall (0.753):** Correctly identifies ~75% of high-income individuals
- **Best F1-Score (0.565):** Best balance of precision and recall
- **Simpler than Neural Network:** Easier to interpret, deploy, and maintain
- **Empirically validated:** Consistent with the principle that tree-based models outperform neural networks on small-to-medium tabular datasets

This model is carried forward to Homework 2 and the Final Project.

---

## 8. Agent System Overview (Homework 2)

The conversational AI agent is built on top of the best model from Stage 1. It can:
- **Answer domain questions** using a knowledge base (RAG system)
- **Make income predictions** using the Random Forest model
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
|------|------|---|
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

11 domain-specific documents stored in `data/documents/`, chunked and embedded into ChromaDB at `data/vector_store/`:

| # | Document | Type | Why Chosen |
|---|----------|------|-----------|
| 1 | Statistical Approach to Adult Census Income Level Prediction | PDF | Covers Stage 1 dataset and ML methodology |
| 2 | Comparative Study of ML Algorithms on Census Income Data | PDF | Model comparison for income classification |
| 3 | Causes and Consequences of Income Inequality | PDF | Socio-economic context |
| 4 | Chartbook of Economic Inequality | PDF | Income distribution statistics |
| 5 | Income Poverty and Valuation of Noncash Benefits (1994) | PDF | Historical income data |
| 6 | Income Inequality in the United States | PDF | US-specific income factors |
| 7 | Income in the United States | PDF | Census Bureau statistics |
| 8 | Minimum Wages, Inequality and the Informal Sector | PDF | Labor market factors |
| 9 | US Income 2024 (U.S. Census Bureau) | PDF | Current income benchmarks |
| 10 | Dictionary of Economics | PDF | Domain terminology |
| 11 | Model Development Summary | TXT | Stage 1 results and methods |

**Chunking & Retrieval:**
- Chunk size: 800 tokens
- Chunk overlap: 150 tokens
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

The Random Forest classifier from Stage 1 is wrapped as a LangChain tool. The preprocessing pipeline replicates Stage 1 exactly:

1. Build single-row DataFrame from input JSON, fill missing keys with NaN
2. Coerce values to numeric (categorical strings → NaN → mode imputation)
3. Apply one-hot encoding for categorical variables
4. Impute numeric missing values using training-set medians
5. Apply feature engineering (`capital_diff`, `hours_log`)
6. Scale using saved `scaler.pkl`
7. Predict using `best_model.pkl`

**Tool Input:** JSON string with any subset of 14 UCI Adult Census features  
**Tool Output:** Prediction label (>50K or ≤50K) + probability

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

Agent: Based on the trained Random Forest model, this person is predicted to
       earn more than $50K, with a probability of 94.0%.
```

### (c) Memory and Follow-up Turn

```
User:  "What if they only had 9 years of education instead?"

Agent: With only 9 years of education (and keeping all other attributes the same),
       the model now predicts ≤$50K, with a probability of earning above $50K
       dropping to 39.6%.
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

- Python 3.8 or higher
- OpenAI API key (for Stage 2 agent only)
- Git

### Stage 1: Model Training

**STEP 1: Clone the Repository**

```bash
git clone https://github.com/yourusername/income-prediction-system.git
cd income-prediction-system
```

**STEP 2: Create and activate virtual environment**

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
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
- `models/best_model.pkl` (Random Forest classifier)
- `models/scaler.pkl` (preprocessing scaler)
- `models/classical_model.pkl` (Logistic Regression for comparison)
- `models/neural_network.pt` (neural network for comparison)
- `models/confusion_matrix_*.png` (visualization)
- `models/loss_curves_*.png` (neural network training curves)
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

```bash
python -m src.rag
```

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
  "response": "Based on the trained Random Forest model, this person is predicted to earn more than $50K, with a probability of 94.0%."
}
```

---

## Project Structure

```
├── data/
│   ├── dataset.csv                    (UCI Adult Census raw data)
│   ├── documents/                     (RAG knowledge base - 11 PDFs + TXT)
│   └── vector_store/                  (ChromaDB embeddings)
├── models/
│   ├── best_model.pkl                 (Random Forest classifier)
│   ├── scaler.pkl                     (StandardScaler)
│   ├── classical_model.pkl            (Logistic Regression)
│   ├── neural_network.pt              (PyTorch neural network)
│   ├── confusion_matrix_*.png         (Evaluation visualizations)
│   ├── loss_curves_*.png              (Training curves)
│   ├── pca_scree.png                  (PCA variance plot)
│   └── pca_2d.png                     (PCA 2D projection)
├── src/
│   ├── preprocessing.py               (Cleaning, encoding, scaling)
│   ├── train_classical.py             (Logistic Regression, Random Forest)
│   ├── train_neural.py                (PyTorch neural network)
│   ├── evaluate.py                    (Metrics, confusion matrices, loss curves)
│   ├── rag.py                         (Knowledge base & retrieval)
│   ├── agent.py                       (LangGraph agent logic)
│   ├── tools.py                       (Prediction, stats, lookup tools)
│   └── api.py                         (FastAPI endpoints)
├── model_main.py                      (Stage 1: Training pipeline)
├── agent_main.py                      (Stage 2: Agent server)
├── requirements.txt                   (Pinned dependencies)
└── README.md                          (Current file)
```

---

## Known Limitations & Future Work

1. **Categorical Feature Encoding:** Previous implementation lost categorical features through improper preprocessing. This version applies one-hot encoding, significantly increasing feature count but preserving information.

2. **Feature Importance Alignment:** The professor noted the importance of documenting how feature importances from Random Forest align with PCA loadings. Both identify work hours, education, and age as dominant features, confirming their predictive value.

3. **Recursion Limit:** The agent uses LangGraph's default recursion limit. For complex multi-step reasoning, consider setting `max_iterations` explicitly.

4. **Activation Function Experiments:** The neural network uses ReLU with dropout. Future work should systematically compare Leaky ReLU, ELU, and Tanh variants, documenting effects on convergence and validation performance.

5. **Hyperparameter Tuning:** Random Forest uses fixed hyperparameters (`n_estimators=100`, `max_depth=10`). RandomizedSearchCV could explore optimal combinations and potentially improve ROC-AUC beyond 0.815.

---

## Summary

**Stage 1** builds a complete ML pipeline with proper preprocessing, feature engineering, and rigorous model comparison. Random Forest emerges as the best model for this income prediction task.

**Stage 2** leverages this model within a conversational AI agent, combining it with a knowledge base (RAG) to create an interactive system that answers questions, makes predictions, and explores the dataset.

Together, these stages demonstrate how classical ML and modern LLM-driven agents can be integrated to build intelligent, deployable systems.
