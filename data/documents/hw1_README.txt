**1. Problem Description**

The dataset used in this project is called _UCI Adult Census Data_. It falls within the domain of Social-Demographic data and it mainly concerns the average income of a group of adult individuals. The problem dealt with here is _binary classification_, as we have to determine whether someone earns more than 50,000$ / year (and so is financially "prosperous") or not - and thus our target variable is the _income_. Income prediction using personal data helps understand the factors behind the size of income accumulation, which should be particularly useful for governmental decisions and policy making.

**2. Dataset Description**

The dataset was acquired from the "UCI Machine Learning Repository", through Kaggle: https://www.kaggle.com/datasets/sagnikpatra/uci-adult-census-data-dataset . The total number of rows is 32,561 with 15 original columns (14 + 1 target). The **Table (2.1)** of features that describe our data is this:

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `age` | Continuous | Age in years |
| 2 | `workclass` | Categorical | Employment type (8 categories) |
| 3 | `fnlwgt` | Continuous | Census sampling weight (statistical factor for population representativeness) |
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
| 15 | `income` | **Target variable** | ≤50k$ or >50k$ |


The value distribution of our target variable is described in the following **Table (2.2)**:

| Class | Label | Count | Percentage |
|-------|-------|-------|------------|
| ≤50k$ | 0 | 24,720 | 75.9% |
| >50k$ | 1 | 7,841 | 24.1% |
| Total | - | 32,561 | 100% |


**3. Preprocessing Approach**

The following preprocessing steps were applied to prepare the dataset for training:

**STEP 1: Data splitting and stratification**

The dataset was split into training (80%), validation (10%), and test (10%) sets. A stratified split was used to keep the same proportion of income classes across all sets. This ensures the model does not become biased due to uneven class distribution.

**STEP 2: Standardized cleaning and Median Imputation**

- The raw data contained different representations of missing values: "?", "n/a", "N/A", "", " ", "NA", "null" and "NULL". These were all converted into a standard "NaN" format.  

- All features were then converted to numeric values using `(pd.to_numeric, errors="coerce")`. This means that any non-numeric or invalid values were automatically converted into `NaN`, allowing them to be handled consistently during the imputation step instead of causing errors or remaining as noisy data.

- Missing values were filled using the **median of each column**, calculated only from the training set. If a column contained only missing values, it was filled with 0. The median was chosen because it is less affected by extreme values compared to the mean.

**STEP 3: Outlier handling using the IQR clipping-method**

Outliers were handled using the Interquartile Range (IQR) method. For each feature, lower and upper bounds were calculated using:

- Lower bound = Q1 − 1.5 × IQR  
- Upper bound = Q3 + 1.5 × IQR  

Values outside this range were clipped to the nearest boundary instead of being removed. This keeps all data points while reducing the influence of extreme values.

**STEP 4: Feature scaling**

Feature scaling was applied using `StandardScaler`, which transforms all features to have a mean of 0 and a standard deviation of 1.
This ensures that all features contribute equally and prevents features with larger values from dominating the model.

_Final Note:_ All preprocessing steps were based only on the training set. These values were then applied to the validation and test sets.

**4. Feature engineering**

We generated two extra features to help in our task, shown in **Table (2.2)**:

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 16 | `capital_diff` | Continuous | Net capital income |
| 17 | `hours_log` | Continuous | Log-transformed work hours |

- **`capital_diff`** = `capital_gain` - `capital_loss`

Instead of giving the model two separate columns of opposite meaning, this combines them into a single number. What matters for predicting income isn't gains or losses alone. For example, a person with a x gain and a y_1 loss is not the same as someone with a x gain and a different, y_2 loss. Using one combined number makes this clearer for the model.

- **`hours_log`** = `log(1 + hours_per_week)`

Most people work around 40 hours per week, but a smaller group works many more. This means that a few large values might dominate the scale, so taking the log reduces their impact and brings the whole range closer together, making the data easier to handle. The *addition of 1* is used to avoid taking the log of zero.

**5. PCA Insights**

The scree plot shows that most of the dataset’s information is concentrated in just a few components. PC1 and PC2 together explain a large share of the variance and by the time we reach four components, we’ve already captured essentially all of it. This means the original features contain a lot of redundancy, and the data can be reduced to about four dimensions without losing meaningful information.

**PC1:** Dominated by `hours_per_week`, `hours_log`, and `education_num` — work hours and education are the strongest signals in the data.

**PC2:** Dominated by `fnlwgt`, `age`, and `education_num` — demographic and sampling factors account for the second most variation.

The vertical banding visible in the 2D projection is a telltale sign of categorical variables (like education level or marital status) strongly influencing the components.

The 2D scatter plot reveals two important insights:

1) **Class Overlap:** The blue (≤50k$) and red (>50k$) points are not cleanly separated. This confirms that the classes cannot be divided by a simple line.

2) **Data Structure:** The distinct stripes in the plot reflect the underlying categorical features (education, occupation, etc). 

This overlap explains why classification is challenging and justifies using more flexible models like Random Forest and Neural Networks to find non-linear decision boundaries. Even then, perfect separation is impossible with the given features, which is why our best AUC scores maximize around 0.81.

**6. Model Comparison**

Firstly, we created _two machine learning models (#1: Logistic Regression, #2: Random Forest)_, to find out which scores better, before making the key comparison with a neural network model. Here is the **Table (6.1)** with the results for the two classical models, presented side by side:

| Model | Training Accuracy | Validation Accuracy | Validation AUC | Test Accuracy |
|-------| ----------------- | ------------------- | -------------- | ------------- |
| Logistic Regression | 0.726 | 0.726 | 0.799 | 0.730 |
| Random Forest | 0.745 | 0.720 | 0.807 | 0.722 |

As we see, the results are comparable, despite Logistic Regression being a simpler model. While Logistic Regression outmatches Random Forest in Validation and Test Accuracies by less than 1% in each case (**0.726** vs 0.720 || **0.730** vs 0.722), and also showing perfect generalization with identical Training and Validation Accuracies (0.726), Random Forest achieved a higher Training Accuracy (**0.745** vs 0.726) and demonstrated superior class separation as measured by Validation AUC (**0.807** vs 0.799).

Therefore, Random Forest is selected as the best classical model and saved to `models/classical_model.pkl`.

**Does the (prevailing) classical model's feature importance align with PCA loadings?**  
Yes, both PCA and Random Forest identified `age`, `education_num`, `hours_per_week`, and `hours_log` as the dominant features, confirming these factors are the most vital for income prediction. Notably, `hours_log` is a log transformation of `hours_per_week`, which explains why they consistently appear together during both analyses.

We now compare _the winning classical model against the neural network_. The following **Table (6.2)** presents the test set metrics of the two models (this behavior proved to be consistent among several runs):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 0.722 | 0.453 | 0.753 | 0.565 | 0.815 |
| Neural Network | 0.794 | 0.645 | 0.324 | 0.431 | 0.812 |

_Note #1:_ **Why did we only use test set metrics in the final comparison?** The reason is that the classical model comparison included a test set evaluation, giving Random Forest a slight advantage in this final comparison.

_Note #2:_ **Does the neural network’s loss curve suggest it had enough data to learn well?** The training history demonstrates a stable learning process. Both training and validation loss show a steady decline, with the most rapid learning occurring before Epoch 40. By Epoch 100 the curves have flattened, indicating that the model has reached convergence and extracted the available predictive information from the dataset - therefore, the cutoff point is appropriate. Crucially, the validation loss closely tracks the training loss without diverging, proving that the model generalizes well and has not proceeded to overfitting. 

As we see, the Neural Network actually manages to edge out Random Forest in terms of sheer Accuracy and, mainly, Precision, but that happens mainly because its Recall rate is low (32.4%). This combination of scores means that the Neural Network "plays it safe" - it is right more often, but finds far fewer high-income individuals. Random Forest on the other hand catches ~75% of the target class, and ultimately does the job better.

This result is not surprising. As noted already, on tabular datasets of such modest size, classical algorithms like Random Forest can outperform neural networks with few hidden layers. Our comparison confirms this empirical finding: the added complexity of the neural network is simply not justified by any noteworthy performance gain.

**7. Best Model Designation**

The best model selected is Random Forest, saved in `models/best_model.pkl`.

**_Justification summary_**:

- Highest ROC-AUC (0.815) indicating more precise class separation.
- Highest Recall (0.753) meaning it correctly identifies ~75% of high-income individuals.
- Best F1-Score (0.565) showing the best balance of precision and recall.

**8. Installation & Execution**

The requirements are:

- Python 3.8 or higher
- Access to git

The steps of the execution are:

**STEP 1: Clone the Repository**

git clone https://github.com/davidtel999/hw1.git
cd hw1

**STEP 2: Create/activate a virtual environment**

python -m venv myvenv

**Windows:**

myvenv\Scripts\activate

**Mac/Linux:**

source myvenv/bin/activate

**STEP 3: Install the listed dependencies:**

pip install -r requirements.txt

**STEP 4: Run the pipeline:**

python main.py