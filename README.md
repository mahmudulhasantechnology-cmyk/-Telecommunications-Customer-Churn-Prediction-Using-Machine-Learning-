# Telecommunications Customer Churn Prediction System

A machine learning-based classification system designed to predict customer churn, mitigate revenue loss, and identify at-risk subscriber segments using Python and Google Colab.

---

## Project Workflow & Tasks

### 1. Data Preprocessing
* **Missing & Duplicate Values:** Identified missing entries (e.g., in `Region`) and handled them via median/mode imputation; dropped all duplicate rows.
* **Categorical Encoding:** Formatted mixed text data types and converted categorical columns (`Gender`, `Region`, `PlanType`) into numerical formats using `OneHotEncoder`.
* **Outlier Capping:** Applied the Interquartile Range (IQR) method to cap extreme values and protect model stability.
* **Feature Scaling:** Applied `StandardScaler` to normalize numeric variations and prevent data leakage.
* **Data Splitting:** Partitioned the database into an 80% training set and a 20% validation test set.

### 2. Exploratory Data Analysis (EDA)
* **Demographic Profiling:** Summarized subscriber age brackets, billing averages, and subscription distributions.
* **Feature Correlation:** Generated behavioral heatmaps detailing the relationship between higher monthly charges and churn density patterns.

### 3. Class Imbalance Resolution
* **Distribution Audit:** Diagnosed significant class imbalances between customer retention states.
* **Random Oversampling:** Applied `RandomOverSampler` strictly to the training split to establish a clean 50/50 balance without causing evaluation bias.

### 4. Model Building
* **Logistic Regression:** Trained a probabilistic baseline model combined with a 3-Fold Stratified Cross-Validation framework.
* **K-Nearest Neighbors (KNN):** Implemented an alternative spatial classification model optimized at `n_neighbors=3`.

### 5. Model Evaluation & Comparison
* **Comprehensive Metrics:** Benchmarked both algorithms across Accuracy, Precision, Recall, and F1-score.
* **Confusion Matrices:** Plotted side-by-side heatmaps tracking true retention versus missed churn instances (False Negatives).
* **Core Focus:** Evaluated performance with a heavy business emphasis on **Recall** and **F1-score** over raw accuracy.

---

## Key Findings & Business Impact

* **Research Question Answer:** Both models effectively map churn risk. However, **Logistic Regression** yields higher stability and validation performance following appropriate preprocessing and class-balancing.
* **Preventive Strategy:** The company should use this system's risk scores to proactively migrate high-billing and vulnerable age brackets onto competitive, value-added bundles before they switch to competitors.
