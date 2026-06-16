# Banking Fraud Detection System

## Project Overview
This project develops a robust fraud detection system using a credit card transaction dataset. The primary goal is to identify fraudulent transactions with high accuracy, addressing the significant class imbalance inherent in such datasets. The solution employs various machine learning models and interpretability techniques to provide actionable insights.

## Dataset
The dataset used is the Kaggle Credit Card Fraud Detection dataset, which contains anonymized transaction features (V1-V28), `Time`, `Amount`, and a `Class` variable indicating fraudulent (1) or legitimate (0) transactions.

## Features
- Data Loading and Initial Inspection
- Comprehensive Exploratory Data Analysis (EDA)
- Feature Engineering to enhance model performance
- Data Preprocessing including scaling and handling class imbalance
- Training and Evaluation of multiple machine learning models
- Model Interpretability using Feature Importance and SHAP values
- Actionable Recommendations for deployment and continuous improvement

## Installation
To run this notebook, you will need the following Python libraries. You can install them using pip:

```bash
pip install pandas matplotlib seaborn numpy scikit-learn imbalanced-learn xgboost shap
```

## Usage
1.  **Download the dataset**: Obtain the `creditcard.csv` file from the [Kaggle Credit Card Fraud Detection]((https://www.kaggle.com/datasets/arockiaselciaa/creditcardcsv)) dataset and place it in the `/content/` directory of your Colab environment.
2.  **Run the notebook**: Execute all cells in the Jupyter notebook sequentially.

## Exploratory Data Analysis (EDA) Highlights
-   **Extreme Class Imbalance**: Only about 0.38% of transactions were fraudulent, necessitating techniques like SMOTE for balancing the training data.
-   **Anonymized Features (V1-V28)**: These principal components were critical, with features like V14, V17, V10, V12, V3, V16, V11, and V4 showing strong correlations with the `Class` variable.
-   **Time and Amount Distributions**: Fraudulent transactions generally involved smaller amounts. The 'Time' feature did not show strong distinguishing diurnal patterns for fraud compared to legitimate transactions. Log-transformation of 'Amount' was applied to normalize its skewed distribution.

## Feature Engineering
-   **Missing Values**: A single row with missing data was dropped.
-   **Log-Transformation of 'Amount'**: Applied `np.log1p` to the `Amount` feature (`Amount_log`) to normalize its distribution.
-   **'Hour' Feature**: Extracted the `Hour` of the day from the `Time` column to capture potential cyclical patterns.

## Data Preprocessing and Splitting
-   **Features and Target**: `Time` and original `Amount` columns were dropped, using `Amount_log` and `Hour` instead.
-   **Stratified Split**: Data was split into 80% training and 20% testing sets, maintaining the original class distribution.
-   **Feature Scaling**: All numerical features were scaled using `StandardScaler`.
-   **Class Imbalance Handling**: SMOTE (Synthetic Minority Oversampling Technique) was applied to the training data to balance the class distribution.

## Model Training and Evaluation
Three classification models were trained and evaluated on the resampled training data and the original-distribution test data. Key metrics included Precision, Recall, F1-score, and ROC-AUC, with a focus on the minority (fraudulent) class.

| Model             | ROC-AUC | Precision | Recall | F1-Score |
|:------------------|:--------|:----------|:-------|:---------|
| Logistic Regression | 0.9957  | 0.6667    | 0.75   | 0.7059   |
| Random Forest     | **0.9999**| **1.0000**| 0.75   | **0.8571** |
| XGBoost           | 0.9984  | 0.8571    | 0.75   | 0.8000   |

**Best Performing Model**: The **Random Forest Classifier** achieved superior performance, with a near-perfect ROC-AUC of 0.9999 and perfect Precision (1.00) for the fraud class, making it the chosen model for interpretability.

## Model Interpretability
Both built-in feature importances from Random Forest and SHAP (SHapley Additive exPlanations) values were used to understand feature contributions:
-   **Key Features**: V14, V17, V10, V12, V3, V11, and V4 consistently showed the highest importance in distinguishing fraudulent transactions.
-   **SHAP Plots**: Provided detailed insights into how each feature influences individual predictions and overall model behavior.

## Actionable Recommendations
1.  **Deploy Random Forest Model**: Integrate the highly accurate Random Forest model into a production fraud detection system.
2.  **Investigate Key Features**: Conduct further domain-specific research into the highly influential anonymized features.
3.  **Real-time Monitoring**: Implement the model within a real-time system to flag suspicious transactions for immediate review.
4.  **Continuous Improvement**: Regularly retrain the model with new data to adapt to evolving fraud patterns.
5.  **Threshold Adjustment**: Optimize the prediction probability threshold based on business requirements to balance false positives and false negatives.

## Potential Enhancements / Future Work
-   **Advanced Anomaly Detection**: Explore unsupervised learning methods for detecting novel fraud patterns.
-   **Deep Learning Models**: Investigate neural networks for potentially better feature extraction from raw data.
-   **Time-Series Analysis**: If actual timestamps are available, incorporate advanced time-series features.
-   **Power BI Dashboard Integration**: Develop a Power BI dashboard for interactive visualization of model performance, fraud trends, and individual transaction explanations.
