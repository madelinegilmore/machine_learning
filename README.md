# machine_learning

This repository contains a comprehensive machine learning project focused on loan approval prediction, developed through an iterative three-stage workflow with peer review feedback.

## Loan Approval Prediction Project

A complete machine learning pipeline that progresses through data cleaning, feature engineering, model development, and comparative analysis.

### Project Workflow: Three Stages

The project is organized into three progressive notebooks that build upon each other:

#### **Notebook 1: Data Cleaning & Preparation**
- **Focus:** Initial data exploration and cleaning based on peer feedback
- **Key Activities:**
  - Loaded raw loan applications dataset (58,645 records, 18 features)
  - Identified and removed unnecessary variables: id, applicant_account_No, bank_sort_code, and sex (98.8% missing)
  - Handled missing values using median and mode imputation
  - Identified and removed outliers from age, loan_interest_rate, and max_loan_amount using domain logic and IQR methods
  - Standardized target variable into consistent categories (approved/denied)
  - Encoded categorical features (education, home_ownership, loan_intent, payment_default_on_file)
  - Exported cleaned dataset for downstream modeling

#### **Notebook 2: Feature Engineering & Model Selection** *(in progress)*
- **Focus:** Preparing data for modeling and initial algorithm exploration
- **Key Activities:**
  - Separated features from target variable
  - Applied StandardScaler to prevent data leakage (fit on training, transform on test)
  - Train-test split: 70% training, 30% testing (stratified)
  - Prepared 11 final features for modeling

#### **Notebook 3: Model Training & Evaluation**
- **Focus:** Training, optimizing, and comparing multiple machine learning algorithms for both classification and regression tasks

- **Classification Models:**

  **1. Logistic Regression (Baseline)**
  - Parameters: C=1, class_weight=None
  - Accuracy: 88.91%
  - Precision (Class 1): 0.73 | Recall: 0.39
  - Assessment: Strong true negative detection but struggles with positive class recall

  **2. K-Nearest Neighbors (KNN) - Optimized** ⭐ *Best Classification Model*
  - Parameters: n_neighbors=10, weights='distance'
  - Accuracy: 91.18%
  - Precision (Class 1): 0.82 | Recall: 0.52
  - Assessment: Superior balance of precision and recall; best overall performance

  **3. Voting Ensemble Classifier**
  - Combines Logistic Regression + KNN with soft voting
  - Accuracy: 90.0%
  - Precision (Class 1): 0.78 | Recall: 0.43
  - Assessment: Moderate performance; ensemble did not outperform optimized KNN

- **Regression Task: Maximum Loan Amount Prediction**
  - **Decision Tree Regressors** trained with varying complexity (unpruned and depth-limited models)
  - **Evaluation Metrics:** Mean Absolute Error (MAE), Mean Squared Error (MSE), R-squared scores
  - **Real-world Application:** Models demonstrated ability to predict loan amounts for new applicants

- **Evaluation Metrics Used (Classification):**
  - Confusion matrices for classification visualization
  - Classification reports (precision, recall, F1-score)
  - ROC curves for AUC assessment
  - Stratified train-test split to maintain class distribution

### Data Processing Summary

- **Initial Dataset:** 58,645 records × 18 features
- **After Cleaning:** 56,405 records × 11 features
- **Removed Features:** sex, id, applicant_account_No, bank_sort_code (irrelevant or excessive missing values)
- **Target Variables:** 
  - Classification: credit_application_acceptance (binary: 0/1)
  - Regression: max_loan_amount(allowed)
- **Key Features Retained:** 
  - age, education_qualifications, income, home_ownership, employment_length
  - loan_intent, loan_amount, loan_interest_rate, loan_income_ratio
  - payment_default_on_file, credit_history_length

### Classification Model Performance Comparison

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|-------|----------|---------------------|------------------|--------------------|
| Logistic Regression | 88.91% | 0.73 | 0.39 | 0.51 |
| **KNN (Optimized)** | **91.18%** | **0.82** | **0.52** | **0.64** |
| Voting Ensemble | 90.00% | 0.78 | 0.43 | 0.55 |

### Key Learnings

1. **Data Cleaning Impact:** Removal of data leakage sources (sex, outliers in max_loan_amount) significantly improved model reliability
2. **Hyperparameter Tuning:** KNN with distance-weighted voting and n_neighbors=10 outperformed default settings
3. **Ensemble Effectiveness:** While soft voting ensemble showed promise, the optimized single KNN model achieved better overall classification performance
4. **Class Imbalance:** Stratified train-test split was essential to maintain minority class representation in both sets
5. **Scaling Necessity:** StandardScaler applied post-split prevents data leakage and improves model convergence
6. **Multi-task Learning:** Successfully implemented both classification (approval prediction) and regression (loan amount prediction) on the same dataset

### Project Outputs

This project demonstrates a complete machine learning workflow from:
- Data exploration and preprocessing through outlier removal and imputation
- Feature selection and encoding
- Model training, optimization, and evaluation for both classification and regression
- Real-world prediction applications across multiple prediction tasks
