# machine_learning

This repository contains a collection of my machine learning coursework and projects, developed as part of my academic studies and independent exploration of machine learning concepts.

## Loan Approval Prediction Project

This was an ongoing semester project focused on building a machine learning model to predict loan approval decisions using financial and applicant data.

### Project Overview

The project involved a comprehensive machine learning pipeline for analyzing loan application data, tackling both classification and regression tasks:

**Classification Task:** Predicting loan approval status (approved/denied)
- Implemented multiple algorithms including Logistic Regression, K-Nearest Neighbors (KNN), and an ensemble VotingClassifier combining Logistic Regression and KNN
- Evaluated models using confusion matrices, classification reports, accuracy scores, and ROC curves
- Addressed data leakage issues by applying StandardScaler after train-test split

**Regression Task:** Predicting maximum loan amount (allowed)
- Trained Decision Tree Regressors with varying complexity (unpruned and depth-limited)
- Evaluated performance using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared scores
- Demonstrated real-world application by predicting loan amounts for new applicants

### Data Processing

The project included rigorous data cleaning and preprocessing:
- Removed irrelevant features (sex, id, bank_sort_code, applicant_account_No)
- Handled missing values across all features using median and mode imputation
- Standardized the target variable into consistent categories (approved/denied)
- Identified and removed outliers in age, loan_interest_rate, and maximum loan amount using domain logic and IQR methods
- Encoded categorical features and exported the final processed dataset

This project demonstrates a complete machine learning workflow from data exploration and preprocessing through model training, evaluation, and real-world prediction applications.
