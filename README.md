# Credit Risk Prediction Web App

A minimal **Streamlit** app that predicts **credit default probability** based on user inputs such as loan amount, income, interest rate, and more. The underlying model is an **XGBoost** (Gradient Boosting) classifier trained on a public credit risk dataset.

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)     
3. [Model Details](#model-details)  
4. [Live Demo](#live-demo)  
5. [Conclusion](#conclusion)
---

## **Project Overview**

Credit risk modeling is crucial for financial institutions to assess the likelihood that a borrower will default on a loan. This project demonstrates how to:

- Load and preprocess a **credit risk dataset** from Kaggle.   
- Train a classification model (Logistic Regression, RandomForest, and **XGBoost**)  
- Deploy the final chosen model (XGBoost) into a **Streamlit** web application  
- Allow users to enter loan and income information for an **instant** default probability estimate  

---

## **Key Features**

1. **Interactive Input**: Users can input raw loan amount and annual income, which the app internally log-transforms (to match training data).  
2. **Immediate Prediction**: The app instantly calculates the likelihood of default based on the trained XGBoost model.  
3. **Clean UI**: Minimalist sliders, select boxes, and number inputs—easy for anyone to use.  
4. **Extendable**: The code can easily be extended to use other models or more features.  

---

## **Model Details**

1. Dataset

We used a dataset named credit_risk_dataset.csv, containing features like loan_amount, person_income, loan_int_rate, and more. Rows with missing data were dropped or imputed (depending on your EDA/cleaning choice).

2. Preprocessing

Clipped extreme ages (capped at 80).
Applied log transforms to income and loan amounts for better model performance.
One-hot encoded categorical variables (e.g., home ownership, loan intent).
Addressed class imbalance via SMOTE (oversampling the minority class).


3. Model

Final chosen model: **XGBoost** (with scale_pos_weight to further address imbalance).
Trained model is saved as credit_risk_model.pkl.

4. Evaluation

Metrics include Precision, Recall, F1-score, and ROC-AUC.
Typical ROC-AUC for this project is around 0.80–0.90 (depending on hyperparameter tuning and splits).

---

## **Live Demo** 

To see the live demo, press the link down below to see the model in action! 

https://creditriskprediction.streamlit.app/

---

## **Conclusion**
In this project, I tackled the full pipeline for credit risk modeling, from cleaning and engineering the data to handling class imbalance with SMOTE and trying out multiple classifiers. I ended up selecting XGBoost, which consistently delivered solid results (with an ROC-AUC generally in the 0.80–0.90 range and the accuracy being around 90%). I also built a Streamlit web app so anyone can enter basic loan and income details to get an instant default probability prediction. Although there’s always room for improvement (like deeper hyperparameter tuning or additional features), I hope this shows a practical approach to merging machine learning with an accessible interface, making data-driven insights more approachable!











