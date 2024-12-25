# Credit Risk Prediction Web App

A minimal **Streamlit** app that predicts **credit default probability** based on user inputs such as loan amount, income, interest rate, and more. The underlying model is an **XGBoost** (Gradient Boosting) classifier trained on a public credit risk dataset.

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Setup & Installation](#setup--installation)  
4. [Usage](#usage)  
5. [Model Details](#model-details)  
6. [Live Demo (Optional)](#live-demo-optional)  
7. [Project Structure](#project-structure)  
8. [License](#license)  
9. [Contact](#contact)

---

## **Project Overview**

Credit risk modeling is crucial for financial institutions to assess the likelihood that a borrower will default on a loan. This project demonstrates how to:

- Load and preprocess a **credit risk dataset**  
- Train a classification model (Logistic Regression, RandomForest, and **XGBoost**)  
- Deploy the final chosen model (XGBoost) into a **Streamlit** web application  
- Allow users to enter loan and income information for an **instant** default probability estimate  

## **Key Features**

1. **Interactive Input**: Users can input raw loan amount and annual income, which the app internally log-transforms (to match training data).  
2. **Immediate Prediction**: The app instantly calculates the likelihood of default based on the trained XGBoost model.  
3. **Clean UI**: Minimalist sliders, select boxes, and number inputs—easy for anyone to use.  
4. **Extendable**: The code can easily be extended to use other models or more features.  

---

## Setup & Installation

### Prerequisites

- [Python 3.7+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads) (optional, if you want to clone directly)
- A command line or terminal

### Steps

1. **Clone or Download** this repository:
   ```bash
   git clone https://github.com/your-username/credit-risk-ml-models.git

https://creditriskprediction.streamlit.app/
