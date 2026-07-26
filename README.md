## 📊 Customer Churn Analysis & Prediction

An interactive Streamlit web app for analyzing customer churn patterns and predicting whether a customer is likely to churn, built on a telecom customer dataset.

## 🔗 Live Demo
](https://customer-churn-prediction-akhil.streamlit.app/)

## 📌 Overview

This project analyzes customer churn behavior using Exploratory Data Analysis (EDA) and Machine Learning. It helps identify **why customers leave** and predicts **which customers are at risk of churning**, so businesses can take proactive retention actions.

## ✨ Features

- **📈 EDA Dashboard** — Visual breakdown of churn by gender, contract type, tenure, payment method, and services used
- **🤖 Model Training** — Compares Logistic Regression and Random Forest models with accuracy, classification reports, and feature importance
- **🎯 Live Prediction** — Enter a new customer's details and instantly get a churn prediction with probability score

## 🗂️ Dataset

The dataset (`Customer Churn.csv`) contains telecom customer records with features such as:

- Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- Account info: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`
- Services: `PhoneService`, `InternetService`, `OnlineSecurity`, `TechSupport`, `StreamingTV`, etc.
- Charges: `MonthlyCharges`, `TotalCharges`
- Target: `Churn` (Yes/No)

## 🔍 Key Insights

- **~26.5%** of customers in the dataset have churned
- Customers on **month-to-month contracts** churn significantly more than those on 1-year/2-year contracts
- Customers paying via **electronic check** show higher churn rates
- Customers with **low tenure (1-2 months)** are most likely to churn
- Customers without add-on services like **OnlineSecurity** and **TechSupport** churn more

## 🛠️ Tech Stack

- **Python**
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — visualization
- **Scikit-learn** — machine learning (Logistic Regression, Random Forest)
- **Streamlit** — web app framework

## 📁 Project Structure

```
customer-churn-app/
├── app.py                 # Streamlit application
├── requirements.txt        # Python dependencies
├── Customer Churn.csv      # Dataset
└── README.md               # Project documentation
```


### 🧠 Model Performance

| Model | Accuracy |
|---|---|
| Logistic Regression | ~80% |
| Random Forest | ~79-80% |

*(Exact numbers may vary slightly on each run/train-test split)*

## 📊 App Preview

- **EDA Tab** — churn distribution, contract analysis, tenure histograms, payment method breakdown
- **Model Tab** — accuracy metrics, classification reports, top 10 important features
- **Prediction Tab** — interactive form to predict churn for a new customer

## 🔮 Future Improvements

- Add hyperparameter tuning (GridSearchCV) for better model performance
- Handle class imbalance using SMOTE
- Add SHAP explainability for individual predictions
- Save trained model as `.pkl` for faster loading instead of retraining on each run
- Add more model options (XGBoost, SVM)

## 👤 Author

Made by [AKHIL ANAND] — feel free to connect on [[LinkedIn](https://www.linkedin.com/in/akhilanand01/)]

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
