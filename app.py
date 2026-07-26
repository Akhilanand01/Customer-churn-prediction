import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Customer Churn.csv")
    df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")
    df["TotalCharges"] = df["TotalCharges"].astype("float")
    df["SeniorCitizen"] = df["SeniorCitizen"].apply(lambda x: "yes" if x == 1 else "no")
    return df

df = load_data()

st.title("📊 Customer Churn Analysis & Prediction")

tab1, tab2, tab3 = st.tabs(["🔍 EDA Dashboard", "🤖 Model Training", "🎯 Predict Churn"])

# ---------------------------------------------------------
# TAB 1: EDA
# ---------------------------------------------------------
with tab1:
    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Churn Count**")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.countplot(x="Churn", data=df, ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("**Churn %**")
        fig, ax = plt.subplots(figsize=(4, 3))
        gb = df.groupby("Churn").agg({"Churn": "count"})
        ax.pie(gb["Churn"], labels=gb.index, autopct="%1.2f%%")
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Churn by Contract Type**")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.countplot(x="Contract", data=df, hue="Churn", ax=ax)
        st.pyplot(fig)

    with col4:
        st.markdown("**Tenure Distribution**")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.histplot(x="tenure", data=df, bins=30, hue="Churn", ax=ax)
        st.pyplot(fig)

    st.markdown("**Churn by Payment Method**")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(x="PaymentMethod", data=df, hue="Churn", ax=ax)
    plt.xticks(rotation=30)
    st.pyplot(fig)

# ---------------------------------------------------------
# 2. PREPARE DATA FOR MODELING (shared by tab2 & tab3)
# ---------------------------------------------------------
@st.cache_resource
def train_models(df):
    df_model = df.drop("customerID", axis=1)
    df_model["Churn"] = df_model["Churn"].map({"Yes": 1, "No": 0})
    df_model = pd.get_dummies(df_model, drop_first=True)

    X = df_model.drop("Churn", axis=1)
    y = df_model["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_scaled, y_train)
    y_pred_log = log_model.predict(X_test_scaled)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    results = {
        "log_model": log_model,
        "rf_model": rf_model,
        "scaler": scaler,
        "X_columns": X.columns,
        "log_acc": accuracy_score(y_test, y_pred_log),
        "rf_acc": accuracy_score(y_test, y_pred_rf),
        "log_report": classification_report(y_test, y_pred_log),
        "rf_report": classification_report(y_test, y_pred_rf),
        "log_cm": confusion_matrix(y_test, y_pred_log),
        "rf_cm": confusion_matrix(y_test, y_pred_rf),
    }
    return results

results = train_models(df)

# ---------------------------------------------------------
# TAB 2: MODEL TRAINING RESULTS
# ---------------------------------------------------------
with tab2:
    st.subheader("Model Performance Comparison")

    col1, col2 = st.columns(2)
    col1.metric("Logistic Regression Accuracy", f"{results['log_acc']*100:.2f}%")
    col2.metric("Random Forest Accuracy", f"{results['rf_acc']*100:.2f}%")

    st.markdown("**Logistic Regression Report**")
    st.text(results["log_report"])

    st.markdown("**Random Forest Report**")
    st.text(results["rf_report"])

    st.markdown("**Feature Importance (Random Forest)**")
    importances = pd.Series(
        results["rf_model"].feature_importances_, index=results["X_columns"]
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    importances.sort_values(ascending=False).head(10).plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    st.pyplot(fig)

# ---------------------------------------------------------
# TAB 3: LIVE PREDICTION
# ---------------------------------------------------------
with tab3:
    st.subheader("Predict Churn for a New Customer")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", df["gender"].unique())
            senior = st.selectbox("Senior Citizen", ["no", "yes"])
            partner = st.selectbox("Partner", df["Partner"].unique())
            dependents = st.selectbox("Dependents", df["Dependents"].unique())
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with col2:
            phone = st.selectbox("Phone Service", df["PhoneService"].unique())
            multiline = st.selectbox("Multiple Lines", df["MultipleLines"].unique())
            internet = st.selectbox("Internet Service", df["InternetService"].unique())
            contract = st.selectbox("Contract", df["Contract"].unique())
            paperless = st.selectbox("Paperless Billing", df["PaperlessBilling"].unique())

        with col3:
            payment = st.selectbox("Payment Method", df["PaymentMethod"].unique())
            monthly = st.number_input("Monthly Charges", value=70.0)
            total = st.number_input("Total Charges", value=800.0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Build a single-row dataframe matching training format
        new_data = df.drop("customerID", axis=1).iloc[0:0].copy()
        row = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multiline, "InternetService": internet,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
        }
        # Fill remaining columns with the mode/default from training data
        for col in new_data.columns:
            if col not in row and col != "Churn":
                row[col] = df[col].mode()[0] if col in df.columns else 0

        new_df = pd.DataFrame([row])
        full = pd.concat([df.drop("customerID", axis=1), new_df], ignore_index=True)
        full["Churn"] = full["Churn"].map({"Yes": 1, "No": 0})
        full_encoded = pd.get_dummies(full.drop("Churn", axis=1), drop_first=True)
        full_encoded = full_encoded.reindex(columns=results["X_columns"], fill_value=0)

        new_encoded = full_encoded.iloc[[-1]]
        pred = results["rf_model"].predict(new_encoded)[0]
        prob = results["rf_model"].predict_proba(new_encoded)[0][1]

        if pred == 1:
            st.error(f"⚠️ This customer is LIKELY TO CHURN (probability: {prob*100:.1f}%)")
        else:
            st.success(f"✅ This customer is likely to STAY (churn probability: {prob*100:.1f}%)")
