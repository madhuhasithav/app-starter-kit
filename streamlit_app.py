import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
st.title("Email Marketing Campaign Prediction")
st.write("Upload your dataset:")
uploaded_file = st.file_uploader("Choose a file", type=["xlsx"])
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    def remove_outliers(data):
        df_clean = data.copy()
        for col in df_clean.select_dtypes(include=['float64', 'int64']).columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        return df_clean
    df_cleaned = remove_outliers(data)
    X = df_cleaned.drop('Opened_Previous_Emails', axis=1)
    y = df_cleaned['Opened_Previous_Emails']  # Target variable
    numerical_features = ['Customer_Age', 'Emails_Opened', 'Emails_Clicked', 'Purchase_History',
                          'Time_Spent_On_Website', 'Days_Since_Last_Open', 'Customer_Engagement_Score']
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)
    xgb_model = XGBClassifier(random_state=42, max_depth=30)
    xgb_model.fit(X_train, y_train)
    mlp_model = MLPClassifier(
        learning_rate_init=0.01,
        hidden_layer_sizes=(32, 64),
        alpha=0.1,
        activation='tanh',
        max_iter=500,
        random_state=42
    )
    mlp_model.fit(X_train, y_train)
    st.write("Enter the following details:")
    customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
    emails_opened = st.number_input("Emails Opened", min_value=0, max_value=100, value=5)
    emails_clicked = st.number_input("Emails Clicked", min_value=0, max_value=100, value=2)
    purchase_history = st.number_input("Purchase History", min_value=0, max_value=10, value=0)
    time_spent_on_website = st.number_input("Time Spent on Website (in minutes)", min_value=0, max_value=500, value=10)
    days_since_last_open = st.number_input("Days Since Last Open", min_value=0, max_value=365, value=5)
    customer_engagement_score = st.number_input("Customer Engagement Score", min_value=0, max_value=100, value=50)
    opened_previous_emails = st.selectbox("Opened Previous Emails", options=[0, 1], index=0)
    clicked_previous_emails = st.selectbox("Clicked Previous Emails", options=[0, 1], index=0)
    device_type = st.selectbox("Device Type", options=[0, 1, 2], index=0)  # Use integers 0, 1, 2 for Device_Type
    input_data = {
        'Customer_Age': customer_age,
        'Emails_Opened': emails_opened,
        'Emails_Clicked': emails_clicked,
        'Purchase_History': purchase_history,
        'Time_Spent_On_Website': time_spent_on_website,
        'Days_Since_Last_Open': days_since_last_open,
        'Customer_Engagement_Score': customer_engagement_score,
        'Opened_Previous_Emails': opened_previous_emails,
        'Clicked_Previous_Emails': clicked_previous_emails,
        'Device_Type': device_type
    }
    input_df = pd.DataFrame([input_data])
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])
    input_df = input_df.drop('Opened_Previous_Emails', axis=1)
    if st.button('Predict'):
        rf_pred = rf_model.predict(input_df)
        rf_pred_prob = rf_model.predict_proba(input_df)[:, 1]
        xgb_pred = xgb_model.predict(input_df)
        xgb_pred_prob = xgb_model.predict_proba(input_df)[:, 1]
        mlp_pred = mlp_model.predict(input_df)
        mlp_pred_prob = mlp_model.predict_proba(input_df)[:, 1]
        st.subheader("Prediction Results")
        st.write(f"Random Forest Prediction: {'Opened' if rf_pred[0] == 1 else 'Not Opened'} (Probability: {rf_pred_prob[0]:.2f})")
        st.write(f"XGBoost Prediction: {'Opened' if xgb_pred[0] == 1 else 'Not Opened'} (Probability: {xgb_pred_prob[0]:.2f})")
        st.write(f"MLP Prediction: {'Opened' if mlp_pred[0] == 1 else 'Not Opened'} (Probability: {mlp_pred_prob[0]:.2f})")
else:
    st.warning("Please upload a dataset to proceed.")
st.markdown(
    """
    <style>
    footer {
        visibility: hidden;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: #000;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        <p>Web app created by <b>Madhu Hasitha</b></p>
    </div>
    """,
    unsafe_allow_html=True
)
