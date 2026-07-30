import streamlit as st
import pandas as pd
import joblib

# -----------------------------

# Load Models

# -----------------------------

logistic_model = joblib.load('logistic_model.pkl')
knn_model = joblib.load('knn_model.pkl')
naive_model = joblib.load('naive_bayes_model.pkl')
scaler = joblib.load('scaler.pkl')

# -----------------------------

# Page

# -----------------------------

st.set_page_config(page_title='Algorithm Comparison', page_icon='🤖', layout='centered')

st.title('Algorithm Comparison App')
st.subheader('Bank Customer Churn Prediction')
st.write('Choose an algorithm and predict whether the customer will **Stay** or **Exit**.')

st.divider()

# -----------------------------

# Algorithm Selection

# -----------------------------

algorithm = st.selectbox(
'Select Algorithm',
['Logistic Regression', 'KNN', 'Naive Bayes']
)

st.divider()

# -----------------------------

# Input Fields

# -----------------------------

col1, col2 = st.columns(2)

with col1:
 credit_score = st.number_input('Credit Score', 300, 900, 650)
age = st.number_input('Age', 18, 100, 30)

with col2:
 balance = st.number_input('Balance', 0.0, 200000.0, 50000.0)
salary = st.number_input('Estimated Salary', 0.0, 200000.0, 50000.0)

st.divider()

# -----------------------------

# Prediction

# -----------------------------

if st.button('Predict'):


 input_data = pd.DataFrame([[credit_score, 0, 1, age, 5, balance, 1, 1, 1, salary]],
                          columns=['CreditScore', 'Geography', 'Gender', 'Age',
                                   'Tenure', 'Balance', 'NumOfProducts',
                                   'HasCrCard', 'IsActiveMember',
                                   'EstimatedSalary'])

input_scaled = scaler.transform(input_data)

if algorithm == 'Logistic Regression':
    prediction = logistic_model.predict(input_scaled)
elif algorithm == 'KNN':
    prediction = knn_model.predict(input_scaled)
else:
    prediction = naive_model.predict(input_scaled)

st.subheader('Prediction Result')

if prediction[0] == 1:
    st.error('Customer is likely to **Exit**')
else:
    st.success('Customer is likely to **Stay**')


st.divider()

# -----------------------------

# Information

# -----------------------------

with st.expander('Sample Test Values'):
 st.write('**Stay:** Credit Score 750, Age 30, Balance 15000, Salary 70000')
 st.write('**Exit:** Credit Score 420, Age 58, Balance 120000, Salary 30000')

st.caption('Developed by Achal Chaudhari')
