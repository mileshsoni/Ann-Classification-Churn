import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import numpy as np
import sklearn

model = tf.keras.models.load_model('model.h5')

with open('onehot_encoder_geo.pkl', 'rb') as file :
    label_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file :
    scaler = pickle.load(file)
    
with open('Label_encoder_gender.pkl', 'rb') as file :
    label_encoder_gender = pickle.load(file)


#streamlit app
st.title("Customer Churn Prediction")
st.write(f"scikit-learn version: {sklearn.__version__}")

#user input
geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 3)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns = label_encoder_geo.get_feature_names(geography))
input_data_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis = 1)
input_scaled = scaler.transform(input_data_df)
pred = model.predict(input_scaled)[0][0]
st.write(f'Churn Probability : {pred:.2f}')
if pred > 0.5 :
    st.write("Customer will leave the Bank")
else :
    st.write("Customer will not leave the Bank")
