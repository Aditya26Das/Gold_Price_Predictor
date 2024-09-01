import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

try:
    gold_data = pd.read_csv('./gld_price_data.csv')
    X = gold_data.drop(['Date','GLD'],axis=1)
    Y = gold_data['GLD']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)
    regressor = RandomForestRegressor(n_estimators=100)
    regressor.fit(X_train,Y_train)

    # Title of the app
    st.title("Gold Price Prediction")

    # Input fields for user input
    spx = st.number_input("Enter The Standard and Poor's 500 index value:", min_value=0.0)
    uso = st.number_input("Enter The United States Oil Fund price per stock at present (in $):", min_value=0.0)
    slv = st.number_input("Enter the current Silver price per gram:", min_value=0.0)
    usd = st.number_input("Enter  Euro to US dollar exchange ratio:", min_value=0.0)

    # Create a DataFrame for the input data
    input_parameters = pd.DataFrame([[spx, uso, slv, usd]], columns=["SPX", "USO", "SLV", "EUR/USD"])

    # Predict button
    if st.button("Predict"):
        prediction_output = regressor.predict(input_parameters)
        st.write(f"The predicted value of Gold is: {prediction_output[0]:.2f} per gram")
except:
    print("Exception Occurred")
