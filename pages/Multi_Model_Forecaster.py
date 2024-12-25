import streamlit as st
from predictor_multi_model import *

st.title("📈 Multi-Model Forecasting with Machine Learning")
st.write("Leverage various ML models to forecast stock and index data.")

# Include the content of the main function from `predictor_multi_model.py` here
if __name__ == "__main__":
    st.sidebar.header("Multi-Model Forecasting Settings")
    # Logic from `predictor_multi_model.py`
    
fetcher = Fetcher()

choice = st.sidebar.selectbox("Would you like to get data for a Stock or an Index?", ("Stock", "Index"))
if choice == "Stock":
    ticker = st.sidebar.text_input("Enter the stock ticker (e.g., RELIANCE for Reliance Industries): ")
    ticker_with_ns = ticker + ".NS"
else:
    index_name = st.sidebar.selectbox("Choose an index:", list(fetcher.indian_indices_tickers.keys()))
    ticker_with_ns = fetcher.indian_indices_tickers[index_name]

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=180, value=30)
test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.3, value=0.2)

model_choice = st.sidebar.selectbox("Select a ML Model", ["Linear Regression", "Random Forest", "SVR", "XGBoost", "LSTM"])

# Hyperparameters input
hyperparameter_used = {}

if model_choice == "Random Forest":
    n_estimators = st.slider("Number of Estimators", min_value=10, max_value=200, value=100)
    max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=None)
    hyperparameter_used = {"n_estimators": n_estimators, "max_depth": max_depth}

elif model_choice == "SVR":
    C = st.slider("C", min_value=0.1, max_value=10.0, value=1.0)
    epsilon = st.slider("Epsilon", min_value=0.01, max_value=1.0, value=0.1)
    hyperparameter_used = {"C": C, "epsilon": epsilon}

elif model_choice == "XGBoost":
    n_estimators = st.slider("Number of Estimators", min_value=10, max_value=200, value=100)
    learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=1.0, value=0.1)
    hyperparameter_used = {"n_estimators": n_estimators, "learning_rate": learning_rate}

elif model_choice == "Polynomial Regression":
    degree = st.slider("Degree for Polynomial Regression", min_value=2, max_value=5, value=2)
    hyperparameter_used = {"degree": degree}

elif model_choice == "LSTM":
    lookback_period = st.slider("Lookback Period for LSTM", min_value=1, max_value=60, value=30)
    lstm_epochs = st.slider("Epochs for LSTM", min_value=10, max_value=100, value=20)
    lstm_batch_size = st.slider("Batch Size for LSTM", min_value=16, max_value=64, value=32)
    hyperparameter_used = {"lookback_period": lookback_period, "lstm_epochs": lstm_epochs, "lstm_batch_size": lstm_batch_size}

# Fetch the stock data
stock_data = fetcher.get_stock_or_index_data(ticker_with_ns, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

# Add a "Save" button
if stock_data is not None:
    save_button = st.button("Save User Input to Database")
    
    # If the "Save" button is clicked
    if save_button:
        date_used = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        store_user_input_to_db(
            ticker_with_ns,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            forecast_horizon,
            test_size,
            model_choice,
            hyperparameter_used,  # Store the hyperparameters as a dictionary
            date_used
        )
        st.success("User input saved successfully to the database!")

# Show the stock data and other visualizations
if stock_data is not None:
    st.write("### 📜Historical Stock Data")
    st.dataframe(stock_data)

    st.write("Graphs with Technical Indicators:")
    plot1 = plot_with_indicators(stock_data)
    st.plotly_chart(plot1)
    
    
    forecasting_models = ForecastingModels(stock_data)
    X_train, X_test, y_train, y_test = forecasting_models.prepare_data(forecast_horizon, test_size)

    
    if model_choice == "Linear Regression":
        model = forecasting_models.train_linear_regression(X_train, y_train)
    elif model_choice == "Random Forest":
        model = forecasting_models.train_random_forest(X_train, y_train)
    elif model_choice == "SVR":
        model = forecasting_models.train_svr(X_train, y_train)
    elif model_choice == "XGBoost":
        model = forecasting_models.train_xgboost(X_train, y_train)
    elif model_choice == "Polynomial Regression":
        model, poly_features = forecasting_models.train_polynomial_regression(X_train, y_train, degree=degree)
    elif model_choice == "LSTM":
        model, scaler, scaled_data, training_data_len = forecasting_models.train_lstm(
            stock_data, look_back=lookback_period, epochs=lstm_epochs, batch_size=lstm_batch_size
        )

    if model_choice == "LSTM":
        test_data = scaled_data[training_data_len - lookback_period:, :]
        X_test_lstm = []
        y_test_lstm = stock_data['Close'].values[training_data_len:]

        for i in range(lookback_period, len(test_data)):
            X_test_lstm.append(test_data[i - lookback_period:i, 0])
        X_test_lstm = np.array(X_test_lstm)
        X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

        mse, mae, predictions = forecasting_models.evaluate_model(model, X_test_lstm, y_test_lstm, lstm=True)
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        forecasted_prices = forecasting_models.forecast_future(model, X_test_lstm, forecast_horizon, lstm=True)
        forecasted_prices = scaler.inverse_transform(forecasted_prices.reshape(-1, 1)).flatten()
    else:
        mse, mae, predictions = forecasting_models.evaluate_model(model, X_test, y_test)
        forecasted_prices = forecasting_models.forecast_future(model, X_test, forecast_horizon)

    st.write("### Model Performance")
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"Mean Absolute Error: {mae}")

    st.write("###🔮Predictions and Forecast")
    display_forecasted_data(y_test.values, predictions, forecasted_prices)

    fig = plot_model_output(stock_data, predictions, forecasted_prices)
    st.plotly_chart(fig)
else:
    st.error("No data available for the selected stock or index. Please try again.")
    
    # Add a "Save" button
    if stock_data is not None:
        save_button = st.button("Save User Input to Database")
        
        # If the "Save" button is clicked
        if save_button:
            date_used = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            store_user_input_to_db(
                ticker_with_ns,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                forecast_horizon,
                test_size,
                model_choice,
                hyperparameter_used,  # Store the hyperparameters as a dictionary
                date_used
            )
            st.success("User input saved successfully to the database!")