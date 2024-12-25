from tkinter import TRUE
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
import math
import ta
import mysql.connector
from mysql.connector import Error
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima



class Fetcher:
    indian_indices_tickers = {
        "Nifty 50": "^NSEI",
        "Nifty Bank": "^NSEBANK",
        "Nifty IT": "^CNXIT",
        "Nifty Next 50": "^NSENEXT50",
        "Nifty Midcap 50": "^CNXMDCP",
        "Nifty Smallcap 100": "^CNXSMALL",
        "BSE Sensex": "^BSESN",
        "Nifty 500": "^CNX500",
        "Nifty FMCG": "^CNXFMCG",
        "Nifty Auto": "^CNXAUTO",
        "Nifty Metal": "^CNXMETAL",
        "Nifty Pharma": "^CNXPHARMA",
        "Nifty Realty": "^CNXREALTY",
        "Nifty Energy": "^CNXENERGY",
        "Nifty Media": "^CNXMEDIA",
        "Nifty PSU Bank": "^CNXPSUBANK",
        "Nifty Private Bank": "^CNXPRIVATEBANK",
        "Nifty Infrastructure": "^CNXINFRA",
        "Nifty Financial Services": "^CNXFINANCE",
        "Nifty Commodities": "^CNXCOMMODITIES",
        "Nifty Services Sector": "^CNXSERVICES",
        "Nifty MNC": "^CNXMNC",
        "Nifty Growth Sectors 15": "^CNXGROWTHSECTORS15",
        "Nifty Dividend Opportunities 50": "^CNXDIVOPPORTUNITIES50",
        "Nifty 100": "^CNX100",
        "Nifty 200": "^CNX200",
    }

    def get_stock_or_index_data(self, ticker_with_ns, start_date, end_date):
        stock = yf.Ticker(ticker_with_ns)
        stock_data = stock.history(start=start_date, end=end_date)

        if not stock_data.empty:
            stock_data.index = stock_data.index.strftime('%d-%m-%Y')
            return stock_data
        else:
            return None

class ForecastingModels:
    def __init__(self, data):
        self.data = data

    def prepare_data(self, forecast_horizon=30, test_size=0.2):
        data = self.data.copy()
        data['Date'] = pd.to_datetime(data.index, format='%d-%m-%Y')

        for i in range(1, forecast_horizon + 1):
            data[f'lag_{i}'] = data['Close'].shift(i)

        data.dropna(inplace=True)
        X = data.drop(['Close', 'Date'], axis=1)
        y = data['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_linear_regression(self, X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def train_random_forest(self, X_train, y_train, n_estimators=100, max_depth=None):
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        return model

    def train_svr(self, X_train, y_train, C=1.0, epsilon=0.1):
        model = SVR(kernel='rbf', C=C, epsilon=epsilon)
        model.fit(X_train, y_train)
        return model

    def train_xgboost(self, X_train, y_train, n_estimators=100, learning_rate=0.1):
        model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
        model.fit(X_train, y_train)
        return model

    def train_polynomial_regression(self, X_train, y_train, degree=2):
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X_train)
        model = LinearRegression()
        model.fit(X_poly, y_train)
        return model, poly_features
    
    def train_lstm(self, data, look_back=90, epochs=20, batch_size=32):
        data = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        training_data_len = math.ceil(len(scaled_data) * 0.8)
        train_data = scaled_data[:training_data_len, :]

        X_train, y_train = [], []
        for i in range(look_back, len(train_data)):
            X_train.append(train_data[i - look_back:i, 0])
            y_train.append(train_data[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

        return model, scaler, scaled_data, training_data_len
    
    
    def evaluate_model(self, model, X_test, y_test, lstm=False):
        if lstm:
            predictions = model.predict(X_test)
            predictions = predictions.flatten()
        else:
            predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        return mse, mae, predictions

    def forecast_future(self, model, X, forecast_horizon=30, lstm=False):
        if lstm:
            predictions = model.predict(X)
            predictions = predictions.flatten()
        else:
            predictions = model.predict(X[-forecast_horizon:])
        return predictions


def store_user_input_to_db(ticker, start_date, end_date, forecast_horizon, test_size, model_choice, hyperparameter_used, date_used):
    connection = None  # Initialize connection variable
    try:
        # Establish the MySQL connection
        connection = mysql.connector.connect(
            host="localhost",
            port="3306",   
            user="root",
            password="root",
            database="user_input",
            auth_plugin='mysql_native_password'
        )

        if connection and connection.is_connected():
            cursor = connection.cursor()
            # Fixing the missing comma in the query
            query = """
            INSERT INTO user_inputs (ticker, start_date, end_date, forecast_horizon, test_size, model_choice, hyperparameter_used, date_used)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (ticker, start_date, end_date, forecast_horizon, test_size, model_choice, str(hyperparameter_used), date_used)
            cursor.execute(query, values)
            connection.commit()
            st.success("User input saved to the database successfully!")
        else:
            st.error("Connection to the database could not be established.")
    except mysql.connector.Error as e:
        st.error(f"Error while connecting to MySQL: {e}")
    finally:
        # Ensure proper cleanup of the connection and cursor
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def plot_with_indicators(data):
    if 'Close' not in data.columns:
        st.error("Close price data is missing.")
        return None

    # User-defined parameters for indicators
    st.sidebar.write("### Customize Indicators")
    
    sma_window = st.sidebar.slider("SMA Window", min_value=5, max_value=200, value=50)
    ema_window = st.sidebar.slider("EMA Window", min_value=5, max_value=200, value=50)
    rsi_window = st.sidebar.slider("RSI Window", min_value=5, max_value=50, value=14)
    bb_window = st.sidebar.slider("Bollinger Bands Window", min_value=5, max_value=50, value=20)
    bb_std_dev = st.sidebar.slider("Bollinger Bands Standard Deviation", min_value=1, max_value=4, value=2)

    # Compute indicators with user-defined parameters
    data['SMA_Custom'] = data['Close'].rolling(window=sma_window).mean()
    data['EMA_Custom'] = ta.trend.ema_indicator(data['Close'], window=ema_window)
    data['RSI_Custom'] = ta.momentum.rsi(data['Close'], window=rsi_window)

    bb = ta.volatility.BollingerBands(data['Close'], window=bb_window, window_dev=bb_std_dev)
    data['Upper_BB_Custom'] = bb.bollinger_hband()
    data['Lower_BB_Custom'] = bb.bollinger_lband()

    # Plotting the indicators
    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick'
    ))

    # SMA
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['SMA_Custom'],
        mode='lines',
        name=f'SMA {sma_window}',
        line=dict(color='green')
    ))

    # EMA
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['EMA_Custom'],
        mode='lines',
        name=f'EMA {ema_window}',
        line=dict(color='purple')
    ))

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Upper_BB_Custom'],
        mode='lines',
        name=f'Upper BB ({bb_window}, {bb_std_dev}σ)',
        line=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Lower_BB_Custom'],
        mode='lines',
        name=f'Lower BB ({bb_window}, {bb_std_dev}σ)',
        line=dict(color='blue')
    ))

    # RSI
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['RSI_Custom'],
        mode='lines',
        name=f'RSI ({rsi_window})',
        line=dict(color='orange')
    ))

    # Update layout
    fig.update_layout(
        title="Stock Price with Customized Technical Indicators",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        template="plotly_dark"
    )
    
    return fig

def plot_model_output(data, predictions, forecasted_prices):
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick'
    )])

    fig.add_trace(go.Scatter(
        x=data.index[-len(predictions):],
        y=predictions,
        mode='lines',
        name='Model Predictions',
        line=dict(color='orange')
    ))

    forecast_dates = pd.date_range(start=data.index[-1], periods=len(forecasted_prices)+1, freq='B')[1:]
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecasted_prices,
        mode='lines',
        name='Forecasted Prices',
        line=dict(color='blue', dash='dash')
    ))

    fig.update_layout(
        title="Model Predictions and Forecasted Prices",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        template="plotly_dark"
    )
    return fig
def display_forecasted_data(actual, predictions, forecast):
    # Align lengths of actual and predictions (test data)
    actual = actual[:len(predictions)]

    # Align forecast index
    forecast_index_start = len(actual)  # Start index for forecasted prices
    forecast_index_end = forecast_index_start + len(forecast)  # End index for forecasted prices

    # Create DataFrame
    forecast_table = pd.DataFrame({
        "Actual Prices": pd.Series(actual, index=range(len(actual))),
        "Predicted Prices": pd.Series(predictions, index=range(len(predictions))),
        "Forecasted Prices": pd.Series(forecast, index=range(forecast_index_start, forecast_index_end))
    })

    st.write("### Forecasted Data")
    st.dataframe(forecast_table)


# Assuming the rest of your code is intact
if __name__ == "__main__":
    st.title("📈Equity and Index Data Forecasting with ML Model")
    st.sidebar.header("🔍 Data Selection")

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




