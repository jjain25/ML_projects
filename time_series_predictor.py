import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import mysql.connector
from datetime import datetime
from mysql.connector import Error


            

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

# Forecasting Models
def arima_forecast(data, p, d, q, forecast_horizon=30):
    model = ARIMA(data['Close'], order=(p, d, q))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_horizon)
    return forecast, model_fit

def sarimax_forecast(data, p, d, q, P, D, Q, S, forecast_horizon=30):
    model = SARIMAX(data['Close'], order=(p, d, q), seasonal_order=(P, D, Q, S))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=forecast_horizon)
    return forecast, model_fit

def auto_arima_forecast(data, forecast_horizon=30):
    model = auto_arima(data['Close'], seasonal=True, stepwise=True, trace=True)
    forecast = model.predict(n_periods=forecast_horizon)
    return forecast, model

def exponential_smoothing_forecast(data, trend='add', seasonal='add', period=12, alpha=None, forecast_horizon=30):
    try:
        model = ExponentialSmoothing(data['Close'], trend=trend, seasonal=seasonal, seasonal_periods=period)
        
        # Fit the model with or without alpha
        if alpha:
            model_fit = model.fit(smoothing_level=alpha)
        else:
            model_fit = model.fit()
        
        forecast = model_fit.forecast(steps=forecast_horizon)
        return forecast, model_fit
    except Exception as e:
        st.error(f"Error in Exponential Smoothing: {e}")
        return None, None




def store_user_input_to_db(ticker, start_date, end_date, model_choice, parameters):
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
            # Fixing the table name to user_timeseries_data
            query = """
            INSERT INTO user_timeseries_data (ticker_name, start_date, end_date, `current_date`, model_selected, parameters)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            current_date = datetime.now().strftime('%Y-%m-%d')
            values = (ticker, start_date, end_date, current_date, model_choice, str(parameters))
            cursor.execute(query, values)
            connection.commit()
            st.success("User forecast data saved to the database successfully!")
        else:
            st.error("Connection to the database could not be established.")
    except mysql.connector.Error as e:
        st.error(f"Error while connecting to MySQL: {e}")
    finally:
        # Ensure proper cleanup of the connection and cursor
        if connection and connection.is_connected():
            cursor.close()
            connection.close()


def plot_forecast(data, forecast, model_name="Model"):
    # Debug: Display forecast data and dates
    st.write("Forecast Length:", len(forecast))
    st.write("Data Index Last Date:", data.index[-1])
    
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Historical Data'
    )])

    forecast_dates = pd.date_range(start=data.index[-1], periods=len(forecast)+1, freq='B')[1:]
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast,
        mode='lines',
        name=f'{model_name} Forecast',
        line=dict(color='blue', dash='dash')
    ))

    fig.update_layout(
        title=f"{model_name} Forecast",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        template="plotly_dark"
    )
    return fig


def plot_acf_pacf(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_acf(data['Close'], ax=axes[0], lags=40)
    axes[0].set_title('ACF Plot')

    plot_pacf(data['Close'], ax=axes[1], lags=40)
    axes[1].set_title('PACF Plot')

    # Display the ACF and PACF plots in Streamlit
    st.pyplot(fig)

if __name__ == "__main__":
    st.title("📈 Equity and Index Data Forecasting With Time Series Models")
    st.sidebar.header("🔍 Data Selection")

    fetcher = Fetcher()

    # Sidebar configuration
    choice = st.sidebar.selectbox("Select Data Type", ["Stock", "Index"], key="choice")
    if choice == "Stock":
        ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., RELIANCE):", key="stock_ticker")
        ticker_with_ns = ticker + ".NS"
    else:
        index_name = st.sidebar.selectbox("Select an Index:", list(fetcher.indian_indices_tickers.keys()), key="index")
        ticker_with_ns = fetcher.indian_indices_tickers[index_name]

    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-01-01"), key="start_date")
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"), key="end_date")

    # Fetching data
    stock_data = fetcher.get_stock_or_index_data(ticker_with_ns, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if stock_data is not None:
        st.write("### Historical Data")
        st.dataframe(stock_data)

        # ACF and PACF plot option
        show_acf_pacf = st.sidebar.checkbox("Show ACF and PACF Plots", value=True)

        # ACF and PACF plots
        if show_acf_pacf:
            st.write("### ACF and PACF Plots")
            plot_acf_pacf(stock_data)

        # Initialize parameters
        parameters = {}
        model_choice = st.sidebar.selectbox(
            "Select Model", ["ARIMA", "SARIMAX", "Auto ARIMA", "Exponential Smoothing"], key="model_choice"
        )

        # Model parameter inputs
        if model_choice == "ARIMA":
            p = st.sidebar.slider("p (Auto-regressive Order)", 1, 10, 5, key="p")
            d = st.sidebar.slider("d (Differencing Order)", 0, 3, 1, key="d")
            q = st.sidebar.slider("q (Moving Average Order)", 1, 10, 5, key="q")
            parameters = {"p": p, "d": d, "q": q}

        elif model_choice == "SARIMAX":
            p = st.sidebar.slider("p (Auto-regressive Order)", 1, 10, 5, key="p")
            d = st.sidebar.slider("d (Differencing Order)", 0, 3, 1, key="d")
            q = st.sidebar.slider("q (Moving Average Order)", 1, 10, 5, key="q")
            P = st.sidebar.slider("P (Seasonal Auto-regressive Order)", 0, 5, 1, key="P")
            D = st.sidebar.slider("D (Seasonal Differencing Order)", 0, 2, 1, key="D")
            Q = st.sidebar.slider("Q (Seasonal Moving Average Order)", 0, 5, 1, key="Q")
            S = st.sidebar.slider("S (Seasonal Periodicity)", 1, 12, 12, key="S")
            parameters = {"p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q, "S": S}

        elif model_choice == "Exponential Smoothing":
            trend = st.sidebar.selectbox("Trend", ["add", "mul"], index=0, key="trend")
            seasonal = st.sidebar.selectbox("Seasonal", ["add", "mul"], index=0, key="seasonal")
            period = st.sidebar.slider("Seasonal Periodicity", 1, 12, 12, key="seasonal_period")
            alpha = st.sidebar.slider("Alpha (Smoothing Level)", 0.01, 1.0, 0.1, step=0.01, key="alpha")
            parameters = {"trend": trend, "seasonal": seasonal, "period": period, "alpha": alpha}

        forecast_horizon = st.sidebar.number_input("Forecast Horizon (Days)", min_value=1, max_value=365, value=30, key="forecast_horizon")
        parameters["forecast_horizon"] = forecast_horizon

        # Forecast generation
        forecast, model_fit = None, None
        try:
            if model_choice == "ARIMA":
                forecast, model_fit = arima_forecast(stock_data, p, d, q, forecast_horizon)
            elif model_choice == "SARIMAX":
                forecast, model_fit = sarimax_forecast(stock_data, p, d, q, P, D, Q, S, forecast_horizon)
            elif model_choice == "Auto ARIMA":
                forecast, model_fit = auto_arima_forecast(stock_data, forecast_horizon)
            elif model_choice == "Exponential Smoothing":
                forecast, model_fit = exponential_smoothing_forecast(
                    stock_data, trend=trend, seasonal=seasonal, period=period, alpha=alpha, forecast_horizon=forecast_horizon
                )
        except Exception as e:
            st.error(f"Error generating forecast: {e}")

        # Display output
        if forecast is not None and model_fit is not None:
            st.write(f"### {model_choice} Model Summary")
            try:
                st.text(model_fit.summary())
            except AttributeError:
                st.write("Model summary not available for this model.")

            # Plot forecast
            try:
                fig = plot_forecast(stock_data, forecast, model_name=model_choice)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating plot: {e}")

            # Display forecasted data
            forecast_dates = pd.date_range(start=stock_data.index[-1], periods=len(forecast) + 1, freq='B')[1:]
            forecast_df = pd.DataFrame({
                "Date": forecast_dates.strftime('%d-%m-%Y'),
                "Forecasted Prices": forecast
            })
            st.write("### Forecasted Data")
            st.dataframe(forecast_df)

            # Download button
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Forecast Data as CSV",
                data=csv,
                file_name="forecasted_data.csv",
                mime="text/csv"
            )
        else:
            st.warning("No forecast generated. Please check your input parameters.")

        # Save data option
        if st.button("Save Data"):
            store_user_input_to_db(ticker_with_ns, start_date, end_date, model_choice, str(parameters))
            st.success("Data saved successfully!")
