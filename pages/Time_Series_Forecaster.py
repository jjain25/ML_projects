import streamlit as st
from time_series_predictor import *

st.title("⏳ Time-Series Forecasting Models")
st.write("Analyze equity data using ARIMA, SARIMAX,Auto ARIMA AND EXPONENTIAL SMOOTHING")

# Include the content of the main function from `time_series_predictor.py` here
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
