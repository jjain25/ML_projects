import streamlit as st

# Set the app title and layout
st.set_page_config(page_title="Multi-Page Forecasting App", layout="wide")

# Default home page
st.title("Welcome to the Multi-Page Forecasting App")
st.write(
    "Explore forecasting using different models. Interact with the widgets below to see forecasting in action!"
)

# Add an interactive slider for selecting a year range for forecasting
year_range = st.slider(
    "Select the year range for forecasting:",
    min_value=2020, max_value=2030, step=1,
    value=(2022, 2025)
)
st.write(f"You selected the range: {year_range[0]} to {year_range[1]}")

# Interactive text input for user to enter a custom time-series
custom_series_input = st.text_area(
    "Enter a custom time-series data (comma-separated values):",
    "100, 120, 130, 140, 150"
)
if custom_series_input:
    st.write("You entered the following time-series data:")
    st.write(custom_series_input)

# Interactive button for generating a forecast based on the input data
if st.button("Generate Forecast"):
    st.write("Generating forecast for the selected time-series data...")
    # Here, you would add logic to process the input data and generate a forecast.
    st.write("Forecasting results will appear here.")

# Add more interactive elements, like a dropdown or radio buttons, for different forecasting models
model_choice = st.radio(
    "Select a forecasting model to use:",
    ["ARIMA", "Exponential Smoothing", "Prophet", "Custom Model"]
)
st.write(f"You selected the {model_choice} model.")

# Example for dynamic chart or plot (using random data as an example)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate some random data for demonstration purposes
dates = pd.date_range("2022-01-01", periods=100, freq="D")
values = np.random.rand(100) * 100

# Create a plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(dates, values, label="Forecast Data")
ax.set_title("Example Forecast Plot")
ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.legend()

# Display the plot in the app
st.pyplot(fig)

# Add a markdown section to explain the features
st.markdown(
    "### How to use the app"
    "\n1. Use the slider to select a year range for forecasting."
    "\n2. Enter your custom time-series data in the input box."
    "\n3. Click 'Generate Forecast' to simulate a forecasting process."
    "\n4. Select a forecasting model to see how the results may vary."
)
