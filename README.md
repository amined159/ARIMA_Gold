# ARIMA Gold Price Forecast

This project forecasts gold prices using an ARIMA model. It includes a Python script and a Jupyter notebook version of the same workflow.

## Files
- `GoldPrices.csv`: input dataset
- `arima_model.py`: ARIMA workflow as a script
- `arima_model.ipynb`: ARIMA workflow as a notebook

## Quick Start
Run the script:
```bash
python arima_model.py
```

Or open the notebook:
```bash
jupyter notebook arima_model.ipynb
```

## What It Does
1. Loads and preprocesses the data.
2. Splits into train/test sets.
3. Checks stationarity and applies differencing.
4. Fits an ARIMA(1,1,1) model.
5. Forecasts the test window and plots results.
6. Reports MAE, MSE, and RMSE.
