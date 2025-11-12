# Stock Prediction Studio

Flask web app for forecasting stock prices or rolling volatility using Yahoo Finance data. Users choose a ticker, sampling interval, forecast horizon, and modeling approach (Prophet or LSTM). The app displays summary stats, evaluation metrics, and a forecast table in a dark-themed dashboard.

## Quick Start

```bash
python3 -m pip install -r requirements.txt
python3 stock_prediction_app/app.py
```

Open `http://127.0.0.1:5000/` to use the interface.

## Features

- Fetches historical data via `yfinance`.
- Predicts either adjusted close price or annualized 21-day volatility.
- Two models:
  - **Prophet** for decomposable time-series forecasting with seasonality controls.
  - **LSTM** neural network trained on lookback windows for sequence prediction.
- Dark-mode UI highlights latest actual value, next-day forecast, horizon forecast, and error metrics (MAE, RMSE, MAPE).
- Forecast table lists expected values (with Prophet confidence intervals when available).

## Deploy to Render

1. Push this repository to GitHub (or another Git host).
2. The included `render.yaml` describes a free-tier Render web service.
3. On [Render](https://render.com):
   - Create a **New Web Service** linked to your repo.
   - Render will automatically use the build/start commands from `render.yaml`
     - Build: `pip install -r requirements.txt`
     - Start: `gunicorn stock_prediction_app.app:app`
4. Deploy and share the Render-provided URL.

## Project Structure

```
.
├── requirements.txt
├── render.yaml
└── stock_prediction_app
    ├── __init__.py
    ├── app.py
    ├── data.py
    ├── modeling.py
    ├── static
    │   └── css
    │       └── styles.css
    └── templates
        ├── base.html
        ├── forecast.html
        └── index.html
```

## Notes

- Prophet’s first run may take a minute while CmdStan toolchain compiles.
- LSTM training uses TensorFlow; macOS Apple Silicon users install via `tensorflow-macos` (handled in `requirements.txt`).
- Adjust environment variables (`HOST`, `PORT`, `FLASK_DEBUG`) as needed; Render sets `PORT` automatically.
