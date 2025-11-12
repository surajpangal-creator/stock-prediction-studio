"""Model training utilities for forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
import pandas as pd
from prophet import Prophet
from pandas.tseries.frequencies import to_offset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


@dataclass
class ForecastResult:
    model: Literal["prophet", "lstm"]
    target: Literal["close", "volatility"]
    fitted_values: pd.Series
    forecast: pd.DataFrame
    metrics: Dict[str, float]


def _calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    common_index = y_true.index.intersection(y_pred.index)
    if common_index.empty:
        return {}
    actual = pd.Series(np.asarray(y_true.loc[common_index]).reshape(-1), index=common_index)
    predicted = pd.Series(np.asarray(y_pred.loc[common_index]).reshape(-1), index=common_index)
    aligned = pd.DataFrame({"actual": actual, "predicted": predicted})
    aligned = aligned.dropna()
    if aligned.empty:
        return {}
    mae = mean_absolute_error(aligned["actual"], aligned["predicted"])
    mse = mean_squared_error(aligned["actual"], aligned["predicted"])
    rmse = float(np.sqrt(mse))
    mape = np.mean(
        np.abs((aligned["actual"] - aligned["predicted"]) / np.clip(np.abs(aligned["actual"]), 1e-8, None))
    )
    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(float(mape * 100), 2),
    }


def run_prophet_forecast(
    series: pd.Series, periods: int, seasonality: Literal["auto", "weekly", "monthly"]
) -> ForecastResult:
    df = series.reset_index()
    df.columns = ["ds", "y"]

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=seasonality in {"auto", "weekly"},
        yearly_seasonality=True,
    )
    if seasonality == "auto":
        model.daily_seasonality = True
    if seasonality == "monthly":
        model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    model.fit(df)

    future = model.make_future_dataframe(periods=periods)
    forecast_df = model.predict(future).set_index("ds")
    fitted = forecast_df.loc[df["ds"], "yhat"]
    forecast = forecast_df.tail(periods)[["yhat", "yhat_lower", "yhat_upper"]]
    forecast.columns = ["forecast", "lower", "upper"]

    metrics = _calculate_metrics(series, fitted)
    return ForecastResult(
        model="prophet",
        target="close" if series.name == "price" else "volatility",
        fitted_values=pd.Series(fitted, index=df["ds"], name="fitted"),
        forecast=forecast,
        metrics=metrics,
    )


def run_lstm_forecast(
    series: pd.Series,
    periods: int,
    lookback: int = 60,
    epochs: int = 20,
) -> ForecastResult:
    values = series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, y = _build_sequences(scaled, lookback)
    if len(X) < 10:
        raise ValueError("Not enough data points for LSTM training.")

    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = _build_lstm_model(input_shape=(lookback, 1))
    callbacks = _maybe_get_callbacks()
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
        callbacks=callbacks,
    )

    fitted_scaled = model.predict(X, verbose=0)
    fitted_values = scaler.inverse_transform(fitted_scaled)
    fitted_series = pd.Series(
        fitted_values.flatten(),
        index=series.index[lookback:],
        name="fitted",
    )

    forecast_points = _iterative_forecast(model, scaler, scaled, lookback, periods)

    inferred_freq = pd.infer_freq(series.index)
    if inferred_freq is None and len(series.index) > 1:
        step = series.index[-1] - series.index[-2]
        inferred_freq = to_offset(step).freqstr
    freq = inferred_freq or "D"

    forecast_index = pd.date_range(
        start=series.index[-1] + to_offset(freq),
        periods=periods,
        freq=freq,
    )
    forecast_df = pd.DataFrame(
        {
            "forecast": forecast_points,
        },
        index=forecast_index,
    )

    test_index = series.index[lookback:][split_index:]
    test_truth = series.loc[test_index]
    test_pred = fitted_series.loc[test_index]
    metrics = _calculate_metrics(test_truth, test_pred)

    return ForecastResult(
        model="lstm",
        target="close" if series.name == "price" else "volatility",
        fitted_values=fitted_series,
        forecast=forecast_df,
        metrics=metrics,
    )


def _build_sequences(data: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback : i])
        y.append(data[i])
    return np.array(X), np.array(y)


def _build_lstm_model(input_shape: tuple[int, int]):
    from tensorflow.keras import Input, Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM

    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def _maybe_get_callbacks():
    from tensorflow.keras.callbacks import EarlyStopping

    return [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]


def _iterative_forecast(
    model,
    scaler: MinMaxScaler,
    scaled_series: np.ndarray,
    lookback: int,
    periods: int,
) -> np.ndarray:
    history = scaled_series[-lookback:].reshape(1, lookback, 1)
    outputs = []

    for _ in range(periods):
        next_point = model.predict(history, verbose=0)[0, 0]
        outputs.append(next_point)
        history = np.append(history[:, 1:, :], [[[next_point]]], axis=1)

    outputs = np.array(outputs).reshape(-1, 1)
    return scaler.inverse_transform(outputs).flatten()

