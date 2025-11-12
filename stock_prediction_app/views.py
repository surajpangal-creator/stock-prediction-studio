"""Flask views for stock prediction web app."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

import pandas as pd
from flask import Blueprint, flash, redirect, render_template, request, url_for

from .data import compute_target_series, fetch_market_data
from .modeling import ForecastResult, run_lstm_forecast, run_prophet_forecast

bp = Blueprint("stock_prediction", __name__)


@bp.route("/", methods=["GET", "POST"])
def index():
    min_start = datetime.today() - timedelta(days=365 * 5)
    default_start = (datetime.today() - timedelta(days=365 * 2)).date()
    default_end = datetime.today().date()

    if request.method == "POST":
        form = request.form
        ticker = form.get("ticker", "").strip().upper()
        model_choice: Literal["prophet", "lstm"] = form.get("model", "prophet")  # type: ignore[assignment]
        target_choice: Literal["close", "volatility"] = form.get("target", "close")  # type: ignore[assignment]
        horizon = int(form.get("horizon", 30))
        interval = form.get("interval", "1d")
        seasonality = form.get("seasonality", "auto")

        try:
            start_date = datetime.strptime(form.get("start_date", str(default_start)), "%Y-%m-%d")
            end_date = datetime.strptime(form.get("end_date", str(default_end)), "%Y-%m-%d")
        except ValueError:
            flash("Invalid date format. Please use YYYY-MM-DD.", "error")
            return redirect(url_for("stock_prediction.index"))

        if not ticker:
            flash("Please provide a ticker symbol.", "error")
            return redirect(url_for("stock_prediction.index"))

        if start_date >= end_date:
            flash("Start date must be before end date.", "error")
            return redirect(url_for("stock_prediction.index"))

        try:
            prices = fetch_market_data(ticker, start=start_date, end=end_date, interval=interval)
            target_series = compute_target_series(prices, target_choice)
            historical_series = target_series.copy()

            if model_choice == "prophet":
                result = run_prophet_forecast(target_series, periods=horizon, seasonality=seasonality)
            else:
                result = run_lstm_forecast(target_series, periods=horizon)

            latest_actual = float(historical_series.iloc[-1])
            next_forecast = float(result.forecast.iloc[0]["forecast"])
            forecast_end = float(result.forecast.iloc[-1]["forecast"])
            delta_next = next_forecast - latest_actual
            delta_pct = (delta_next / latest_actual) * 100 if latest_actual else None

            return render_template(
                "forecast.html",
                ticker=ticker,
                model=model_choice,
                target=target_choice,
                horizon=horizon,
                interval=interval,
                result=result,
                latest_actual=latest_actual,
                next_forecast=next_forecast,
                forecast_end=forecast_end,
                delta_next=delta_next,
                delta_pct=delta_pct,
            )
        except Exception as exc:  # pylint: disable=broad-except
            flash(str(exc), "error")
            return redirect(url_for("stock_prediction.index"))

    return render_template(
        "index.html",
        min_start=min_start.date(),
        default_start=default_start,
        default_end=default_end,
    )

