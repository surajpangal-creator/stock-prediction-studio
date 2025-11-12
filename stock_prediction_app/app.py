"""Standalone runner for the stock prediction Flask app."""

from __future__ import annotations

import os

from stock_prediction_app import create_app

app = create_app()


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    debug_env = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug_env)

