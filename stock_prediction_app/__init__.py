"""Stock prediction web application package."""

from flask import Flask

from .views import bp


def create_app() -> Flask:
    """Application factory."""
    app = Flask(__name__)
    app.config.setdefault("SECRET_KEY", "change-me-in-production")

    app.register_blueprint(bp)
    return app

