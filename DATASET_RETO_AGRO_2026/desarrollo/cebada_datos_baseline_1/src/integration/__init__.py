"""Módulo de integración AgroCebada 2026."""
from .build import build
from .validate import validate_all

__all__ = ["build", "validate_all", "load_predictor", "load_xai"]

from .predictor_adapter import load_predictor, PredictorAdapter
from .xai_adapter import load_xai, XAIAdapter

__version__ = "1.0.0"