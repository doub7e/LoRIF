"""LoRIF: Low-Rank Influence Functions for training data attribution."""

from .collector import GradientCollector, LayerSpec
from .curvature import CurvatureMode, CurvatureModel, fit_curvature
from .factors import Factors, factorize, factor_inner_products, project_factors
from .influence import influence_scores
from .lds import LDSResult, linear_datamodeling_score
from .store import FactorStore, FactorWriter

__all__ = [
    "CurvatureMode",
    "CurvatureModel",
    "FactorStore",
    "FactorWriter",
    "Factors",
    "GradientCollector",
    "LDSResult",
    "LayerSpec",
    "factor_inner_products",
    "factorize",
    "fit_curvature",
    "influence_scores",
    "linear_datamodeling_score",
    "project_factors",
]

__version__ = "0.1.0"
