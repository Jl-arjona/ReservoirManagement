# framework/ml/penalty_models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score


@dataclass
class TrainedPenaltyModel:
    """
    Envoltorio: modelo de ML entrenado para predecir penalizaciones
    a partir de features del problema.
    """
    model_name: str
    feature_names: List[str]
    param_names: List[str]
    model: MultiOutputRegressor

    def predict_params(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Devuelve un dict {param_name: predicted_value}.
        """
        x = np.array([[features[name] for name in self.feature_names]], dtype=float)
        y_pred = self.model.predict(x)[0]
        return {name: float(val) for name, val in zip(self.param_names, y_pred)}


def train_best_penalty_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    param_names: List[str],
    candidate_models: Dict[str, object] | None = None,
    cv: int = 3,
) -> TrainedPenaltyModel:
    """
    Entrena varios modelos y selecciona el mejor por validación cruzada (MSE).

    X: shape (n_samples, n_features)
    y: shape (n_samples, n_params)
    """
    if candidate_models is None:
        candidate_models = {
            "random_forest": RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                random_state=0,
                n_jobs=-1,
            ),
            "extra_trees": ExtraTreesRegressor(
                n_estimators=200,
                random_state=0,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                random_state=0,
            ),
        }

    best_name = None
    best_score = float("-inf")
    best_model = None

    for name, base_model in candidate_models.items():
        m = MultiOutputRegressor(clone(base_model))
        scores = cross_val_score(
            m, X, y,
            cv=cv,
            scoring="neg_mean_squared_error",
        )
        mean_score = float(scores.mean())
        print(f"[ML] Modelo {name}: score (neg MSE) = {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_model = m

    assert best_model is not None
    best_model.fit(X, y)

    print(f"[ML] Mejor modelo: {best_name} (score={best_score:.4f})")
    return TrainedPenaltyModel(
        model_name=best_name,
        feature_names=feature_names,
        param_names=param_names,
        model=best_model,
    )
