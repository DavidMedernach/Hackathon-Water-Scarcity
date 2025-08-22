"""Predition functions for different model types."""
from typing import Callable, List, Any
import numpy as np


def create_predict_function(models: List[Any], i: int, model_name: str) -> Callable:
    """Create a prediction function based on the specified model type.

    Args:
        model_list (List[Any]): A list of trained models.
        i (int): The index of the model to use from the list.
        model (str): The type of model, either 'mapie' or other types.

    Returns:
        Callable: A function that takes input data X and returns predictions.
    """

    def predict(X):
        if model_name == "mapie":
            return models[i].predict(X)[0]
        elif model_name == "deep_ensemble":
            y_pred_deep = []
            for m in models[i]:
                y_pred_deep.append(m.predict(X))
            y_pred_deep = np.array(y_pred_deep)
            return np.mean(y_pred_deep, axis=0)
        elif model_name == "ebm":
            y_pred_ebm = []
            for m in models[i]:
                y_pred_ebm.append(m.predict(X))
            y_pred_ebm = np.array(y_pred_ebm)
            return np.mean(y_pred_ebm, axis=0)
        else:
            return models[i].predict(X)

    return predict


def create_quantile_function(
    models: List[Any], i: int, model: str, a: float = 0.1
) -> Callable:
    """Create a quantile prediction function based on the specified model type.

    Args:
        model_list (List[Any]): A list of trained models.
        i (int): The index of the model to use from the list.
        model (str): The type of model, either 'mapie' or 'qrf'.
        alpha (float): The confidence level for the quantile prediction.

    Returns:
        Callable: A function that takes input data X
        and returns quantile predictions.
    """

    def predict_quantile(X):
        if model == "mapie":
            return models[i].predict(X)[1]
        elif model == "qrf" or model == "qrf_100":
            return models[i].predict(X, quantiles=[a / 2, 1 - a / 2])
        elif model == "deep_ensemble":
            y_pred_deep = []
            for m in models[i]:
                y_pred_deep.append(m.predict(X))
            y_pred_deep = np.array(y_pred_deep)
            return np.quantile(y_pred_deep, [a / 2, 1 - a / 2], axis=0)
        elif model == "ebm":
            y_pred_ebm = []
            for m in models[i]:
                y_pred_ebm.append(m.predict(X))
            y_pred_ebm = np.array(y_pred_ebm)
            return np.quantile(y_pred_ebm, [a / 2, 1 - a / 2], axis=0).T
        raise ValueError(f"Unsupported model type: {model}")

    return predict_quantile
