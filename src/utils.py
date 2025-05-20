import numpy as np
import yaml


def is_pareto_efficient(costs):
    """
    Find the Pareto-efficient points.
    `costs` is a 2D array where each row is a solution, and each column is an objective.
    Returns a boolean array indicating whether each solution is Pareto efficient.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = not np.any(
            np.all(costs <= c, axis=1) & np.any(costs < c, axis=1)
        )
    return is_efficient


def load_parameters_from_yaml(file_path: str) -> dict:
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data["parameters"]
