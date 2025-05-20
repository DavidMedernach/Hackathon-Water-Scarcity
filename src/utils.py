"""Utility functions for Pareto efficiency and loading parameters from a YAML file."""
import numpy as np
import yaml


def is_pareto_efficient(costs):
    """Find the Pareto-efficient points.

    Args:
        costs (np.ndarray): Array of shape (n_points, n_objectives) representing
                            the cost of each point.

    Returns:
        np.ndarray: Boolean array indicating whether each point is Pareto-efficient.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = not np.any(
            np.all(costs <= c, axis=1) & np.any(costs < c, axis=1)
        )
    return is_efficient


def load_parameters_from_yaml(file_path: str) -> dict:
    """Load parameters from a YAML file.

    Args:
        file_path (str): Path to the YAML file.
        
    Returns:
        dict: Dictionary containing the parameters.
    """
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data["parameters"]
