"""Saving utilities."""
import os
import matplotlib.pyplot as plt


def save_or_create(plt: plt.Figure, save_path: str):
    """Save a plot to a file, creating the directory if it does not exist.

    Args:
        plt (plt.Figure): The matplotlib figure to save.
        save_path (str): The path where the plot will be saved.

    Returns:
        None
    """
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
