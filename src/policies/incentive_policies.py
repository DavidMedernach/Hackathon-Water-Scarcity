import numpy as np


def mixed_policy(
    self,
    actions: np.ndarray,  # for testing purposes only
    actors_priority: np.ndarray,
    avg_incomes: np.ndarray,
    water_pump: np.ndarray,
    avg_pump: np.ndarray,
    is_crisis: np.ndarray,
    water_flows: np.ndarray,
    quota: np.ndarray,
    DOE=15,
    DCR=10,
) -> np.ndarray:
    actions = actions.astype(float) - 0.5
    return -actions * 5


def fine_policy(
    self,
    actions: np.ndarray,  # for testing purposes only
    actors_priority: np.ndarray,
    avg_incomes: np.ndarray,
    water_pump: np.ndarray,
    avg_pump: np.ndarray,
    is_crisis: np.ndarray,
    water_flows: np.ndarray,
    quota: np.ndarray,
    DOE=15,
    DCR=10,
) -> np.ndarray:
    actions = actions.astype(float) - 1.0
    return -actions * 5


def subvention_policy(
    self,
    actions: np.ndarray,  # for testing purposes only
    actors_priority: np.ndarray,
    avg_incomes: np.ndarray,
    water_pump: np.ndarray,
    avg_pump: np.ndarray,
    is_crisis: np.ndarray,
    water_flows: np.ndarray,
    quota: np.ndarray,
    DOE=15,
    DCR=10,
) -> np.ndarray:
    actions = actions.astype(float)
    return -actions * 5


def no_policy(
    self,
    actions: np.ndarray,  # for testing purposes only
    actors_priority: np.ndarray,
    avg_incomes: np.ndarray,
    water_pump: np.ndarray,
    avg_pump: np.ndarray,
    is_crisis: np.ndarray,
    water_flows: np.ndarray,
    quota: np.ndarray,
    DOE=15,
    DCR=10,
) -> np.ndarray:
    return np.zeros(self.nb_actors)
