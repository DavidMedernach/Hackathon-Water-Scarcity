"""Example of a policy that uses the incentive mechanism."""

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
    """Mixed policy.

    This policy combines fine, subvention, and no policy approaches based on the actions
    taken by the actors and the current state of the system.

    Args:
        actions (np.ndarray): Actions taken by the actors.
        actors_priority (np.ndarray): Priority levels of the actors.
        avg_incomes (np.ndarray): Average incomes of the actors.
        water_pump (np.ndarray): Water pumped by each actor.
        avg_pump (np.ndarray): Average water pumped.
        is_crisis (np.ndarray): Crisis status of the system.
        water_flows (np.ndarray): Water flows in the system.
        quota (np.ndarray): Quota assigned to each actor.
        DOE (int): Ecological optimal flow. Default is 15.
        DCR (int): Crisis flow threshold. Default is 10.
        
    Returns:
        np.ndarray: Adjusted actions based on the mixed policy.
    """
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
    """Fine policy.

    This policy applies a fine to the actors based on their cooperation.

    Args:
        actions (np.ndarray): Actions taken by the actors.
        actors_priority (np.ndarray): Priority levels of the actors.
        avg_incomes (np.ndarray): Average incomes of the actors.
        water_pump (np.ndarray): Water pumped by each actor.
        avg_pump (np.ndarray): Average water pumped.
        is_crisis (np.ndarray): Crisis status of the system.
        water_flows (np.ndarray): Water flows in the system.
        quota (np.ndarray): Quota assigned to each actor.
        DOE (int): Ecological optimal flow. Default is 15.
        DCR (int): Crisis flow threshold. Default is 10.

    Returns:
        np.ndarray: Adjusted actions based on the fine policy.
    """
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
    """Subvention policy.

    This policy applies a subvention to the actors based on their cooperation.
    Args:
        actions (np.ndarray): Actions taken by the actors.
        actors_priority (np.ndarray): Priority levels of the actors.
        avg_incomes (np.ndarray): Average incomes of the actors.
        water_pump (np.ndarray): Water pumped by each actor.
        avg_pump (np.ndarray): Average water pumped.
        is_crisis (np.ndarray): Crisis status of the system.
        water_flows (np.ndarray): Water flows in the system.
        quota (np.ndarray): Quota assigned to each actor.
        DOE (int): Ecological optimal flow. Default is 15.
        DCR (int): Crisis flow threshold. Default is 10.

    Returns:
        np.ndarray: Adjusted actions based on the subvention policy.
    """
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
    """No policy.

    This policy does not apply any incentives or penalties to the actors.
    Args:
        actions (np.ndarray): Actions taken by the actors.
        actors_priority (np.ndarray): Priority levels of the actors.
        avg_incomes (np.ndarray): Average incomes of the actors.
        water_pump (np.ndarray): Water pumped by each actor.
        avg_pump (np.ndarray): Average water pumped.
        is_crisis (np.ndarray): Crisis status of the system.
        water_flows (np.ndarray): Water flows in the system.
        quota (np.ndarray): Quota assigned to each actor.
        DOE (int): Ecological optimal flow. Default is 15.
        DCR (int): Crisis flow threshold. Default is 10.

    Returns:
        np.ndarray: No adjustment to actions.
    """
    return np.zeros(self.nb_actors)
