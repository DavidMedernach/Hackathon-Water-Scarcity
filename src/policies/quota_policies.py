"""Quota policies for the Water Management Simulation.

This module provides implementations of different quota policies
that can be used in the simulation to regulate water allocation.
"""

import numpy as np


def no_quota(
    self,
    crisis_level: int,
    actors_priority: np.ndarray,
    avg_pump: np.ndarray,
    DOE: float,
    DCR: float,
) -> np.ndarray:
    """No restrictions on quota policy.

    Allows all actors to pump twice their average amount regardless of crisis level.

    Args:
        crisis_level: Current water crisis level.
        actors_priority: Priority levels for each actor.
        avg_pump: Average pumping for each actor.
        DOE: Ecological optimal flow threshold.
        DCR: Crisis flow threshold.

    Returns:
        Array of water quotas for each actor.
    """
    quota = avg_pump.copy() * 2
    return quota


def hard_quota(
    self,
    crisis_level: int,
    actors_priority: np.ndarray,
    avg_pump: np.ndarray,
    DOE: float,
    DCR: float,
) -> np.ndarray:
    """Hard quota policy based on priority and crisis level.

    Sets quotas to zero for actors with priority less than the current crisis level.

    Args:
        crisis_level: Current water crisis level.
        actors_priority: Priority levels for each actor.
        avg_pump: Average pumping for each actor.
        DOE: Ecological optimal flow threshold.
        DCR: Crisis flow threshold.

    Returns:
        Array of water quotas for each actor.
    """
    quota = avg_pump.copy()
    p = actors_priority < crisis_level
    quota[p] = 0
    return quota
