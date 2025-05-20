"""
Water allocation module for the Water Management Simulation.

This module handles all water allocation related functionality including:
- Water pumping calculations
- Quota determination and enforcement
- Cost and reward calculations
- Economic impact calculations
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union


class WaterAllocator:
    """
    Manages water allocation in the simulation.

    This class handles the calculation of water pumping, economic rewards,
    and incentives based on actors' decisions and water availability.

    Attributes:
        sim: Reference to the parent simulation instance.
    """

    def __init__(self, simulation):
        """
        Initialize the water allocator.

        Args:
            simulation: The parent simulation instance.
        """
        self.sim = simulation

    def compute_pump(
        self,
        actions: np.ndarray,
        riverflow_predictions: np.ndarray,
        storage: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the amount of water each actor will pump from the river.

        Determines how much water each actor will take based on their cooperation status,
        current crisis level, storage capacity, and water availability predictions.

        Args:
            actions: Boolean array indicating which actors are cooperating (True) or defecting (False).
            riverflow_predictions: Array of predicted river flows for each actor.
            storage: Current water storage for each actor.

        Returns:
            Array of water volumes that each actor will pump from the river.
        """
        # Get cooperating actors and calculate basic needs
        coop = np.where(actions)[0]
        capa = self._calculate_capacity(storage)
        minimal_needs = self._calculate_minimal_needs(storage)

        # Determine allocation based on current crisis level
        is_normal = self.sim.w_crisis[self.sim.it, self.sim.trn] == -1

        if is_normal:
            # Normal situation (no crisis)
            conso = self._allocate_normal(
                coop, capa, minimal_needs, riverflow_predictions, actions
            )
        else:
            # Alert/Crisis situation
            conso = self._allocate_crisis(coop, capa, minimal_needs)

        # Scale down if total consumption exceeds actual river flow
        conso = self._apply_flow_constraint(conso)

        return conso

    def _calculate_capacity(self, storage: np.ndarray) -> np.ndarray:
        """Calculate maximum capacity each actor can pump based on storage."""
        return self.sim.actors_storage_capa - storage + self.sim.actors_demands

    def _calculate_minimal_needs(self, storage: np.ndarray) -> np.ndarray:
        """Calculate minimum water needed to meet demands given current storage."""
        return np.maximum(self.sim.actors_demands - storage, 0)

    def _allocate_normal(
        self,
        coop: np.ndarray,
        capa: np.ndarray,
        minimal_needs: np.ndarray,
        riverflow_predictions: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Allocate water during normal conditions."""
        # Initialize with maximum capacity
        conso = capa.copy()

        # If no one cooperates, everyone pumps at capacity
        if len(coop) == 0:
            return conso

        # Calculate available water based on cooperators' predictions
        ava_water = self._estimate_available_water(coop, riverflow_predictions)
        total_capacity = self._extrapolate_total_capacity(coop)

        # If plenty of water available, everyone takes their full capacity
        if ava_water >= total_capacity:
            return conso

        # Otherwise, allocate based on available water and priorities
        total_demand = self._extrapolate_total_demand(coop, self.sim.actors_demands)
        total_needs = self._extrapolate_total_demand(coop, minimal_needs)

        if total_demand <= ava_water:
            # Everyone can get their demands
            conso[coop] = self.sim.actors_demands[coop]
        elif total_needs <= ava_water:
            # Everyone can get their minimal needs
            conso[coop] = minimal_needs[coop]
        else:
            # Prioritized allocation based on actor priority
            conso = self._allocate_by_priority(conso, actions, minimal_needs, ava_water)

        return conso

    def _estimate_available_water(
        self, coop: np.ndarray, riverflow_predictions: np.ndarray
    ) -> float:
        """Estimate available water based on cooperators' predictions."""
        return np.mean(riverflow_predictions[coop]) - self.sim.DOE

    def _extrapolate_total_capacity(self, coop: np.ndarray) -> float:
        """Extrapolate total capacity from cooperating actors."""
        coop_capacity = np.sum(self.sim.actors_storage_capa[coop])
        return coop_capacity * self.sim.nb_actors / len(coop)

    def _extrapolate_total_demand(self, coop: np.ndarray, demands: np.ndarray) -> float:
        """Extrapolate total demand from cooperating actors."""
        coop_demand = np.sum(demands[coop])
        return coop_demand * self.sim.nb_actors / len(coop)

    def _allocate_by_priority(
        self,
        conso: np.ndarray,
        actions: np.ndarray,
        minimal_needs: np.ndarray,
        ava_water: float,
    ) -> np.ndarray:
        """Allocate water based on actor priorities."""
        # Create masks for different priority levels
        P2_actors = (self.sim.actors_priority == 2) & (actions == 1)  # High priority
        P1_actors = (self.sim.actors_priority == 1) & (actions == 1)  # Medium priority
        P0_actors = (self.sim.actors_priority == 0) & (actions == 1)  # Low priority

        # Highest priority actors get their minimal needs
        conso[P2_actors] = minimal_needs[P2_actors]
        remaining_water = np.maximum(ava_water - np.sum(conso[P2_actors]), 0)

        # Medium priority actors share remaining water
        nb_p1 = np.sum(P1_actors)
        if nb_p1 > 0:
            conso[P1_actors] = np.minimum(
                remaining_water / nb_p1, minimal_needs[P1_actors]
            )
            remaining_water = np.maximum(remaining_water - np.sum(conso[P1_actors]), 0)

        # Low priority actors share any water left
        nb_p0 = np.sum(P0_actors)
        if nb_p0 > 0:
            conso[P0_actors] = remaining_water / nb_p0

        return conso

    def _allocate_crisis(
        self, coop: np.ndarray, capa: np.ndarray, minimal_needs: np.ndarray
    ) -> np.ndarray:
        """Allocate water during crisis conditions."""
        quota = self.sim.h_quota[:, self.sim.it, self.sim.trn]
        conso = capa.copy()

        # Defecting actors take as much as possible considering quota and capacity
        defectors = ~np.isin(np.arange(self.sim.nb_actors), coop)
        conso[defectors] = np.minimum(conso[defectors], quota[defectors])
        # Ensure they meet minimum needs
        conso[defectors] = np.maximum(conso[defectors], minimal_needs[defectors])

        # Cooperating actors take only their minimal needs up to quota
        conso[coop] = np.minimum(minimal_needs[coop], quota[coop])

        return conso

    def _apply_flow_constraint(self, conso: np.ndarray) -> np.ndarray:
        """Scale down consumption if it exceeds available river flow."""
        total_consumption = np.sum(conso)
        actual_flow = self.sim.w_riverflows[self.sim.it, self.sim.trn]

        if total_consumption > actual_flow:
            scaling_factor = actual_flow / total_consumption
            return conso * scaling_factor

        return conso

    def compute_cost(
        self,
        actions: np.ndarray,
        riverflow: np.ndarray,
        water_pump: np.ndarray,
        water_used_by_actor: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute rewards and incentives for each actor.

        Calculates the economic rewards for each actor based on water usage,
        cooperation status, and environmental impacts.

        Args:
            actions: Boolean array indicating which actors are cooperating (True) or defecting (False).
            riverflow: Array of river flows for each turn.
            water_pump: Array of water consumption (pumping) for each actor.
            water_used_by_actor: Array of water actually used by each actor.

        Returns:
            Tuple containing:
                - Array of rewards for each actor
                - Array of incentives (fines or subsidies) for each actor
        """
        # Ensure actions is a boolean array
        actions = np.array(actions, dtype=bool)

        # Calculate base rewards from water usage
        reward = water_used_by_actor * self.sim.actors_values
        # Apply reputation penalty for defectors
        water = water_used_by_actor[~actions]
        values = self.sim.actors_values[~actions]
        reputation_factor = 1.0 - self.sim.actors_reput_cost[~actions]
        reward[~actions] = water * values * reputation_factor

        # Calculate ecological impact
        total_water_pump = np.sum(water_pump)
        cur_riverflow = riverflow[self.sim.trn]
        remaining_water = cur_riverflow - total_water_pump - self.sim.DCR

        # Apply ecological impact penalty if water usage exceeds ecological threshold
        if remaining_water < 0:
            reward += self.sim.actors_ecol_impact

        # Calculate incentives (fines or subsidies)
        incentives = self.sim.incentive_policy(
            actions=actions,
            actors_priority=self.sim.actors_priority,
            avg_incomes=self.sim.h_taxed_incomes[:, self.sim.it, self.sim.trn],
            water_pump=water_pump,
            avg_pump=self.sim.h_avg_pump[:, self.sim.it, self.sim.trn],
            is_crisis=self.sim.w_crisis[self.sim.it, 0 : self.sim.trn + 1],
            water_flows=riverflow[: self.sim.trn + 1],
            quota=self.sim.h_quota[:, self.sim.it, self.sim.trn],
            DOE=self.sim.DOE,
            DCR=self.sim.DCR,
        )

        # make sure incentives are between threshold values
        incentives = np.clip(
            incentives, -self.sim.incentive_threshold, self.sim.incentive_threshold
        )

        reward = (
            reward - np.where(actions, self.sim.cost_of_negotiation, 0) - incentives
        )

        return reward, incentives

    def compute_avg_incomes(self) -> None:
        """
        Compute and update the average income for each actor.

        Calculates a rolling average of actor incomes over the past 52 turns (one year),
        updating only on complete years.
        """
        if self.sim.last_compute_avg_incomes == 52:
            # Calculate annual average of income over last 52 turns
            avg_incomes = np.mean(
                self.sim.h_rewards[:, self.sim.it, self.sim.trn - 52 : self.sim.trn],
                axis=1,
            )
            self.sim.h_taxed_incomes[:, self.sim.it, self.sim.trn] = avg_incomes
            self.sim.last_compute_avg_incomes = 0
        elif self.sim.trn > 0:
            # Keep previous values during the year
            self.sim.h_taxed_incomes[:, self.sim.it, self.sim.trn] = (
                self.sim.h_taxed_incomes[:, self.sim.it, self.sim.trn - 1]
            )
            self.sim.last_compute_avg_incomes += 1
        else:
            # First turn initialization
            self.sim.last_compute_avg_incomes += 1

    def compute_avg_pump(self) -> None:
        """
        Compute and update the average water pumping for each actor.

        Calculates a rolling average of water pumping over the past 52 turns (one year),
        updating only on complete years.
        """
        if self.sim.last_compute_avg_incomes == 52:
            # Calculate annual average of water pumping over last 52 turns
            avg_pump = np.mean(
                self.sim.h_water_pump[:, self.sim.it, self.sim.trn - 52 : self.sim.trn],
                axis=1,
            )
            self.sim.h_avg_pump[:, self.sim.it, self.sim.trn] = avg_pump
        elif self.sim.trn > 0:
            # Keep previous values during the year
            self.sim.h_avg_pump[:, self.sim.it, self.sim.trn] = self.sim.h_avg_pump[
                :, self.sim.it, self.sim.trn - 1
            ]

    def compute_h_econ_impacts(self) -> None:
        """
        Compute economic impacts under different allocation scenarios.

        Calculates and stores:
        - Maximum possible economic impact (optimal allocation by value)
        - Minimum sustainable economic impact (allocation by priority up to DCR)
        - Actual economic impact achieved
        """
        # Extract parameters
        riverflow = self.sim.w_riverflows[self.sim.it, self.sim.trn].copy()
        demands = self.sim.actors_demands
        values = self.sim.actors_values
        priority = self.sim.actors_priority

        def greedy_allocation(
            available: float, demands: np.ndarray, order: np.ndarray
        ) -> np.ndarray:
            """
            Perform greedy allocation of water resources.

            Args:
                available: Amount of water available for allocation.
                demands: Array of water demands for each actor.
                order: Array of actor indices in allocation order.

            Returns:
                Array of water allocations for each actor.
            """
            allocation = np.zeros_like(demands, dtype=float)
            for i in order:
                alloc = min(demands[i], available)
                allocation[i] = alloc
                available -= alloc
                if available <= 0:
                    break
            return allocation

        # Allocate by economic value (maximum economic benefit)
        order_max = np.argsort(-values)  # descending order (highest value first)
        alloc_max = greedy_allocation(riverflow, demands, order_max)
        self.sim.h_max_econ_impacts[:, self.sim.it, self.sim.trn] = alloc_max * values

        # Allocate by priority for sustainability (DCR-constrained)
        available_sustain = max(riverflow - self.sim.DCR, 0)
        order_s = np.argsort(-priority)  # descending order (highest priority first)
        alloc_sustain = greedy_allocation(available_sustain, demands, order_s)
        self.sim.h_min_econ_impacts[:, self.sim.it, self.sim.trn] = (
            alloc_sustain * values
        )

        # Calculate actual economic impacts achieved
        incentives = self.sim.h_policies[:, self.sim.it, self.sim.trn]
        water_benefits = self.sim.h_water_used[:, self.sim.it, self.sim.trn] * values
        # Subtract fines, 1% administration cost for subsidies
        water_benefits -= np.where(incentives > 0, incentives, -incentives / 100.0)
        self.sim.h_econ_impacts[:, self.sim.it, self.sim.trn] = water_benefits
