"""
Core simulation module for the Water Management Simulation.

This module provides the main WaterManagementSimulation class that coordinates
the different components of the simulation: actors, water allocation, and ecological impacts.
"""

import numpy as np
import pandas as pd
from typing import List, Any
from types import MethodType

# These imports will be from actual modules once they're created
from .actors import ActorManager
from .water_allocation import WaterAllocator
from .ecology import EcologyManager


class WaterManagementSimulation:
    """
    Water Management Simulation that models the behavior of actors sharing water resources.

    This simulation allows for modeling different scenarios of water resource allocation
    with multiple actors who can cooperate or defect based on their learning from previous
    experiences. The simulation tracks economic and ecological impacts over time.
    """

    # Allowed parameters for simulation configuration
    ALLOWED_PARAMETERS = {
        "total_turns",
        "nb_iterations",
        "mean_riverflow",
        "reputational_cost",
        "cost_of_negotiation",
        "scenario_name",
        "scarcity",
        "DOE",
        "DCR",
        "station",
        "negotiation_difficulty",
        "storage_loss",
        "global_forecast_bias",
        "global_forecast_uncertainty",
        "actor_files",
        "actors_forecast_bias",
        "actors_forecast_uncertainty",
        "actors_storage_capa",
        "actors_initial_storage",
        "actors_type",
        "actors_ict",
        "actors_value",
        "actors_learning_rate",
        "actors_lr_factor",
        "actors_demands",
        "actors_ecol_impact",
        "actors_priority",
        "actors_values",
        "actors_reput_cost",
        "actors_baseline_income_factor",
        "actors_name",
        "verbose",
        "incentive_threshold",
    }

    def __init__(self, **args):
        """
        Initialize the water management simulation with the provided parameters.

        Args:
            **args: Arbitrary keyword arguments. Valid keys are defined in ALLOWED_PARAMETERS.
                   If a parameter is not provided, it will need to be set later.

        Raises:
            ValueError: If any parameter name is not in ALLOWED_PARAMETERS.
        """
        # Validate parameters
        invalid_params = set(args) - self.ALLOWED_PARAMETERS
        if invalid_params:
            raise ValueError(f"Unknown parameter(s): {', '.join(invalid_params)}")

        # Set simulation parameters
        for key, value in args.items():
            if isinstance(value, list):
                setattr(self, key, np.array(value))
            else:
                setattr(self, key, value)

        # Initialize managers
        self.get_real_riverflows()

        # These will be initialized after the modules are created
        self.actor_manager = ActorManager(self)
        self.water_allocator = WaterAllocator(self)
        self.ecology_manager = EcologyManager(self)

        # Current iteration and turn state
        self.it = 0
        self.trn = 0
        self.last_compute_avg_incomes = 0

        # Create method attributes for policies that can be overridden
        self.incentive_policy = MethodType(self._default_incentive_policy, self)
        self.compute_actor_quota = MethodType(self._default_compute_quota, self)

    def get_real_riverflows(self) -> None:
        """Get the real river flows for the given station."""
        if hasattr(self, "station") and self.station != -1:
            data = pd.read_csv("parameters/data.csv")
            riverflows = data[data["station_code"] == self.station]
            self.real_riverflows = riverflows.water_flow.values
            self.mean_riverflow = self.real_riverflows.mean()
            self.real_predictions = riverflows.predictions.values
        else:
            self.real_riverflows = None

    def set_parameter(self, value: Any, name: str) -> None:
        """
        Set a specific parameter of the simulation.

        Args:
            value: The value to set for the parameter.
            name: The name of the parameter to set.

        Raises:
            ValueError: If the parameter name is not recognized.
        """
        if name not in self.ALLOWED_PARAMETERS:
            raise ValueError(f"Unknown parameter: {name}")
        setattr(self, name, value)

    def get_data(self, data_type: str) -> np.ndarray:
        """
        Get history data for a specific attribute.

        Args:
            data_type: The name of the attribute to retrieve.

        Returns:
            The requested attribute data.

        Raises:
            ValueError: If the data type is not recognized.
        """
        if not hasattr(self, data_type):
            raise ValueError(f"Unknown data type: {data_type}")
        return getattr(self, data_type)

    def run_simulation(self, seed: int = None) -> None:
        """
        Run the complete water management simulation.

        Executes the simulation for the specified number of iterations and turns,
        processing each turn in sequence.
        """
        for iteration in range(self.nb_iterations):
            self.it = iteration
            self.last_compute_avg_incomes = 0
            # Reset random number generator for reproducibility
            if seed is not None:
                self.rng_river = np.random.default_rng(seed=seed)
                self.rng_actors = np.random.default_rng(seed=seed)
            else:
                self.rng_river = np.random.default_rng()
                self.rng_actors = np.random.default_rng()

            for turn in range(self.total_turns):
                self.trn = turn
                self._process_turn()

    def _process_turn(self) -> None:
        """
        Process a single turn of the simulation.

        Executes all calculations for a single time step, including actor actions,
        water allocation, rewards, and environmental impacts.
        """
        # Initialize tendencies for first turn
        if self.trn == 0:
            self.h_tendencies[:, self.it, self.trn] = self.actors_ict

        # 1. Actor actions and river flow
        actor_tendency = self.h_tendencies[:, self.it, self.trn]
        actions = self.actor_manager.get_actor_actions(actor_tendency)
        self.h_actions[:, self.it, self.trn] = actions
        self.w_riverflows[self.it, self.trn] = self.ecology_manager.calculate_riverflow(
            self.trn
        )

        # 2. Water predictions and crisis determination
        riverflow_predictions = self.actor_manager.get_water_predictions()
        self.water_allocator.compute_avg_pump()

        self.ecology_manager.compute_crisis()
        self.h_quota[:, self.it, self.trn] = self.compute_actor_quota(
            crisis_level=int(self.w_crisis[self.it, self.trn]),
            actors_priority=self.actors_priority,
            avg_pump=self.h_avg_pump[:, self.it, self.trn],
            DOE=self.DOE,
            DCR=self.DCR,
        )

        # 3. Water allocation and usage calculation
        water_pump = self.water_allocator.compute_pump(
            actions=actions,
            riverflow_predictions=riverflow_predictions,
            storage=self.actor_manager.get_previous_storage(),
        )
        self.h_water_pump[:, self.it, self.trn] = water_pump

        # Calculate actual water used by each actor
        max_ava_water = water_pump + self.actor_manager.get_previous_storage()
        w_used_by_actor = np.minimum(max_ava_water, self.actors_demands)
        self.h_water_used[:, self.it, self.trn] = w_used_by_actor

        # 4. Update storage, average incomes, and calculate impacts
        self.actor_manager.update_storage()
        self.water_allocator.compute_avg_incomes()

        # 5. Calculate rewards and incentives
        rewards, incentives = self.water_allocator.compute_cost(
            actions=actions,
            riverflow=self.w_riverflows[self.it, :],
            water_pump=water_pump,
            water_used_by_actor=w_used_by_actor,
        )

        self.h_rewards[:, self.it, self.trn] = rewards
        self.h_policies[:, self.it, self.trn] = incentives

        # 6. Calculate economic and ecological impacts
        self.water_allocator.compute_h_econ_impacts()
        self.ecology_manager.compute_ecol_impacts()

        # 7. Calculate alternative rewards for learning
        diff_rewards = self.actor_manager.compute_alternative_rewards(
            riverflow_predictions
        )
        self.h_diff_rewards[:, self.it, self.trn] = diff_rewards
        self.h_alt_storage[:, self.it, self.trn] = (
            self.actor_manager.get_previous_storage()
        )

        # 8. Update tendencies for next turn
        if self.trn < self.total_turns - 1:
            self.actor_manager.update_actor_h_tendencies(diff_rewards)

    def get_final_scores_scaled(self) -> List[float]:
        """
        Calculate final normalized scores for the simulation.

        Returns:
            List containing [ecological_breach_score, economic_impact_score].
        """
        return self.ecology_manager.calculate_final_scores()

    def _default_compute_quota(
        self,
        crisis_level: int,
        actors_priority: np.ndarray,
        avg_pump: np.ndarray,
        DOE: float,
        DCR: float,
    ) -> np.ndarray:
        """
        Quota policy placeholder.

        This method can be overridden with a custom implementation.

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
        return quota

    def _default_incentive_policy(
        self,
        actions: np.ndarray,
        actors_priority: np.ndarray,
        avg_incomes: np.ndarray,
        water_pump: np.ndarray,
        avg_pump: np.ndarray,
        is_crisis: np.ndarray,
        water_flows: np.ndarray,
        quota: np.ndarray,
        DOE: float = 15,
        DCR: float = 10,
    ) -> np.ndarray:
        """
        Incentive policy placeholder.

        Args:
            actions: Boolean array indicating cooperation status.
            actors_priority: Priority levels for each actor.
            avg_incomes: Average incomes for each actor.
            water_pump: Historical water pumping data.
            avg_pump: Average pumping for each actor.
            is_crisis: Historical crisis levels.
            water_flows: Historical river flows.
            quota: Current water quotas for each actor.
            DOE: Ecological optimal flow threshold.
            DCR: Crisis flow threshold.

        Returns:
            Array of incentives for each actor.
        """
        return np.zeros(self.nb_actors)
