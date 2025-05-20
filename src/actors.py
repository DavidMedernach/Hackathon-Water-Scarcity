"""
Actors module for the Water Management Simulation.

This module handles all actor-related functionality including:
- Actor initialization and parameter management
- Actor decision-making (cooperation vs defection)
- Actor learning and adaptation
- Actor storage management
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import src.utils as utils


class ActorManager:
    """
    Manages the actors in the water management simulation.

    This class handles actor initialization, decision-making, learning,
    and storage management for all actors in the simulation.

    Attributes:
        sim: Reference to the parent simulation instance.
    """

    def __init__(self, simulation):
        """
        Initialize the actor manager.

        Args:
            simulation: The parent simulation instance.
        """
        self.sim = simulation
        self._initialize_actors_parameters()

    def _initialize_actors_parameters(self) -> None:
        """
        Initialize actor parameters from configuration files.

        Loads actor parameters from YAML files and processes them to prepare for simulation.
        This includes setting up actor-specific attributes and scaling demands and learning rates.
        """
        for actor_file in self.sim.actor_files:
            params = utils.load_parameters_from_yaml(f"parameters/{actor_file}")
            # Add actors_ prefix to keys in params
            renamed_params = {f"actors_{key}": value for key, value in params.items()}

            invalid_params = set(renamed_params) - self.sim.ALLOWED_PARAMETERS
            if invalid_params:
                raise ValueError(f"Unknown parameter(s): {', '.join(invalid_params)}")

            for key, value in renamed_params.items():
                # If attribute doesn't exist, create it
                if not hasattr(self.sim, key):
                    setattr(self.sim, key, [value])
                # If attribute exists, append the value
                else:
                    current = getattr(self.sim, key)
                    current.append(value)
                    setattr(self.sim, key, current)

        self.sim.nb_actors = len(self.sim.actor_files)

        # Convert all actor parameters to numpy arrays
        for key in self.sim.ALLOWED_PARAMETERS:
            if key.startswith("actors_") and hasattr(self.sim, key):
                value_list = getattr(self.sim, key)
                setattr(self.sim, key, np.array(value_list))

        self._scale_actors_demands()
        self._scale_incentive_thresholds()
        self._scale_learning_rate()
        self._initialize_history_containers()

    def _scale_actors_demands(self) -> None:
        """
        Scale actor demands based on scarcity and negotiation difficulty.

        Adjusts the water demands of actors to ensure they are appropriate for the
        simulation scenario, based on water scarcity and difficulty of negotiation.
        This also scales storage capacity and initial storage relative to demands.
        """
        total_demand = np.sum(self.sim.actors_demands)
        base_flow = self.sim.mean_riverflow
        base_available = base_flow - self.sim.DCR

        # Set scaling factor based on scarcity level
        scarcity_factors = {"extreme": 0.5, "high": 0.4, "medium": 0.3, "low": 0.2}

        if self.sim.scarcity not in scarcity_factors:
            raise ValueError(f"Invalid scarcity level: {self.sim.scarcity}")

        factor = scarcity_factors[self.sim.scarcity]
        scaling = factor * base_available / total_demand

        # Scale demands and related parameters
        self.sim.actors_demands = self.sim.actors_demands * scaling
        self.sim.actors_storage_capa = (
            self.sim.actors_storage_capa * self.sim.actors_demands
        )
        self.sim.actors_initial_storage = (
            self.sim.actors_initial_storage * self.sim.actors_demands
        )

        # Set negotiation cost based on difficulty
        scaling = base_available / 30.0
        negotiation_cost_factors = {
            "extreme": 1.0,
            "high": 0.5,
            "medium": 0.25,
            "low": 0.1,
            "none": 0.0,
        }

        if self.sim.negotiation_difficulty not in negotiation_cost_factors:
            raise ValueError(
                f"Invalid negotiation difficulty: {self.sim.negotiation_difficulty}"
            )

        factor = negotiation_cost_factors[self.sim.negotiation_difficulty]
        self.sim.cost_of_negotiation = scaling * factor

    def _scale_incentive_thresholds(self) -> None:
        """
        Scale incentive thresholds based on demands and water values.

        """
        max_benefit = np.sum(self.sim.actors_demands * self.sim.actors_values)
        max_total_benefit = max_benefit * self.sim.total_turns

        self.sim.incentive_threshold = max_total_benefit * 10

    def _scale_learning_rate(self) -> None:
        """
        Scale the learning rate for each actor.

        Adjusts learning rates based on maximum possible benefits to ensure
        appropriate learning speed.
        """
        max_benef = self.sim.actors_values * self.sim.actors_demands
        self.sim.actors_learning_rate = self.sim.actors_lr_factor / max_benef

    def _initialize_history_containers(self) -> None:
        """
        Initialize all containers as NumPy arrays with pre-allocated space.

        Sets up the arrays that will store simulation history, including actor
        actions, rewards, water usage, and ecological impacts over all iterations.
        """
        shapes_in_3d = (
            self.sim.nb_actors,
            self.sim.nb_iterations,
            self.sim.total_turns,
        )
        shapes_in_2d = (self.sim.nb_iterations, self.sim.total_turns)

        # Parameters that need actor dimension
        with_actors = [
            "h_econ_impacts",
            "h_tendencies",
            "h_actions",
            "h_rewards",
            "h_diff_rewards",
            "h_alt_storage",
            "h_water_pump",
            "h_storage",
            "h_policies",
            "h_water_used",
            "h_max_econ_impacts",
            "h_min_econ_impacts",
            "h_fines",
            "h_subventions",
            "h_taxed_incomes",
            "h_avg_pump",
            "h_quota",
        ]

        # Parameters without actor dimension
        without_actors = [
            "w_ecol_impact",
            "w_riverflows",
            "w_min_ecol_impact",
            "w_max_ecol_impact",
            "w_crisis",
        ]

        # Initialize arrays
        for attr in with_actors:
            setattr(self.sim, attr, np.zeros(shapes_in_3d))
        for attr in without_actors:
            setattr(self.sim, attr, np.zeros(shapes_in_2d))

        # Set initial values
        initial_taxed_incomes = (
            self.sim.actors_demands
            * self.sim.actors_values
            * self.sim.actors_baseline_income_factor
        )
        self.sim.h_taxed_incomes[:, :, 0] = initial_taxed_incomes[:, np.newaxis]
        self.sim.h_avg_pump[:, :, 0] = self.sim.actors_demands[:, np.newaxis]

    def get_actor_actions(self, actor_tendency: np.ndarray) -> np.ndarray:
        """
        Determine actor actions based on their tendency to cooperate.

        Generates a boolean array where True indicates cooperation and False indicates defection.

        Args:
            actor_tendency: Array of cooperation tendencies for each actor.

        Returns:
            Boolean array of actor actions (True = cooperate, False = defect).
        """
        return self.sim.rng_actors.random(self.sim.nb_actors) < actor_tendency

    def get_water_predictions(self) -> np.ndarray:
        """
        Generate water availability predictions for each actor.

        Calculates predicted river flows for each actor based on actual flow,
        forecast bias, and uncertainty.

        Returns:
            Array of predicted river flows for each actor.
        """
        # Get actual river flow
        if self.sim.station == -1:
            riverflow = self.sim.w_riverflows[self.sim.it, self.sim.trn]
        else:
            riverflow = self.sim.real_predictions[self.sim.trn]

        # Min uncertainty to make sure actors do not have the same prediction
        min_uncertainty = 0.1

        # Calculate prediction with bias and uncertainty
        variances = (
            self.sim.actors_forecast_uncertainty
            + self.sim.global_forecast_uncertainty
            + min_uncertainty
        )
        water_pred = (
            self.sim.actors_forecast_bias * riverflow
            + self.sim.global_forecast_bias * riverflow
            + riverflow
            - riverflow * variances
            + riverflow * 2 * variances * self.sim.rng_actors.random(self.sim.nb_actors)
        )

        # Ensure predictions are non-negative
        return np.maximum(0, water_pred)

    def update_storage(self) -> None:
        """
        Update the water storage for each actor.

        Calculates new storage levels based on previous storage, water pumped,
        demand usage, and storage losses.
        """
        water_pump = self.sim.h_water_pump[:, self.sim.it, self.sim.trn]
        storage = self.get_previous_storage()

        # Update storage: add pumped water, subtract demand, apply constraints
        storage += water_pump - self.sim.actors_demands
        self.sim.h_storage[:, self.sim.it, self.sim.trn] = np.clip(
            storage, 0, self.sim.actors_storage_capa
        )

    def get_previous_storage(self, turn: int = 1) -> np.ndarray:
        """
        Get the water storage levels from a previous turn.

        Args:
            turn: Number of turns to look back (default: 1).

        Returns:
            Array of storage levels for each actor from the specified previous turn.
        """
        if self.sim.trn - turn < 0:
            return self.sim.actors_initial_storage.copy()
        else:
            return self.sim.h_storage[:, self.sim.it, self.sim.trn - turn].copy()

    def get_previous_actions(self, turn: int = 1) -> np.ndarray:
        """
        Get the actions from a previous turn.

        Args:
            turn: Number of turns to look back (default: 1).

        Returns:
            Array of actor actions from the specified previous turn.
        """
        if self.sim.trn - turn < 0:
            return self.sim.h_actions[:, self.sim.it, 0].copy()
        else:
            return self.sim.h_actions[:, self.sim.it, self.sim.trn - turn].copy()

    def get_previous_alt_storage(self, turn: int = 1) -> np.ndarray:
        """
        Get the alternative water storage levels from a previous turn.

        Used for calculating counterfactual scenarios in learning.

        Args:
            turn: Number of turns to look back (default: 1).

        Returns:
            Array of alternative storage levels for each actor from the specified previous turn.
        """
        if self.sim.trn - turn < 0:
            return self.sim.actors_initial_storage.copy()
        else:
            return self.sim.h_alt_storage[:, self.sim.it, self.sim.trn - turn].copy()

    def compute_alternative_rewards(
        self, riverflow_predictions: np.ndarray
    ) -> np.ndarray:
        """
        Compute alternative rewards to determine the value of cooperation/defection.

        For each actor, calculates the difference in rewards between their
        actual decision and the opposite decision, considering potential
        future impacts on storage.

        Args:
            riverflow_predictions: Array of predicted river flows for each actor.

        Returns:
            Array of alternative reward differences for each actor.
        """
        actors_actions = self.sim.h_actions[:, self.sim.it, self.sim.trn]
        alt_rewards = np.zeros(self.sim.nb_actors)

        for i in range(self.sim.nb_actors):
            # Consider storage history depth based on capacity
            h_depth = int(self.sim.actors_storage_capa[i] / self.sim.actors_demands[i])
            diff_r = np.zeros(h_depth + 1)
            divisor = 0
            for j in range(h_depth + 1):
                # Get alternative storage scenario
                prev_storage = self.get_previous_storage()
                old_storage = self.get_previous_alt_storage(1 + j)[i]

                prev_storage[i] = (
                    prev_storage[i] + old_storage - self.get_previous_storage(1 + j)[i]
                )
                prev_storage[i] = np.clip(
                    prev_storage[i], 0, self.sim.actors_storage_capa[i]
                )

                # Create alternative action scenario
                alternative_actions = actors_actions.copy()
                if j == 0:
                    # For immediate impact, flip the action
                    alternative_actions[i] = not alternative_actions[i]
                actor_action = not self.get_previous_actions(1 + j)[i]
                actor_action = np.where(actor_action, -1, 1)

                # Calculate pumping under alternative scenario
                water_pump = self.sim.water_allocator.compute_pump(
                    actions=alternative_actions,
                    riverflow_predictions=riverflow_predictions,
                    storage=prev_storage,
                )

                # Calculate water usage and storage changes
                max_ava_water = water_pump + prev_storage
                w_used_by_actor = np.minimum(max_ava_water, self.sim.actors_demands)
                alt_storage = water_pump - self.sim.actors_demands + prev_storage.copy()
                alt_storage = np.clip(alt_storage, 0, self.sim.actors_storage_capa)

                # Store alternative storage for current turn
                if j == 0:
                    self.sim.h_alt_storage[i, self.sim.it, self.sim.trn] = alt_storage[
                        i
                    ]
                # Calculate alternative reward

                r, _ = self.sim.water_allocator.compute_cost(
                    actions=alternative_actions,
                    riverflow=self.sim.w_riverflows[self.sim.it, :],
                    water_pump=water_pump,
                    water_used_by_actor=w_used_by_actor,
                )

                # Calculate reward difference and weight by time depth
                diff_r[j] = r[i] - self.sim.h_rewards[i, self.sim.it, self.sim.trn]
                diff_r[j] = (
                    diff_r[j]
                    / (j + 1.0)
                    * np.where(
                        self.sim.h_actions[i, self.sim.it, self.sim.trn],
                        1,  # Positive difference for cooperators
                        -1,  # Negative difference for defectors
                    )
                )
                divisor += 1 / (j + 1.0)
                diff_r[j] = (
                    r[i] - self.sim.h_rewards[i, self.sim.it, self.sim.trn]
                ) * actor_action
                diff_r[j] = diff_r[j] / (j + 1.0)

            # Calculate weighted average of reward differences
            alt_rewards[i] = sum(diff_r) / divisor

        return alt_rewards

    def update_actor_h_tendencies(self, diff_rewards: np.ndarray) -> None:
        """
        Update actor tendencies to cooperate based on reward differences.

        Adjusts each actor's cooperation tendency based on their learning rate
        and the difference between actual and alternative rewards.

        Args:
            diff_rewards: Array of reward differences for each actor.
        """
        tendency = self.sim.h_tendencies[:, self.sim.it, self.sim.trn]
        new_h_tendencies = tendency - self.sim.actors_learning_rate * diff_rewards
        self.sim.h_tendencies[:, self.sim.it, self.sim.trn + 1] = np.clip(
            new_h_tendencies, 0, 1
        )
