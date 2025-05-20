"""
Ecology module for the Water Management Simulation.

This module handles all ecology-related functionality including:
- River flow calculations
- Crisis level determination
- Ecological impact calculations
- Final score calculations
"""

import numpy as np
from typing import List, Tuple


class EcologyManager:
    """
    Manages ecological aspects of the water management simulation.

    This class handles river flow calculations, ecological impact assessments,
    and final score calculations for the simulation.

    Attributes:
        sim: Reference to the parent simulation instance.
    """

    def __init__(self, simulation):
        """
        Initialize the ecology manager.

        Args:
            simulation: The parent simulation instance.
        """
        self.sim = simulation

    def calculate_riverflow(self, turn: int) -> float:
        """
        Calculate the river flow for a given turn.

        Models the river flow as a sinusoidal function with random variation
        or uses real data if available.

        Args:
            turn: The current turn number.

        Returns:
            The river flow value for the given turn.
        """
        if self.sim.station == -1:
            return (
                self.sim.mean_riverflow
                + 10 * np.sin(2 * np.pi * turn / 53)  # Seasonal variation
                + 5 * self.sim.rng_river.random()
                - 5
            )  # Random variation
        else:
            return self.sim.real_riverflows[turn]

    def compute_crisis(self) -> int:
        """
        Determine the crisis level based on the river flow.

        Calculates the current crisis level based on how the river flow compares to
        the DOE (ecological optimal flow) and DCR (crisis flow threshold).

        Returns:
            int: Crisis level:
                -1: Normal (river flow above DOE)
                0: Alert (river flow between DOE and intermediate threshold)
                1: Crisis level 1 (river flow between intermediate threshold and DCR)
                2: Crisis level 2 (river flow below DCR)
        """
        # Get the previous river flow or use initial value if at first turn
        if self.sim.trn - 1 >= 0:
            last_riverflow = (
                self.sim.w_riverflows[self.sim.it, self.sim.trn - 1]
                - self.sim.h_water_pump[:, self.sim.it, self.sim.trn - 1].sum()
            )
        else:
            last_riverflow = self.sim.mean_riverflow

        # Determine crisis level based on river flow thresholds
        eco_threshold = (self.sim.DCR + self.sim.DOE) / 2
        if last_riverflow < self.sim.DCR:  # Crisis level 2
            crisis = 2
        elif last_riverflow < eco_threshold:  # Crisis level 1
            crisis = 1
        elif last_riverflow < self.sim.DOE:  # Alert
            crisis = 0
        else:  # Normal
            crisis = -1

        # Store crisis level in history
        self.sim.w_crisis[self.sim.it, self.sim.trn] = crisis
        return crisis

    def compute_ecol_impacts(self) -> None:
        """
        Compute ecological impacts based on river flow and water usage.

        Calculates actual, minimum, and maximum ecological impacts and stores them
        in the respective history arrays.
        """
        riverflow = self.sim.w_riverflows[self.sim.it, self.sim.trn]
        total_water_used = np.sum(self.sim.h_water_pump[:, self.sim.it, self.sim.trn])

        # Actual ecological impact
        ecol_impact = -min(riverflow - self.sim.DCR - total_water_used, 0)
        self.sim.w_ecol_impact[self.sim.it, self.sim.trn] = ecol_impact

        # Minimum ecological impact (if no water was used)
        w_min_ecol_impact = -min(riverflow - self.sim.DCR, 0)
        self.sim.w_min_ecol_impact[self.sim.it, self.sim.trn] = w_min_ecol_impact

        # Maximum ecological impact (if all actors took their full demand)
        total_demand = np.sum(self.sim.actors_demands)
        max_ecol_impact = -min(riverflow - self.sim.DCR - total_demand, 0)
        self.sim.w_max_ecol_impact[self.sim.it, self.sim.trn] = max_ecol_impact

    def _compute_class_sats(
        self, high_idxs: np.ndarray, medium_idxs: np.ndarray, low_idxs: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute the mean satisfaction ratio for different priority classes.

        Calculates how well actors' water demands were met across the simulation,
        grouped by their priority levels.

        Args:
            high_idxs: Indices of high-priority actors.
            medium_idxs: Indices of medium-priority actors.
            low_idxs: Indices of low-priority actors.

        Returns:
            Tuple containing:
                - Average satisfaction ratio for high-priority actors
                - Average satisfaction ratio for medium-priority actors
                - Average satisfaction ratio for low-priority actors
        """
        # Expand demands to match dimensions of h_water_used
        demands_expanded = self.sim.actors_demands[:, None, None]

        # Calculate satisfaction ratio (water used / demand)
        sat = self.sim.h_water_used / demands_expanded

        # Average over actors, iterations, and turns
        high_sat = sat[high_idxs].mean() if len(high_idxs) > 0 else 0.0
        med_sat = sat[medium_idxs].mean() if len(medium_idxs) > 0 else 0.0
        low_sat = sat[low_idxs].mean() if len(low_idxs) > 0 else 0.0

        return high_sat, med_sat, low_sat

    def calculate_final_scores(self) -> List[float]:
        """
        Calculate final normalized scores for the simulation.

        Computes ecological breach and economic impact scores scaled to the best
        possible outcomes, and enforces a constraint that medium-priority actors
        should outperform low-priority ones by at least a stress-dependent factor.

        Returns:
            List containing [ecological_breach_score, economic_impact_score].
        """
        # 1) Ecological breach scaling
        total_ecol_breach = np.sum(self.sim.w_ecol_impact > 0)
        min_ecol_breach = np.sum(self.sim.w_min_ecol_impact > 0)
        max_ecol_breach = np.sum(self.sim.w_max_ecol_impact > 0)

        # Normalize ecological breach (0 = best, 1 = worst)
        breach_diff = max_ecol_breach - min_ecol_breach
        ecol_breach = (
            (total_ecol_breach - min_ecol_breach) / breach_diff
            if breach_diff > 0
            else 0.0
        )

        # 2) Economic impact scaling
        final_econ = np.mean(self.sim.h_econ_impacts)
        max_econ = np.mean(self.sim.h_max_econ_impacts)

        # Normalize economic impact (1 = best, 0 = worst)
        economic_impact = final_econ / max_econ if max_econ > 0 else 0.0

        # 3) Prepare fines/subventions arrays
        self.sim.h_fines = np.where(self.sim.h_policies > 0, self.sim.h_policies, 0)
        self.sim.h_subventions = -np.where(
            self.sim.h_policies < 0, self.sim.h_policies, 0
        )

        # 4) Identify priority classes
        high_idxs = np.where(self.sim.actors_priority == 2)[0]
        medium_idxs = np.where(self.sim.actors_priority == 1)[0]
        low_idxs = np.where(self.sim.actors_priority == 0)[0]

        # 5) Calculate satisfaction ratios by priority class
        high_sat, med_sat, low_sat = self._compute_class_sats(
            high_idxs, medium_idxs, low_idxs
        )

        # 6) Compute stress-scaled factor for constraint
        alpha = 0.01
        stress_ratio = np.mean(
            (self.sim.w_crisis == 0) | (self.sim.w_crisis == 1)
        )  # fraction of turns in alert/crisis
        factor = 1.0 + alpha * stress_ratio

        # 7) Enforce constraint: medium priority satisfaction must exceed low priority by factor
        if len(medium_idxs) > 0 and len(low_idxs) > 0 and med_sat <= low_sat * factor:
            if hasattr(self.sim, "verbose") and self.sim.verbose:
                print(
                    "Violation: medium-priority actors must outperform low-priority actors.",
                    factor,
                )
                print(
                    "High satisfaction:",
                    high_sat,
                    "Medium satisfaction:",
                    med_sat,
                    "Low satisfaction:",
                    low_sat,
                )
            # Return scores with penalties
            return [ecol_breach, economic_impact, 0.0]
        if len(high_idxs) > 0 and len(medium_idxs) > 0 and high_sat <= med_sat * factor:
            if hasattr(self.sim, "verbose") and self.sim.verbose:
                print(
                    "Violation: medium-priority actors must outperform low-priority actors.",
                    factor,
                )
                print(
                    "High satisfaction:",
                    high_sat,
                    "Medium satisfaction:",
                    med_sat,
                    "Low satisfaction:",
                    low_sat,
                )
            # Return scores with penalties

            return [ecol_breach, economic_impact, 0.0]
        else:
            # No violation, return normal scores
            return [ecol_breach, economic_impact, 1.0]
