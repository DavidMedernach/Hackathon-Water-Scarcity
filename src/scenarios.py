import src.utils as utils
import src.core as wms
import pandas as pd
import numpy as np
from types import MethodType
from typing import Callable


def run_all_scenarios(
    turns: int,
    iterations: int,
    custom_incentive_policy: Callable,
    custom_quota: Callable,
):
    """
    Run simulations across all defined scenarios and return a DataFrame with results.
    """
    # Scenario parameters
    scenarios = [
        "0.yml",
        "1.yml",
        "0-v.yml",
        "1-v.yml",
        "0-b.yml",
        "1-b.yml",
        "0-c.yml",
        "1-c.yml",
    ]
    scarcity_levels = ["low", "medium", "high"]
    exploration_biases = [0.0, 0.25, -0.25, 0.5, -0.5]
    exploration_uncertainties = [0.0, 0.25, 0.5]

    # Data collection lists
    all_ecological_impact = []
    all_economic_impact = []
    all_biases = []
    all_uncertainties = []
    all_scarcities = []
    all_scenarios = []
    all_stations = []
    all_priority_ok = []  # Priority satisfaction OK

    # New data collection lists
    all_raw_ecological_impact = []  # Raw unscaled ecological impact
    all_raw_economic_impact = []  # Raw unscaled ecological impact
    all_cooperation_percentage = []  # Average percentage of cooperators

    # Station definitions
    stations = {
        1: {"station": 6125320, "DOE": 0.1, "DCR": 0.05},  # Le Tarn
        2: {"station": 6124501, "DOE": 7, "DCR": 3.5},  # La Vézère
    }

    print("Starting simulations across all scenarios...")

    # Run through all scenario combinations
    for station in stations:
        for scarcity in scarcity_levels:
            for scenario in scenarios:
                # Determine whether to vary bias and uncertainty based on scenario name
                if len(scenario) == 5:  # Base scenarios
                    biases = exploration_biases
                    uncertainties = exploration_uncertainties
                else:  # Variant scenarios
                    biases = [0.0]
                    uncertainties = [0.0]

                for bias in biases:
                    for uncertainty in uncertainties:
                        # Configure simulation
                        yaml_path = f"parameters/scenarios/{scenario}"
                        params = utils.load_parameters_from_yaml(yaml_path)
                        params["total_turns"] = turns
                        params["nb_iterations"] = iterations
                        params["scarcity"] = scarcity
                        params["global_forecast_bias"] = bias
                        params["global_forecast_uncertainty"] = uncertainty
                        params["station"] = stations[station]["station"]
                        params["DOE"] = stations[station]["DOE"]
                        params["DCR"] = stations[station]["DCR"]

                        # Initialize simulation
                        simulation = wms.WaterManagementSimulation(**params)

                        # Set custom policies
                        simulation.incentive_policy = MethodType(
                            custom_incentive_policy, simulation
                        )
                        simulation.compute_actor_quota = MethodType(
                            custom_quota, simulation
                        )

                        # Run simulation
                        simulation.run_simulation()

                        # Get scores
                        ecological_impact, economic_impact, priority_ok = (
                            simulation.get_final_scores_scaled()
                        )

                        # Calculate raw ecological impact (total number of breaches)
                        raw_ecol_impact = np.sum(simulation.w_ecol_impact > 0)
                        raw_econ_impact = np.sum(simulation.h_econ_impacts)
                        # Calculate average percentage of cooperators
                        # Average over all iterations and turns
                        avg_coop = np.mean(simulation.h_actions)

                        # Store results
                        all_ecological_impact.append(ecological_impact)
                        all_economic_impact.append(economic_impact)
                        all_biases.append(bias)
                        all_uncertainties.append(uncertainty)
                        all_scarcities.append(scarcity)
                        all_scenarios.append(scenario)
                        all_stations.append(station)
                        all_priority_ok.append(priority_ok)

                        # Store new metrics
                        all_raw_ecological_impact.append(float(raw_ecol_impact))
                        all_raw_economic_impact.append(float(raw_econ_impact))
                        all_cooperation_percentage.append(float(avg_coop))

                        print(
                            f"Scenario: {scenario}, Station: {station}, Scarcity: {scarcity}, "
                            + f"Bias: {bias}, Uncertainty: {uncertainty}, "
                            + f"Eco Impact: {ecological_impact:.3f}, Econ Impact: {economic_impact:.3f}, "
                            + f"Raw Eco Impact: {raw_ecol_impact:.1f}, Cooperation %: {avg_coop*100:.1f}%"
                        )

    # Create DataFrame with results
    results_df = pd.DataFrame(
        {
            "ecological_impact": all_ecological_impact,
            "economic_impact": all_economic_impact,
            "bias": all_biases,
            "uncertainty": all_uncertainties,
            "scarcity": all_scarcities,
            "scenario": all_scenarios,
            "station": all_stations,
            "raw_ecological_impact": all_raw_ecological_impact,
            "raw_economic_impact": all_raw_economic_impact,
            "cooperation_percentage": all_cooperation_percentage,
            "priority_ok": all_priority_ok,
        }
    )

    # Add color mappings
    scarcity_colors = {"low": "yellow", "medium": "orange", "high": "red"}
    scenario_colors = {
        "0.yml": "blue",
        "1.yml": "red",
        "0-v.yml": "purple",
        "1-v.yml": "orange",
        "0-b.yml": "blue",
        "1-b.yml": "red",
        "0-c.yml": "blue",
        "1-c.yml": "red",
    }
    scenario_names = {
        "0.yml": "0",
        "1.yml": "1",
        "0-v.yml": "0-v",
        "1-v.yml": "1-v",
        "0-b.yml": "0-b",
        "1-b.yml": "1-b",
        "0-c.yml": "0-c",
        "1-c.yml": "1-c",
    }
    station_colors = {"1": "green", "2": "blue"}

    results_df["scarcity_color"] = results_df["scarcity"].map(scarcity_colors)
    results_df["scenario_color"] = results_df["scenario"].map(scenario_colors)
    results_df["scenario_name"] = results_df["scenario"].map(scenario_names)
    results_df["station_color"] = results_df["station"].astype(str).map(station_colors)

    print(f"Completed {len(results_df)} simulation runs")

    return results_df
