"""Plot Analysis Module for Water Management Simulation.

This module provides visualization functions for analyzing simulation results,
including river flows, actor behaviors, ecological impacts, and economic outcomes.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import src.core as wms

# =====================================================================
# Styling and Visual Configuration
# =====================================================================

# Font sizes for consistent text across plots
FONT_SIZES = {
    "title": 20,  # Main title font size
    "axis_label": 14,  # Axis labels font size
    "legend": 11,  # Legend text font size
    "tick": 12,  # Tick labels font size
}

# Color schemes for different plot elements
COLOR_SCHEMES = {
    "actor": [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf",  # Cyan
        "#7570b3",  # Indigo
        "#e7298a",  # Magenta
    ],
    "data": {
        # Flow data
        "river_flow": "#33BBEE",  # Blue for river flow
        "eco_threshold": "#E31A1C",  # Red for ecological threshold
        "remaining_water": "#33A02C",  # Green for remaining water
        "residual_water": "#FB9A99",  # Light red for residual water
        # Water usage
        "h_water_used": "#1F78B4",  # Blue for water used
        "h_water_pump": "#E31A1C",  # Red for water pumped
        # Ecological impact
        "w_ecol_impact": "#FF7F00",  # Orange for actual ecological impact
        "w_min_ecol_impact": "#33A02C",  # Green for minimum ecological impact
        "w_max_ecol_impact": "#E31A1C",  # Red for maximum ecological impact
        "w_crisis": "#FB9A99",  # Light red for crisis levels
        # Economic impact
        "h_econ_impacts": "#1F78B4",  # Blue for actual economic impact
        "h_max_econ_impacts": "#E31A1C",  # Red for maximum economic impact
        "h_min_econ_impacts": "#33A02C",  # Green for minimum economic impact
        # Actor behavior
        "h_policies": "#FF7F00",  # Orange for policies
        "h_tendencies": "#6A3D9A",  # Purple for tendencies
        "h_actions": "#A6CEE3",  # Light blue for actions
        "h_rewards": "#B2DF8A",  # Light green for rewards
        "h_diff_rewards": "#FB9A99",  # Light red for reward differences
        # Storage
        "h_storage": "#6A3D9A",  # Purple for storage
        "h_alt_storage": "#CAB2D6",  # Light purple for alternative storage
        # Quotas and incentives
        "h_quota": "#FDBF6F",  # Light orange for quotas
        "h_fines": "#E31A1C",  # Red for fines
        "h_subventions": "#33A02C",  # Green for subventions
        "h_taxed_incomes": "#B15928",  # Brown for taxed incomes
    },
}

# Labels for different data types
DATA_LABELS = {
    # Water flow
    "river_flow": "River Flow",
    "eco_threshold": "Ecological Threshold",
    "remaining_water": "Remaining Water",
    # Water usage
    "h_water_used": "Water Used",
    "h_water_pump": "Water Pumped",
    # Ecological impact
    "w_ecol_impact": "Ecological Impact",
    "w_min_ecol_impact": "Min Ecological Impact",
    "w_max_ecol_impact": "Max Ecological Impact",
    "w_crisis": "Crisis Level",
    # Economic impact
    "h_econ_impacts": "Economic Impacts",
    "h_max_econ_impacts": "Max Economic Impacts",
    "h_min_econ_impacts": "Sustainable Economic Impacts",
    # Actor behavior
    "h_policies": "Policies",
    "h_tendencies": "Proportion of Cooperators",
    "h_actions": "Actions (1=Cooperate, 0=Defect)",
    "h_rewards": "Rewards",
    "h_diff_rewards": "Reward Differences",
    # Storage
    "h_storage": "Storage",
    "h_alt_storage": "Alternative Storage",
    # Quotas and incentives
    "h_quota": "Quota",
    "h_fines": "Fines",
    "h_subventions": "Subventions",
    "h_taxed_incomes": "Taxed Incomes",
}

# Plot titles (if different from labels)
PLOT_TITLES = {
    "h_econ_impacts": "Economic Impacts",
    "h_max_econ_impacts": "Maximum Economic Impacts",
    "h_min_econ_impacts": "Sustainable Economic Impacts",
    "h_tendencies": "Cooperation Tendencies",
    "h_actions": "Actor Actions",
    "w_crisis": "Crisis Levels",
    "h_water_used": "Water Used by Actors",
    "h_water_pump": "Water Pumped by Actors",
}


# =====================================================================
# Helper Functions
# =====================================================================


def get_color(data_type: str) -> str:
    """Get the color associated with a specific data type.

    Parameters:
    -----------
    data_type : str
        The data type identifier

    Returns:
    --------
    str
        Color code for the data type
    """
    if data_type in COLOR_SCHEMES["data"]:
        return COLOR_SCHEMES["data"][data_type]
    return "#000000"  # Default to black if not found


def get_label(data_type: str) -> str:
    """Get the display label for a specific data type.

    Parameters:
    -----------
    data_type : str
        The data type identifier

    Returns:
    --------
    str
        Display label for the data type
    """
    if data_type in DATA_LABELS:
        return DATA_LABELS[data_type]
    return data_type.replace("_", " ").title()  # Convert snake_case to Title Case


def get_title(data_type: str) -> str:
    """Get the plot title for a specific data type.

    Parameters:
    -----------
    data_type : str
        The data type identifier

    Returns:
    --------
    str
        Plot title for the data type
    """
    if data_type in PLOT_TITLES:
        return PLOT_TITLES[data_type]
    elif data_type in DATA_LABELS:
        return DATA_LABELS[data_type]
    return data_type.replace("_", " ").title()


def apply_common_style(
    ax, title=None, xlabel=None, ylabel=None, legend=True, legend_title=None
):
    """Apply consistent styling to a plot.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to style
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    legend : bool
        Whether to show the legend
    legend_title : str, optional
        Title for the legend
    """
    # Set title if provided
    if title:
        ax.set_title(title, fontsize=FONT_SIZES["title"])

    # Set axis labels if provided
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZES["axis_label"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZES["axis_label"])

    # Set tick label font sizes
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZES["tick"])

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="-")

    # Format legend if needed
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            leg = ax.legend(
                handles=handles,
                labels=labels,
                fontsize=FONT_SIZES["legend"],
                frameon=True,
                framealpha=0.8,
                facecolor="white",
                edgecolor="gray",
            )
            if legend_title:
                leg.set_title(legend_title, prop={"size": FONT_SIZES["legend"] + 1})


# =====================================================================
# Data Extraction Functions
# =====================================================================


def get_h_data(
    simulation: wms, data_type: str, mode: str = "sum", by_type: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and process h_* data from simulation.

    Parameters:
    -----------
    simulation : wms
        Simulation object
    data_type : str
        The data type to extract (must start with 'h_')
    mode : str
        Processing mode: "sum", "mean", or "actors"
    by_type : bool
        Whether to group actors by type

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (mean, upper_quantile, lower_quantile)
    """
    # Get raw data (shape: actors, runs, turns)
    data = simulation.get_data(data_type)

    # Process by actor type if requested
    if by_type:
        # Get unique actor types
        types = np.unique(simulation.actors_type)
        per_type = []

        # Process each type separately
        for t in types:
            # Select only actors of this type
            sel = data[
                simulation.actors_type == t, :, :
            ]  # shape: (actors_of_type, runs, turns)

            # Apply aggregation mode
            if mode in ("sum", "actors_sum"):
                # Sum over actors axis → shape: (runs, turns)
                arr = sel.sum(axis=0)
            elif mode in ("mean", "actors_mean"):
                # Mean over actors axis → shape: (runs, turns)
                arr = sel.mean(axis=0)
            else:
                raise ValueError(f"Unknown mode={mode!r} for by_type")

            per_type.append(arr)

        # Stack types into shape: (types, runs, turns)
        arr = np.stack(per_type, axis=0)

        # Compute statistics over runs axis → shape: (types, turns)
        mean = arr.mean(axis=1)
        q_up = np.quantile(arr, 0.9, axis=1)
        q_down = np.quantile(arr, 0.1, axis=1)

        # Fix cases where quantiles are outside the range of data
        # This can happen with skewed distributions
        for i in range(mean.shape[0]):
            q_up[i] = np.maximum(q_up[i], mean[i])
            q_down[i] = np.minimum(q_down[i], mean[i])

        return mean, q_up, q_down

    # Process without type grouping
    axis = 0  # Default aggregation axis

    # Apply aggregation mode
    if mode == "sum":
        data = data.sum(axis=0)  # Sum over actors → shape: (runs, turns)
    elif mode == "mean":
        data = data.mean(axis=0)  # Mean over actors → shape: (runs, turns)
    elif mode == "actors":
        axis = 1  # Keep actors dimension, aggregate over runs

    # Compute statistics over appropriate axis
    mean = data.mean(axis=axis)
    q_up = np.quantile(data, 0.9, axis=axis)
    q_down = np.quantile(data, 0.1, axis=axis)

    # Fix cases where quantiles are outside the range of data
    if axis == 0:  # Single mean series
        q_up = np.maximum(q_up, mean)
        q_down = np.minimum(q_down, mean)
    else:  # Multiple series (one per actor)
        for i in range(mean.shape[0]):
            q_up[i] = np.maximum(q_up[i], mean[i])
            q_down[i] = np.minimum(q_down[i], mean[i])

    return mean, q_up, q_down


def get_w_data(
    simulation: wms, data_type: str, mode: str = "mean"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and process w_* data from simulation.

    Parameters:
    -----------
    simulation : wms
        Simulation object
    data_type : str
        The data type to extract (must start with 'w_')
    mode : str
        Processing mode (currently only "mean" is supported)

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (mean, upper_quantile, lower_quantile)
    """
    # Get raw data (shape: runs, turns)
    data = simulation.get_data(data_type)

    # Compute statistics over runs axis
    mean = np.mean(data, axis=0)
    q_up = np.quantile(data, 0.8, axis=0)
    q_down = np.quantile(data, 0.2, axis=0)

    # Fix cases where quantiles are outside the range of data
    q_up = np.maximum(q_up, mean)
    q_down = np.minimum(q_down, mean)

    return mean, q_up, q_down


# =====================================================================
# Core Plotting Functions
# =====================================================================


def plot_data(
    ax: plt.Axes,
    x_values: np.ndarray,
    simulation: wms,
    data_type: str,
    mode: str = "sum",
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Plot the mean and confidence interval of a data series.

    Parameters:
    -----------
    ax : plt.Axes
        Matplotlib axes to plot on
    x_values : np.ndarray
        X-axis values (typically turn numbers)
    simulation : wms
        Simulation object containing the data
    data_type : str
        Type of data to plot (e.g., 'h_tendencies', 'w_ecol_impact')
    mode : str, optional
        Aggregation mode: "sum", "mean", or "actors"
    ylim : Tuple[float, float], optional
        Y-axis limits
    """
    # Get the appropriate data based on prefix
    if data_type.startswith("w_"):
        mean, q_up, q_down = get_w_data(simulation, data_type, mode)
    elif data_type.startswith("h_"):
        mean, q_up, q_down = get_h_data(simulation, data_type, mode)
    else:
        raise ValueError(f"Unknown data type prefix: {data_type}")

    # Get color, label, and title
    color = get_color(data_type)
    label = get_label(data_type)
    title = get_title(data_type)

    # Plot mean line
    ax.plot(x_values, mean, label=label, color=color, linewidth=2)

    # Add confidence interval
    ax.fill_between(x_values, q_down, q_up, alpha=0.2, color=color)

    # Apply styling
    apply_common_style(ax, title=title, xlabel="Turn (Week)", ylabel=label)

    # Set y-axis limits if provided
    if ylim is not None:
        ax.set_ylim(ylim)


def plot_actor_data(
    ax: plt.Axes,
    x_values: np.ndarray,
    simulation: wms,
    data_type: str,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Plot data for each actor individually.

    Parameters:
    -----------
    ax : plt.Axes
        Matplotlib axes to plot on
    x_values : np.ndarray
        X-axis values (typically turn numbers)
    simulation : wms
        Simulation object containing the data
    data_type : str
        Type of data to plot (must start with 'h_')
    ylim : Tuple[float, float], optional
        Y-axis limits
    """
    # Check data type
    if not data_type.startswith("h_"):
        raise ValueError(f"Actor data must be h_* type, got {data_type}")

    # Get the number of actors and their labels
    nb_actors = simulation.nb_actors
    actor_labels = simulation.actors_name

    # Get data for each actor
    mean, q_up, q_down = get_h_data(simulation, data_type, mode="actors")

    # Get label and title
    title = get_title(data_type)
    ylabel = get_label(data_type)

    # Plot data for each actor
    for i in range(nb_actors):
        color = COLOR_SCHEMES["actor"][i % len(COLOR_SCHEMES["actor"])]
        label = f"Actor {i + 1} ({actor_labels[i]})"

        # Plot mean line
        ax.plot(x_values, mean[i], label=label, color=color, linewidth=1.8)

        # Add confidence interval
        ax.fill_between(x_values, q_down[i], q_up[i], alpha=0.15, color=color)

    # Apply styling
    apply_common_style(
        ax,
        title=f"{title} by Actor",
        xlabel="Turn (Week)",
        ylabel=ylabel,
        legend_title="Actors",
    )

    # Set y-axis limits if provided
    if ylim is not None:
        ax.set_ylim(ylim)


def plot_actor_data_by_type(
    ax: plt.Axes,
    x_values: np.ndarray,
    simulation: wms,
    data_type: str,
    mode: str = "actors_sum",
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Plot data aggregated by actor type.

    Parameters:
    -----------
    ax : plt.Axes
        Matplotlib axes to plot on
    x_values : np.ndarray
        X-axis values (typically turn numbers)
    simulation : wms
        Simulation object containing the data
    data_type : str
        Type of data to plot (must start with 'h_')
    mode : str, optional
        Aggregation mode: "actors_sum" or "actors_mean"
    ylim : Tuple[float, float], optional
        Y-axis limits
    """
    # Check data type
    if not data_type.startswith("h_"):
        raise ValueError(f"Actor data must be h_* type, got {data_type}")

    # Get unique actor types
    types = np.unique(simulation.actors_type)

    # Get data aggregated by type
    mean, q_up, q_down = get_h_data(simulation, data_type, mode=mode, by_type=True)

    # Get label and title
    title = get_title(data_type)
    ylabel = get_label(data_type)

    # Plot data for each actor type
    for i, t in enumerate(types):
        color = COLOR_SCHEMES["actor"][i % len(COLOR_SCHEMES["actor"])]
        label = f'Type "{t}"'

        # Plot mean line
        ax.plot(x_values, mean[i], label=label, color=color, linewidth=2)

        # Add confidence interval
        ax.fill_between(x_values, q_down[i], q_up[i], alpha=0.2, color=color)

    # Apply styling
    apply_common_style(
        ax,
        title=f"{title} by Actor Type",
        xlabel="Turn (Week)",
        ylabel=ylabel,
        legend_title="Actor Types",
    )

    # Set y-axis limits if provided
    if ylim is not None:
        ax.set_ylim(ylim)


def compare_data(
    ax: plt.Axes,
    x_values: np.ndarray,
    cols: List[str],
    simulation: wms,
    xlabel: str = "Turn (Week)",
    ylabel: str = "",
    title: str = "",
    ylim: Optional[Tuple[float, float]] = None,
    cumulative: bool = False,
) -> None:
    """Compare multiple data series on the same plot.

    Parameters:
    -----------
    ax : plt.Axes
        Matplotlib axes to plot on
    x_values : np.ndarray
        X-axis values (typically turn numbers)
    cols : List[str]
        List of data types to plot
    simulation : wms
        Simulation object containing the data
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    title : str, optional
        Plot title
    ylim : Tuple[float, float], optional
        Y-axis limits
    cumulative : bool, optional
        Whether to show cumulative values
    """
    # Plot each data series
    for col in cols:
        # Get the appropriate data based on prefix
        if col.startswith("w_"):
            mean, q_up, q_down = get_w_data(simulation, col, "mean")
        elif col.startswith("h_"):
            mean, q_up, q_down = get_h_data(simulation, col, "mean")
        else:
            raise ValueError(f"Unknown data type prefix: {col}")

        # Apply cumulative sum if requested
        if cumulative:
            mean = np.cumsum(mean)
            q_up = np.cumsum(q_up)
            q_down = np.cumsum(q_down)

        # Get label and color
        label = get_label(col)
        color = get_color(col)

        # Plot mean line
        ax.plot(x_values, mean, label=label, color=color, linewidth=2)

        # Add confidence interval
        ax.fill_between(x_values, q_down, q_up, alpha=0.2, color=color)

    # Apply styling
    apply_common_style(ax, title=title, xlabel=xlabel, ylabel=ylabel)

    # Set y-axis limits if provided
    if ylim is not None:
        ax.set_ylim(ylim)


def plot_river_flow(
    ax: plt.Axes,
    weeks: np.ndarray,
    simulation: wms,
    ylim: tuple = None,
    xlim: tuple = None,
) -> None:
    """Top panel: river flow, each actor’s weekly demand, and the DOE/DCR thresholds.

    Parameters
    ----------
    ax
        Matplotlib axes on which to draw.
    weeks
        1D array of week indices.
    simulation : wms
        Simulation object
    ylim
        Optional (ymin, ymax).
    xlim
        Optional (xmin, xmax).
    """
    DOE = simulation.DOE
    DCR = simulation.DCR
    actor_labels = simulation.actors_name
    actor_demands = simulation.actors_demands
    actor_colors = COLOR_SCHEMES["actor"]
    river_flow = simulation.real_riverflows[weeks]
    # pronounced y=0 axis
    ax.axhline(0, color="black", linewidth=2)

    # river flow
    ax.plot(weeks, river_flow, label="River flow", lw=2)

    # actor demands
    for i, (lbl, d) in enumerate(zip(actor_labels, actor_demands)):
        ax.hlines(
            d,
            weeks[0],
            weeks[-1],
            color=actor_colors[i % len(actor_colors)],
            ls="--",
            lw=1.8,
            label=f"Demand – {lbl}",
        )

    # DOE & DCR
    ax.hlines(DOE, weeks[0], weeks[-1], color="C3", ls="-.", lw=2, label="DOE")
    ax.hlines(DCR, weeks[0], weeks[-1], color="C4", ls=":", lw=2, label="DCR")

    ax.set_title("River Flow & Actor Demands")
    ax.set_xlabel("Week")
    ax.set_ylabel("Flow (units)")
    if ylim:
        ax.set_ylim(*ylim)
    if xlim:
        ax.set_xlim(*xlim)
    ax.legend(ncol=2, fontsize="small", loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_remaining_water(
    ax: plt.Axes,
    weeks: np.ndarray,
    simulation: wms,
    ylim: tuple = None,
    xlim: tuple = None,
) -> None:
    """Bottom panel: water remaining after ecological thresholds and after meeting all demands.

    Parameters
    ----------
    ax
        Matplotlib axes on which to draw.
    weeks
        1D array of week indices.
    simulation : wms
        Simulation
    ylim
        Optional (ymin, ymax).
    xlim
        Optional (xmin, xmax).
    """
    DOE = simulation.DOE
    DCR = simulation.DCR
    actor_demands = simulation.actors_demands
    river_flow = simulation.real_riverflows[weeks]
    # pronounced y=0 axis
    ax.axhline(0, color="black", linewidth=2)

    # surplus above DOE & DCR
    surplus_DOE = river_flow - DOE
    surplus_DCR = river_flow - DCR
    total_demand = actor_demands.sum()
    residual = river_flow - DOE - total_demand

    ax.plot(weeks, river_flow, label="Flow", lw=2)
    ax.plot(weeks, surplus_DOE, label="Flow – DOE", lw=2, ls="--")
    ax.plot(weeks, surplus_DCR, label="Flow – DCR", lw=2, ls="--")
    ax.plot(weeks, residual, label="Residual after demands", lw=2, ls=":")

    # highlight ecological deficit
    deficit = residual < 0
    if deficit.any():
        ax.fill_between(
            weeks,
            residual,
            0,
            where=deficit,
            color="red",
            alpha=0.2,
            label="Ecological deficit",
        )

    ax.set_title("Available Water after Thresholds & Demands")
    ax.set_xlabel("Week")
    ax.set_ylabel("Volume (units)")
    if ylim:
        ax.set_ylim(*ylim)
    if xlim:
        ax.set_xlim(*xlim)
    ax.legend(ncol=2, fontsize="small", loc="upper right")
    ax.grid(True, alpha=0.3)
