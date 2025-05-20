"""Water Management Analysis Module.

This module provides visualization and analysis functions for water allocation
simulation results. It includes tools for analyzing the relationships between
ecological impacts, economic impacts, and various simulation parameters.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Set consistent styling for all plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")


# =====================================================================
# Common Utilities and Color Schemes
# =====================================================================

# Color schemes for different plot elements
COLOR_SCHEMES = {
    "scarcity": {
        "low": "#2E8B57",  # Sea Green - Calm, good conditions
        "medium": "#FFD700",  # Gold - Caution
        "high": "#B22222",  # Firebrick - Danger, severe conditions
    },
    "station": {
        1: "#4682B4",  # Steel Blue - La Vézère
        2: "#8A2BE2",  # Blue Violet - Le Tarn
    },
    "priority_ok": {
        1: "#2E8B57",  # Sea Green - Satisfy priority
        0: "#FF6347",  # Tomato - Do not satisfy priority
    },
    "scenario": {
        "0.yml": {"color": "#1E90FF", "marker": "o", "name": "Base 0"},  # Dodger Blue
        "1.yml": {"color": "#FF6347", "marker": "o", "name": "Base 1"},  # Tomato
        "0-v.yml": {
            "color": "#1E90FF",
            "marker": "s",
            "name": "Variant 0-v",
        },  # Dodger Blue, square
        "1-v.yml": {
            "color": "#FF6347",
            "marker": "s",
            "name": "Variant 1-v",
        },  # Tomato, square
        "0-b.yml": {
            "color": "#1E90FF",
            "marker": "^",
            "name": "Variant 0-b",
        },  # Dodger Blue, triangle
        "1-b.yml": {
            "color": "#FF6347",
            "marker": "^",
            "name": "Variant 1-b",
        },  # Tomato, triangle
        "0-c.yml": {
            "color": "#1E90FF",
            "marker": "D",
            "name": "Variant 0-c",
        },  # Dodger Blue, diamond
        "1-c.yml": {
            "color": "#FF6347",
            "marker": "D",
            "name": "Variant 1-c",
        },  # Tomato, diamond
    },
    "correlation": "coolwarm",
    "boxplot": "GnBu",
    "violin": {
        "low": "#2E8B57",  # Sea Green
        "medium": "#FFD700",  # Gold
        "high": "#B22222",  # Firebrick
    },
}


def create_styled_legend(ax, handles, labels, title, position=None, ncol=1):
    """Create a consistently styled legend.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add the legend to
    handles : list
        Legend handles
    labels : list
        Legend labels
    title : str
        Legend title
    position : tuple, optional
        Position to place legend (default: top-right outside plot)
    ncol : int
        Number of columns in the legend

    Returns:
    --------
    legend : matplotlib.legend.Legend
        The created legend object
    """
    if position is None:
        legend = ax.legend(
            handles=handles,
            labels=labels,
            title=title,
            fontsize=10,
            title_fontsize=11,
            frameon=True,
            facecolor="white",
            edgecolor="gray",
            framealpha=0.95,
            loc="best",
            ncol=ncol,
        )
    else:
        legend = ax.legend(
            handles=handles,
            labels=labels,
            title=title,
            fontsize=10,
            title_fontsize=11,
            frameon=True,
            facecolor="white",
            edgecolor="gray",
            framealpha=0.95,
            bbox_to_anchor=position,
            ncol=ncol,
        )
    legend.get_frame().set_linewidth(0.5)
    return legend


def add_impact_plot_elements(ax, results_df=None):
    """Add common elements to impact plots like diagonal line and best/worst regions.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to annotate
    results_df : pandas.DataFrame, optional
        Results data for calculating min/max values

    Returns:
    --------
    None
    """
    # Get current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Draw diagonal reference line using current limits
    # Inverse relationship: lower ecological impact, higher economic impact
    ax.plot([xlim[0], xlim[1]], [ylim[1], ylim[0]], "k--", alpha=0.3, linewidth=1)

    # Calculate ranges
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    # Add shaded rectangles for best/worst regions
    if results_df is not None:
        # Best region: low ecological impact, high economic impact
        x_best = np.linspace(xlim[0], xlim[0] + 0.33 * x_range, 50)
        y_best_bottom = np.ones_like(x_best) * (ylim[1] - 0.33 * y_range)
        y_best_top = np.ones_like(x_best) * ylim[1]
        ax.fill_between(x_best, y_best_top, y_best_bottom, color="#E5F5E0", alpha=0.2)

        # Worst region: high ecological impact, low economic impact
        x_worst = np.linspace(xlim[0] + 0.67 * x_range, xlim[1], 50)
        y_worst_bottom = np.ones_like(x_worst) * ylim[0]
        y_worst_top = np.ones_like(x_worst) * (ylim[0] + 0.33 * y_range)
        ax.fill_between(
            x_worst, y_worst_top, y_worst_bottom, color="#FEE5D9", alpha=0.2
        )

        # Add labels for best/worst regions
        ax.text(
            xlim[0] + 0.10 * x_range,
            ylim[1] - 0.10 * y_range,
            "Best",
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(
                facecolor="white",
                alpha=0.7,
                edgecolor="green",
                boxstyle="round,pad=0.3",
            ),
        )
        ax.text(
            xlim[1] - 0.10 * x_range,
            ylim[0] + 0.10 * y_range,
            "Worst",
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(
                facecolor="white", alpha=0.7, edgecolor="red", boxstyle="round,pad=0.3"
            ),
        )

    # Improve grid appearance
    ax.grid(True, alpha=0.2, linestyle="-")

    # Enhance boundary lines
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")
        spine.set_linewidth(0.5)

    # Common axis labels for impact plots
    ax.set_xlabel("Ecological Impact (lower is better)", fontsize=12)
    ax.set_ylabel("Economic Impact (higher is better)", fontsize=12)


# =====================================================================
# Primary Analysis Functions
# =====================================================================


def analyze_scenario_impacts(results_df, ylim=None, xlim=None, is_scalled=False):
    """Visualize and analyze the impacts of different scenarios.

    Analyze impacts across different scenarios, stations, and scarcity levels
    with improved visualization aesthetics.

    Parameters
    -----------
    results_df : pd.DataFrame
        DataFrame containing simulation results with ecological_impact, economic_impact,
        and scenario metadata
    ylim : tuple, optional
        Y-axis limits for the plots
    xlim : tuple, optional
        X-axis limits for the plots
    is_scalled : bool, optional
        If True, use scaled impact values; otherwise, use raw values

    Returns
    --------
        None
    """
    # 1. Analysis by scarcity level
    create_impact_by_category_plot(
        results_df,
        category="scarcity",
        color_map=COLOR_SCHEMES["scarcity"],
        title="Trade-off Between Ecological and Economic Impacts by Scarcity Level",
        point_size=60,
        xlim=xlim,
        ylim=ylim,
        is_scalled=is_scalled,
    )

    # 2. Analysis by station (river basin)
    create_impact_by_category_plot(
        results_df,
        category="station",
        color_map=COLOR_SCHEMES["station"],
        label_format=lambda s: f'Station : {"La Vézère" if s==1 else "Le Tarn"}',
        title="Impact by River Basin Size",
        point_size=60,
        xlim=xlim,
        ylim=ylim,
        is_scalled=is_scalled,
    )

    # 3. Analysis by priority satisfaction
    create_impact_by_category_plot(
        results_df,
        category="priority_ok",
        color_map=COLOR_SCHEMES["priority_ok"],
        label_format=lambda s: f'Statisfaction Criteria : {"ok" if s==1.0 else "not ok"}',
        title="Trade-off Between Ecological and Economic Impacts by satisfaction level",
        point_size=60,
        xlim=xlim,
        ylim=ylim,
        is_scalled=is_scalled,
    )

    # 3. Analysis by scenario
    create_impact_by_scenario_plot(
        results_df,
        xlim=xlim,
        ylim=ylim,
        is_scalled=is_scalled,
    )

    # 4. Combined multi-dimensional analysis
    create_multidimensional_impact_plot(results_df, is_scalled=is_scalled)


def analyze_forecast_effects(results_df, is_scalled=False):
    """Analyze the effects of forecast bias and uncertainty on simulation outcomes.

    Creates matrix of plots showing relationships between bias, uncertainty,
    and impact measures.

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing simulation results
    is_scalled : bool, optional
        If True, use scaled impact values; otherwise, use raw values

    Returns:
    --------
    None
        Displays visualizations showing forecast effects
    """
    # Extract base scenarios (without variants) for clearer analysis
    base_df = results_df[results_df["scenario"].isin(["0.yml", "1.yml"])]

    if is_scalled:
        ecological_impact = "ecological_impact"
        economic_impact = "economic_impact"
    else:
        ecological_impact = "raw_ecological_impact"
        economic_impact = "raw_economic_impact"

    # Create forecast analysis plots (2x2 grid)
    fig, axs = plt.subplots(2, 2, figsize=(18, 14))

    # Define variables and labels for each subplot
    plot_configs = [
        {
            "x": "bias",
            "y": economic_impact,
            "size_var": "uncertainty",
            "abs_size": False,
            "ax": axs[0, 0],
        },
        {
            "x": "bias",
            "y": ecological_impact,
            "size_var": "uncertainty",
            "abs_size": False,
            "ax": axs[0, 1],
        },
        {
            "x": "uncertainty",
            "y": economic_impact,
            "size_var": "bias",
            "abs_size": True,
            "ax": axs[1, 0],
        },
        {
            "x": "uncertainty",
            "y": ecological_impact,
            "size_var": "bias",
            "abs_size": True,
            "ax": axs[1, 1],
        },
    ]

    # Create each subplot
    for config in plot_configs:
        create_forecast_effect_plot(base_df, **config)

    # Create a common legend for all subplots instead of individual legends
    scarcity_handles = []
    for scarcity, color in COLOR_SCHEMES["scarcity"].items():
        handle = Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            markersize=10,
            linestyle="None",
            label=f"Scarcity: {scarcity}",
        )
        scarcity_handles.append(handle)

    # Create size legend for uncertainty
    size_handles = []
    sizes = [0.1, 0.3, 0.5]
    size_scale = (50, 200)
    scale_range = size_scale[1] - size_scale[0]
    for size in sizes:
        marker_size = size_scale[0] + scale_range * (size / 0.5)
        handle = Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            markersize=np.sqrt(marker_size / 3),
            linestyle="None",
            label=f"{size:.1f}",
        )
        size_handles.append(handle)

    # Add the legends to the figure, not to any specific axes
    fig.legend(
        handles=scarcity_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        fontsize=12,
        frameon=True,
        facecolor="white",
        edgecolor="gray",
        title="Scarcity Level",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for the legend at the bottom
    plt.show()

    # Create visualization of impact distributions using violin plots
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))

    # Create enhanced violin plots for better distribution visualization
    for i, impact in enumerate([ecological_impact, economic_impact]):
        # Create violin plot
        sns.violinplot(
            data=results_df,
            x="scarcity",
            y=impact,
            palette=COLOR_SCHEMES["violin"],
            ax=axs[i],
            inner="quartile",  # Show quartiles inside violin
            density_norm="width",  # Scale violins to have same width
        )

        # Add individual data points using stripplot
        sns.stripplot(
            data=results_df,
            x="scarcity",
            y=impact,
            ax=axs[i],
            size=4,
            alpha=0.6,
            jitter=True,
            color="black",
            zorder=1,  # Ensure points are above violin plot
        )

        # Improve appearance
        axs[i].set_title(
            f'{impact.replace("_", " ").title()} Distribution by Scarcity Level',
            fontsize=15,
        )
        axs[i].set_xlabel("Scarcity Level", fontsize=12)
        axs[i].set_ylabel(impact.replace("_", " ").title(), fontsize=12)

        # Add mean markers
        mean_values = results_df.groupby("scarcity")[impact].mean()
        for j, scarcity in enumerate(sorted(results_df["scarcity"].unique())):
            axs[i].plot(
                j, mean_values[scarcity], "o", color="white", markersize=8, zorder=3
            )
            axs[i].plot(
                j, mean_values[scarcity], "x", color="black", markersize=6, zorder=4
            )

    plt.tight_layout()
    plt.show()


def correlation_analysis(results_df, is_scalled=False):
    """Correlation analysis.

    Perform targeted correlation analysis focusing on relationships between
    input parameters and simulation outcomes.

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing simulation results
    is_scalled : bool, optional
        If True, use scaled impact values; otherwise, use raw values

    Returns:
    --------
    None
        Displays correlation visualizations
    """
    if is_scalled:
        ecological_impact = "ecological_impact"
        economic_impact = "economic_impact"
    else:
        ecological_impact = "raw_ecological_impact"
        economic_impact = "raw_economic_impact"

    # Separate input and output variables
    output_vars = [ecological_impact, economic_impact]
    input_numeric_vars = ["bias", "uncertainty"]
    input_categorical_vars = ["scarcity", "scenario", "station"]

    # Create dummy variables for categorical inputs
    cat_df = pd.get_dummies(results_df[input_categorical_vars], drop_first=False)

    # Combine all input variables
    input_df = pd.concat([results_df[input_numeric_vars], cat_df], axis=1)

    # Calculate correlation between inputs and outputs
    correlation_results = calculate_input_output_correlation(
        input_df, results_df[output_vars]
    )

    # Visualize correlations
    plot_input_output_correlation_heatmap(correlation_results)

    # Create scatter plots for key numeric relationships
    plot_key_numeric_relationships(results_df, input_numeric_vars, output_vars)

    # Plot feature importance through a simple regression model
    plot_feature_importance(input_df, results_df[output_vars])


def calculate_input_output_correlation(input_df, output_df):
    """Calculate correlation between input variables and output metrics.

    Parameters:
    -----------
    input_df : pd.DataFrame
        DataFrame containing input variables
    output_df : pd.DataFrame
        DataFrame containing output variables

    Returns:
    --------
    pd.DataFrame
        Correlation matrix between inputs and outputs
    """
    # Calculate correlation between all input variables and output metrics
    correlation_matrix = pd.DataFrame()

    for output_var in output_df.columns:
        # Calculate correlation of each input with this output
        correlations = input_df.apply(lambda x: x.corr(output_df[output_var]))
        correlation_matrix[output_var] = correlations

    # Sort by the maximum absolute correlation across outputs
    correlation_matrix["max_abs_corr"] = correlation_matrix.abs().max(axis=1)
    sorted_matrix = correlation_matrix.sort_values("max_abs_corr", ascending=False)

    # Remove the helper column
    sorted_matrix = sorted_matrix.drop("max_abs_corr", axis=1)

    return sorted_matrix


def plot_input_output_correlation_heatmap(correlation_matrix, min_correlation=0.05):
    """Plot a heatmap of correlations between inputs and outputs.

    Parameters:
    -----------
    correlation_matrix : pd.DataFrame
        Correlation matrix between inputs and outputs
    min_correlation : float
        Minimum absolute correlation to include in the plot
    """
    # Filter for significant correlations
    significant_correlations = correlation_matrix[
        correlation_matrix.abs().max(axis=1) >= min_correlation
    ]

    if len(significant_correlations) == 0:
        print("No significant correlations found.")
        return

    # Create a more focused heatmap
    plt.figure(figsize=(10, max(8, len(significant_correlations) * 0.4)))
    sns.heatmap(
        significant_correlations,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
    )
    plt.title(
        "Correlation Between Input Parameters and Simulation Outcomes", fontsize=16
    )
    plt.tight_layout()
    plt.show()

    # Print the top positive and negative correlations for each output
    print_top_correlations(correlation_matrix)


def print_top_correlations(correlation_matrix, top_n=5):
    """Print the top positive and negative correlations for each output variable.

    Parameters:
    -----------
    correlation_matrix : pd.DataFrame
        Correlation matrix between inputs and outputs
    top_n : int
        Number of top correlations to print
    """
    for output_var in correlation_matrix.columns:
        print(f"\nTop impacts on {output_var}:")

        # Get series for this output
        corr_series = correlation_matrix[output_var].sort_values(ascending=False)

        # Print top positive correlations
        print(f"\nTop {top_n} positive correlations:")
        for idx, value in corr_series.head(top_n).items():
            print(f"  {idx}: {value:.3f}")

        # Print top negative correlations
        print(f"\nTop {top_n} negative correlations:")
        for idx, value in corr_series.tail(top_n).items():
            print(f"  {idx}: {value:.3f}")


def plot_key_numeric_relationships(results_df, input_vars, output_vars):
    """Create scatter plots for key numeric relationships.

    Parameters:
    -----------
    results_df : pd.DataFrame
        Complete results DataFrame
    input_vars : list
        List of input variable names
    output_vars : list
        List of output variable names
    """
    # For each numeric input, plot its relationship with outputs
    for input_var in input_vars:
        fig, axes = plt.subplots(1, len(output_vars), figsize=(14, 6), sharex=True)
        if len(output_vars) == 1:
            axes = [axes]  # Make it iterable for the loop below

        for i, output_var in enumerate(output_vars):
            sns.scatterplot(
                x=input_var,
                y=output_var,
                hue="scarcity",
                palette=COLOR_SCHEMES["scarcity"],
                data=results_df,
                alpha=0.7,
                s=60,
                ax=axes[i],
            )

            # Calculate and plot trendline
            z = np.polyfit(results_df[input_var], results_df[output_var], 1)
            p = np.poly1d(z)
            x_range = np.linspace(
                results_df[input_var].min(), results_df[input_var].max(), 100
            )
            axes[i].plot(x_range, p(x_range), "--", color="black", linewidth=1)

            # Calculate correlation coefficient
            corr = results_df[[input_var, output_var]].corr().iloc[0, 1]
            axes[i].set_title(
                f"""{output_var.replace('_', ' ').title()} 
                vs {input_var.replace('_', ' ').title()}\nCorrelation: {corr:.3f}""",
                fontsize=12,
            )

            # Add regression equation to the plot
            equation = f"y = {z[0]:.3f}x + {z[1]:.3f}"
            axes[i].annotate(
                equation,
                xy=(0.05, 0.95),
                xycoords="axes fraction",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )

        fig.suptitle(
            f"Impact of {input_var.replace('_', ' ').title()} on Simulation Outcomes",
            fontsize=16,
            y=1.05,
        )
        plt.tight_layout()
        plt.show()


def plot_feature_importance(input_df, output_df):
    """Calculate and visualize feature importance using a simple model.

    Parameters:
    -----------
    input_df : pd.DataFrame
        DataFrame containing input variables
    output_df : pd.DataFrame
        DataFrame containing output variables
    """
    plt.figure(figsize=(12, 10))

    # Create subplots for each output variable
    fig, axes = plt.subplots(
        len(output_df.columns), 1, figsize=(10, 5 * len(output_df.columns))
    )
    if len(output_df.columns) == 1:
        axes = [axes]

    # Scale the input features for better model performance
    scaler = StandardScaler()
    input_scaled = pd.DataFrame(
        scaler.fit_transform(input_df), columns=input_df.columns
    )

    for i, output_var in enumerate(output_df.columns):
        # Create and train a random forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(input_scaled, output_df[output_var])

        # Get feature importances
        importances = pd.Series(model.feature_importances_, index=input_df.columns)
        importances = importances.sort_values(ascending=False)

        # Plot the top importances
        top_n = min(15, len(importances))  # Show at most 15 features
        importances.head(top_n).plot(kind="barh", ax=axes[i])

        axes[i].set_title(
            f"Feature Importance for {output_var.replace('_', ' ').title()}",
            fontsize=14,
        )
        axes[i].set_xlabel("Importance Score", fontsize=12)

    plt.tight_layout()
    plt.show()


# =====================================================================
# Helper Functions for Plot Creation
# =====================================================================


def create_impact_by_category_plot(
    results_df,
    category,
    color_map,
    title,
    point_size=60,
    label_format=None,
    xlim=None,
    ylim=None,
    is_scalled=False,
):
    """Create a plot showing ecological vs economic impact by a categorical variable.

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing simulation results
    category : str
        Category column to group by (e.g., 'scarcity', 'station')
    color_map : dict
        Mapping of category values to colors
    title : str
        Plot title
    point_size : int
        Size of scatter points
    label_format : callable, optional
        Function to format category value for label
    xlim : tuple, optional
        X-axis limits for the plot
    ylim : tuple, optional
        Y-axis limits for the plot
    is_scalled : bool, optional
        If True, use scaled impact values; otherwise, use raw values

    Returns:
    --------
    None
        Displays the created plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    if is_scalled:
        ecological_impact = "ecological_impact"
        economic_impact = "economic_impact"
    else:
        ecological_impact = "raw_ecological_impact"
        economic_impact = "raw_economic_impact"

    for value, group in results_df.groupby(category):
        # Format label if function provided
        if label_format:
            label = label_format(value)
        else:
            label = f"{category.title()}: {value}"

        # Plot group
        ax.scatter(
            group[ecological_impact],
            group[economic_impact],
            c=color_map[value],
            label=label,
            alpha=0.75,
            s=point_size,
            edgecolor="white",
            linewidth=0.5,
        )

    # Set axis limits if provided
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Add common elements
    add_impact_plot_elements(ax, results_df)
    ax.set_title(title, fontsize=14)

    # Add legend
    create_styled_legend(
        ax,
        handles=ax.get_legend_handles_labels()[0],
        labels=ax.get_legend_handles_labels()[1],
        title=category.title(),
    )

    plt.tight_layout()
    plt.show()


def create_impact_by_scenario_plot(results_df, xlim=None, ylim=None, is_scalled=False):
    """Create a plot showing ecological vs economic impact by scenario.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing simulation results
    xlim : tuple, optional
        X-axis limits for the plot
    ylim : tuple, optional
        Y-axis limits for the plot
    is_scalled : bool, optional
        If True, use scaled impact values; otherwise, use raw values

    Returns:
    --------
    None
        Displays the created plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if is_scalled:
        ecological_impact = "ecological_impact"
        economic_impact = "economic_impact"
    else:
        ecological_impact = "raw_ecological_impact"
        economic_impact = "raw_economic_impact"

    # Get scenario settings
    scenario_settings = COLOR_SCHEMES["scenario"]

    # Plot each scenario
    for scenario in results_df["scenario"].unique():
        group = results_df[results_df["scenario"] == scenario]
        settings = scenario_settings[scenario]

        ax.scatter(
            group[ecological_impact],
            group[economic_impact],
            c=settings["color"],
            marker=settings["marker"],
            label=settings["name"],
            alpha=0.75,
            s=60,
            edgecolor="white",
            linewidth=0.5,
        )

    # Set axis limits if provided
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Add common elements
    add_impact_plot_elements(ax, results_df)
    ax.set_title("Impact by Actor Scenario Configuration", fontsize=14)

    # Get handles and labels for sorted legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    sorted_labels = sorted(by_label.keys())
    sorted_handles = [by_label[label] for label in sorted_labels]

    # Create legend
    create_styled_legend(
        ax,
        handles=sorted_handles,
        labels=sorted_labels,
        title="Scenario",
    )

    plt.tight_layout()
    plt.show()


def create_multidimensional_impact_plot(results_df, is_scalled=False):
    """Create a plot showing ecological vs economic impact.

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing simulation results
    is_scalled : bool, optional
        If True, use scaled impact values; otherwise, use raw values

    Returns:
    --------
    None
        Displays the created plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if is_scalled:
        ecological_impact = "ecological_impact"
        economic_impact = "economic_impact"
    else:
        ecological_impact = "raw_ecological_impact"
        economic_impact = "raw_economic_impact"

    # Get color schemes
    scenario_settings = COLOR_SCHEMES["scenario"]
    scarcity_colors = COLOR_SCHEMES["scarcity"]

    # Plot data points encoding multiple dimensions
    for scenario in results_df["scenario"].unique():
        for scarcity in results_df["scarcity"].unique():
            subset = results_df[
                (results_df["scenario"] == scenario)
                & (results_df["scarcity"] == scarcity)
            ]

            if not subset.empty:
                settings = scenario_settings[scenario]

                # Create a label that only shows on the first occurrence
                label = (
                    f"{settings['name']} ({scarcity})" if scarcity == "medium" else None
                )

                ax.scatter(
                    subset[ecological_impact],
                    subset[economic_impact],
                    c=scarcity_colors[scarcity],
                    marker=settings["marker"],
                    label=label,
                    alpha=0.8,
                    s=70,
                    edgecolor=settings["color"],
                    linewidth=1.5,
                )

    # Add common elements
    add_impact_plot_elements(ax, results_df)
    ax.set_title(
        "Combined Analysis: Scenario, Basin Size and Scarcity Level", fontsize=14
    )

    # Create custom legend elements

    # Marker legends (scenario variants)
    scenario_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            label="Base Scenarios",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            label="V-Variant",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            label="B-Variant",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            label="C-Variant",
        ),
    ]

    # Color legends (scarcity)
    scarcity_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=scarcity_colors["low"],
            markersize=10,
            label="Low Scarcity",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=scarcity_colors["medium"],
            markersize=10,
            label="Medium Scarcity",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=scarcity_colors["high"],
            markersize=10,
            label="High Scarcity",
        ),
    ]

    # Edge color legends (scenario type)
    type_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="w",
            markersize=10,
            markeredgecolor=scenario_settings["0.yml"]["color"],
            markeredgewidth=2,
            label="Type 0",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="w",
            markersize=10,
            markeredgecolor=scenario_settings["1.yml"]["color"],
            markeredgewidth=2,
            label="Type 1",
        ),
    ]

    # Add legends at different positions
    first_legend = create_styled_legend(
        ax,
        handles=scenario_elements,
        labels=[e.get_label() for e in scenario_elements],
        title="Scenario Variant",
        position=(0.85, 0.30),
    )
    ax.add_artist(first_legend)

    second_legend = create_styled_legend(
        ax,
        handles=scarcity_elements,
        labels=[e.get_label() for e in scarcity_elements],
        title="Scarcity Level",
        position=(0.85, 0.45),
    )
    ax.add_artist(second_legend)

    third_legend = create_styled_legend(
        ax,
        handles=type_elements,
        labels=[e.get_label() for e in type_elements],
        title="Scenario Type",
        position=(0.85, 0.8),
    )
    ax.add_artist(third_legend)

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust right margin for legends
    plt.show()


def create_forecast_effect_plot(df, x, y, ax, size_var=None, abs_size=False):
    """Create a scatter plot showing the effect of forecast parameters.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing simulation results
    x : str
        Column name for x-axis
    y : str
        Column name for y-axis
    ax : matplotlib.axes.Axes
        Axes to plot on
    size_var : str, optional
        Column name to use for point sizes
    abs_size : bool
        Whether to use absolute values for sizing

    Returns:
    --------
    None
        Modifies the provided axes
    """
    if size_var:
        sizes = (50, 200)
    else:
        sizes = 80

    # Create the scatter plot
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue="scarcity",
        size=size_var,
        sizes=sizes,
        palette=COLOR_SCHEMES["scarcity"],
        ax=ax,
        legend=False,  # Don't show legend for individual plots
    )

    # Add reference line through origin
    if min(ax.get_xlim()) <= 0 <= max(ax.get_xlim()):
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    # Improve aesthetics
    ax.set_title(f'{x.title()} vs {y.replace("_", " ").title()}', fontsize=15)
    ax.set_xlabel(x.title(), fontsize=12)
    ax.set_ylabel(y.replace("_", " ").title(), fontsize=12)


def analyze_cooperation_patterns(results_df, is_scalled=False):
    """Analyze cooperation patterns across different scenarios.

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing simulation results with cooperation_percentage
        and related parameters
    is_scalled : bool, optional
        If True, use scaled impact values; otherwise, use raw values

    Returns:
    --------
    None
        Displays visualizations showing cooperation patterns
    """
    # Create main figure for analysis
    fig, axs = plt.subplots(2, 2, figsize=(18, 16))

    if is_scalled:
        ecological_impact = "ecological_impact"
        economic_impact = "economic_impact"
    else:
        ecological_impact = "raw_ecological_impact"
        economic_impact = "raw_economic_impact"

    # 1. Cooperation by scarcity level - violin plot
    sns.violinplot(
        data=results_df,
        x="scarcity",
        y="cooperation_percentage",
        palette=COLOR_SCHEMES["scarcity"],
        ax=axs[0, 0],
        inner="quartile",
        density_norm="width",
    )

    # Add individual data points
    sns.stripplot(
        data=results_df,
        x="scarcity",
        y="cooperation_percentage",
        ax=axs[0, 0],
        size=4,
        alpha=0.6,
        jitter=True,
        color="black",
        zorder=1,
    )

    # Improve appearance
    axs[0, 0].set_title("Cooperation Percentage by Scarcity Level", fontsize=15)
    axs[0, 0].set_xlabel("Scarcity Level", fontsize=12)
    axs[0, 0].set_ylabel("Cooperation Percentage", fontsize=12)
    axs[0, 0].set_ylim(0, 1.0)

    # 2. Cooperation vs Economic Impact scatter
    sns.scatterplot(
        data=results_df,
        x="cooperation_percentage",
        y=economic_impact,
        hue="scarcity",
        palette=COLOR_SCHEMES["scarcity"],
        size="bias",
        sizes=(50, 200),
        ax=axs[0, 1],
        alpha=0.7,
    )

    axs[0, 1].set_title("Cooperation vs Economic Impact", fontsize=15)
    axs[0, 1].set_xlabel("Cooperation Percentage", fontsize=12)
    axs[0, 1].set_ylabel("Economic Impact", fontsize=12)
    axs[0, 1].grid(True, alpha=0.3)

    # 3. Cooperation vs Raw Ecological Impact
    sns.scatterplot(
        data=results_df,
        x="cooperation_percentage",
        y=ecological_impact,
        hue="scarcity",
        palette=COLOR_SCHEMES["scarcity"],
        size="uncertainty",
        sizes=(50, 200),
        ax=axs[1, 0],
        alpha=0.7,
    )

    axs[1, 0].set_title("Cooperation vs Raw Ecological Impact", fontsize=15)
    axs[1, 0].set_xlabel("Cooperation Percentage", fontsize=12)
    axs[1, 0].set_ylabel("Raw Ecological Impact (# of breaches)", fontsize=12)
    axs[1, 0].grid(True, alpha=0.3)

    # 4. Cooperation by Watershed
    station_coop = (
        results_df.groupby("station")["cooperation_percentage"]
        .agg(["mean", "std"])
        .reset_index()
    )
    station_coop = station_coop.sort_values("mean", ascending=False)
    station_coop["station"] = station_coop["station"]

    # Create a list of colors for each station
    station_colors = [COLOR_SCHEMES["station"][s] for s in station_coop["station"]]

    # Create bar plot with error bars
    axs[1, 1].bar(
        station_coop["station"].astype(str),
        station_coop["mean"],
        yerr=station_coop["std"],
        alpha=0.8,
        capsize=5,
        color=station_colors,
    )

    axs[1, 1].set_title("Average Cooperation by Station", fontsize=15)
    axs[1, 1].set_xlabel("Station", fontsize=12)
    axs[1, 1].set_ylabel("Cooperation Percentage", fontsize=12)
    axs[1, 1].set_ylim(0, 1.0)
    axs[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()


def analyze_raw_impact(results_df):
    """Analyze raw (unscaled) ecological impact across different scenarios.

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing simulation results with raw_ecological_impact

    Returns:
    --------
    None
        Displays visualizations showing raw ecological impact patterns
    """
    # Create figure with 2 rows, 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # 4. Raw Economic impact vs Economic impact colored by cooperation
    scatter = axs[0].scatter(
        results_df["raw_economic_impact"],
        results_df["economic_impact"],
        c=results_df["cooperation_percentage"],
        cmap="viridis",
        s=80,
        alpha=0.7,
    )
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axs[0])
    cbar.set_label("Cooperation Percentage", fontsize=12)

    # Add labels
    axs[0].set_xlabel("Raw Economic Impact", fontsize=12)
    axs[0].set_ylabel("Economic Impact", fontsize=12)
    axs[0].set_title("Trade-off: Raw Economic vs Economic Impact", fontsize=15)
    axs[0].grid(True, alpha=0.3)

    # 5. Raw ecological impact vs ecological impact colored by cooperation
    scatter = axs[1].scatter(
        results_df["raw_ecological_impact"],
        results_df["ecological_impact"],
        c=results_df["cooperation_percentage"],
        cmap="viridis",
        s=80,
        alpha=0.7,
    )
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axs[1])
    cbar.set_label("Cooperation Percentage", fontsize=12)

    # Add labels
    axs[1].set_xlabel("Raw Ecological Impact (# of breaches)", fontsize=12)
    axs[1].set_ylabel("Economic Impact", fontsize=12)
    axs[1].set_title("Trade-off: Raw Ecological vs Ecological Impact", fontsize=15)
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_cooperation_by_forecast_params(results_df):
    """Analyze how forecast parameters (bias and uncertainty) affect cooperation levels.

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing simulation results

    Returns:
    --------
    None
        Displays visualizations showing cooperation patterns
    """
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))

    # 1. Cooperation vs Bias by uncertainty
    # Create pivot table for heatmap
    pivot_bias = results_df.pivot_table(
        index="bias",
        columns="uncertainty",
        values="cooperation_percentage",
        aggfunc="mean",
    )

    # Create heatmap
    sns.heatmap(
        pivot_bias,
        annot=True,
        cmap="viridis",
        fmt=".2f",
        ax=ax,
        cbar_kws={"label": "Cooperation Percentage"},
    )

    ax.set_title("Cooperation % by Forecast Bias and Uncertainty", fontsize=15)
    ax.set_xlabel("Uncertainty", fontsize=12)
    ax.set_ylabel("Bias", fontsize=12)

    plt.tight_layout()
    plt.show()


def comprehensive_analysis(results_df, is_scalled=False):
    """Perform a comprehensive analysis of all key metrics and their relationships.

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing simulation results
    is_scalled : bool, optional
        If True, use scaled impact values; otherwise, use raw values

    Returns:
    --------
    None
        Displays comprehensive visualizations and summary statistics
    """
    if is_scalled:
        ecological_impact = "ecological_impact"
        economic_impact = "economic_impact"
    else:
        ecological_impact = "raw_ecological_impact"
        economic_impact = "raw_economic_impact"
    # Print overall summary statistics
    print("=== Summary Statistics ===")
    print("\nOverall Metrics:")
    print(f"Average Cooperation: {results_df['cooperation_percentage'].mean():.2f}")
    print(
        f"Average Raw Ecological Impact: {results_df['raw_ecological_impact'].mean():.1f} breaches"
    )
    print(
        f"Average Scaled Ecological Impact: {results_df['ecological_impact'].mean():.3f}"
    )
    print(f"Average Economic Impact: {results_df[economic_impact].mean():.3f}")

    print("\nCorrelation Matrix:")
    corr_matrix = results_df[
        [
            "cooperation_percentage",
            "raw_ecological_impact",
            "ecological_impact",
            "economic_impact",
            "raw_economic_impact",
            "bias",
            "uncertainty",
        ]
    ].corr()
    print(corr_matrix.round(2))

    # Create summary by scarcity level
    print("\nMetrics by Scarcity Level:")
    scarcity_summary = results_df.groupby("scarcity").agg(
        {
            "cooperation_percentage": "mean",
            "raw_ecological_impact": "mean",
            "ecological_impact": "mean",
            "raw_economic_impact": "mean",
            "economic_impact": "mean",
        }
    )
    print(scarcity_summary.round(3))

    # Create figure for comprehensive visualization
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))

    # 1. 3D-like scatter plot: Ecological vs Economic vs Cooperation
    scatter = axs[0].scatter(
        results_df[ecological_impact],
        results_df[economic_impact],
        c=results_df["cooperation_percentage"],
        s=80,
        cmap="viridis",
        alpha=0.7,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axs[0])
    cbar.set_label("Cooperation Percentage", fontsize=12)

    # Add reference diagonal line
    xlim = axs[0].get_xlim()
    ylim = axs[0].get_ylim()
    axs[0].plot([xlim[0], xlim[1]], [ylim[1], ylim[0]], "k--", alpha=0.3, linewidth=1)

    # Improve appearance
    axs[0].set_title("Impact Metrics Colored by Cooperation", fontsize=15)
    axs[0].set_xlabel("Ecological Impact (lower is better)", fontsize=12)
    axs[0].set_ylabel("Economic Impact (higher is better)", fontsize=12)
    axs[0].grid(True, alpha=0.3)

    # 2. Raw Ecological Impact vs Forecast Parameters
    ax2 = axs[1]

    try:
        # Create pivot table for contour plot
        pivot_raw_eco = results_df.pivot_table(
            index="bias",
            columns="uncertainty",
            values="raw_ecological_impact",
            aggfunc="mean",
        )

        # Create contour plot with filled contours
        contour = ax2.contourf(
            pivot_raw_eco.columns,
            pivot_raw_eco.index,
            pivot_raw_eco.values,
            levels=15,
            cmap="RdYlGn_r",  # Red for high impact (bad), green for low impact (good)
        )

        # Add contour lines
        ax2.contour(
            pivot_raw_eco.columns,
            pivot_raw_eco.index,
            pivot_raw_eco.values,
            levels=15,
            colors="k",
            alpha=0.3,
            linewidths=0.5,
        )

        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax2)
        cbar.set_label("Raw Ecological Impact (# of breaches)", fontsize=12)

        # Improve appearance
        ax2.set_title("Raw Ecological Impact by Forecast Parameters", fontsize=15)
        ax2.set_xlabel("Uncertainty", fontsize=12)
        ax2.set_ylabel("Bias", fontsize=12)
    except Exception as e:
        print(f"Error in contour plot: {e}")
        ax2.text(0.5, 0.5, "Contour plot error", ha="center", va="center", fontsize=14)

    plt.tight_layout()
    plt.show()
