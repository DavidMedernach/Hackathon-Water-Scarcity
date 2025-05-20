"""Preprocessing utilities for spatial-temporal interpolation and data merging."""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
import itertools
from cartopy import crs as ccrs
from cartopy import feature as cfeature


def plot_water_flows(
    train: pd.DataFrame,
    test_spatio: pd.DataFrame = None,
    test_spatio_tempo: pd.DataFrame = None,
    max_stations: int = 50,
    display: bool = True,
    save: bool = False,
) -> None:
    """Plot the water flow for training and testing datasets for each station.

    Args:
        train (pd.DataFrame): Training dataset with water flow data.
        test_spatio (pd.DataFrame, optional): Testing dataset
            with water flow data. If not provided,
            only the training data will be plotted.
        test_spatio_tempo (pd.DataFrame, optional): Testing dataset
            with water flow data. If not provided,
            only the training data will be plotted.
        max_stations (int): Maximum number of stations to plot (default: 50).
        display (bool): Whether to display the plot (default: True).
        save (bool): Whether to save the plot as a PNG file (default: False).

    Returns:
        None. Displays or saves the generated plot.
    """
    # Prepare training data and, if available, testing data
    train = train.reset_index().rename(columns={"water_flow_week1": "water_flow_train"})
    if test_spatio is not None:
        test_spatio = test_spatio.reset_index().rename(
            columns={"water_flow_week1": "water_flow_ts"}
        )
    if test_spatio_tempo is not None:
        test_spatio_tempo = test_spatio_tempo.reset_index().rename(
            columns={"water_flow_week1": "water_flow_tst"}
        )

    data = pd.concat([train, test_spatio, test_spatio_tempo], ignore_index=False)
    data["ObsDate"] = pd.to_datetime(data["ObsDate"])

    # Group data by station code and only process the first max_stations groups
    station_groups = itertools.islice(
        data.groupby("station_code", sort=False), max_stations
    )
    groups_list = list(station_groups)
    nb_fig = len(groups_list)

    # Create subplots for each station
    fig, axes = plt.subplots(nb_fig, 1, figsize=(20, 5 * nb_fig), sharex=True)
    if nb_fig == 1:
        axes = [axes]

    for ax, (station_code, station_data) in zip(axes, groups_list):
        # Plot training data
        ax.plot(
            station_data["ObsDate"],
            station_data["water_flow_train"],
            label="Training",
            color="blue",
        )
        # Plot testing data if available
        if test_spatio is not None:
            ax.plot(
                station_data["ObsDate"],
                station_data["water_flow_ts"],
                label="Temporal Split",
                color="orange",
            )
        if test_spatio_tempo is not None:
            ax.plot(
                station_data["ObsDate"],
                station_data["water_flow_tst"],
                label="Spatio-temporal Split",
                color="red",
            )
        ax.set_title(f"Water Flow for Station: {station_code}")
        ax.set_xlabel("Observation Date")
        ax.set_ylabel("Average weekly water Flow (m³/s)")
        ax.legend()
        ax.grid(True)

        # Set x-axis ticks to display every 3 months and format the dates
        ax.xaxis.set_major_locator(MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        current_date = pd.Timestamp.now().strftime("%d-%m-%Y_%H-%M")
        save_path = f"../../figures/data/{current_date}_wf_stations.png"
        fig.savefig(save_path)
        print(f"Plot saved to {save_path}")

    if display:
        plt.show()
    else:
        plt.close(fig)


def plot_hydrographic_maps(area: str, gdf_dict: dict, bbox: dict) -> None:
    """Generate 4 hydrographic maps for a given area.

    Args:
        area (str): The area to plot.
        gdf_dict (dict): Dictionary containing GeoDataFrames.
        bbox (dict): Dictionary containing bounding box for each area.
    """
    fig, axes = plt.subplots(
        1, 4, figsize=(15, 4), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    fig.suptitle("French Hydrographic Division - 4 Levels", fontsize=16)
    BBOX_FRANCE_DISPLAY = [bbox[area][0], bbox[area][2], bbox[area][1], bbox[area][3]]

    titles = [
        "Hydrographic Region (1st Order)",
        "Hydrographic Sector (2nd Order)",
        "Hydrographic Sub-Sector (3rd Order)",
        "Hydrographic Zone (4th Order)",
    ]
    colors = ["lightblue", "lightgreen", "khaki", "salmon"]
    divisions = ["region", "sector", "sub_sector", "zone"]
    for i, (key, color) in enumerate(zip(divisions, colors)):
        gdf_dict[key][area].plot(ax=axes[i], color=color, edgecolor="black")
        axes[i].set_title(titles[i])
        axes[i].axis("off")
        axes[i].set_extent(BBOX_FRANCE_DISPLAY, crs=ccrs.PlateCarree())
        axes[i].add_feature(
            cfeature.BORDERS, linestyle=":", edgecolor="black", linewidth=1, zorder=7
        )
        axes[i].add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=7)
        axes[i].add_feature(cfeature.LAKES, facecolor="lightblue", zorder=7)
        axes[i].add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_water_flow_predictions(
    ground_truth, prediction, y_pis, prefixe, save=False, display=True
):
    """Plot water flow predictions versus actual water flow for each station.

    Args:
        ground_truth (pd.DataFrame): Waterflow ground truth.
        prediction (array-like): Predicted water flow values.
        y_pis (array-like): prediction intervals.
        prefixe (str): Prefix for the filename when saving the plot.
        save (bool, optional): If True, the plot is saved
        as a PNG file in the '../../figures/models/'.
        display (bool, optional): If True, the plot is displayed.

    Returns:
        None. Saves the plot as a PNG file in the specified directory.
    """
    water_flow = ground_truth.copy()
    water_flow["predictions"] = prediction
    if len(y_pis.shape) == 3:
        water_flow["predictions_up"] = y_pis[:, 1, 0]
        water_flow["predictions_dw"] = y_pis[:, 0, 0]
    else:
        water_flow["predictions_up"] = y_pis[:, 1]
        water_flow["predictions_dw"] = y_pis[:, 0]
    water_flow = water_flow.reset_index()

    unique_names = water_flow["station_code"].unique()

    fig, axes = plt.subplots(
        len(unique_names), 1, figsize=(20, 5 * len(unique_names)), sharex=True
    )
    if len(unique_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, unique_names):
        wf_station = water_flow[water_flow["station_code"] == name]

        ax.plot(
            wf_station["ObsDate"],
            wf_station["predictions"],
            label="Predictions",
            color="red",
            linewidth=2,
        )
        ax.plot(
            wf_station["ObsDate"],
            wf_station["predictions_up"],
            label="predictions_up",
            color="orange",
            linewidth=1,
        )
        ax.plot(
            wf_station["ObsDate"],
            wf_station["predictions_dw"],
            label="predictions_dw",
            color="orange",
            linewidth=1,
        )
        ax.plot(
            wf_station["ObsDate"],
            wf_station["water_flow_week1"],
            label="Water Flow",
            color="blue",
            linewidth=2,
        )

        ax.set_title(f"Water Flow for Station: {name}", fontsize=24)
        ax.set_xlabel("Observation Date", fontsize=18)
        ax.set_ylabel("Water Flow", fontsize=18)
        ax.legend()
        ax.grid(True)
        ax.legend(fontsize=12)
        ax.tick_params(axis="x", rotation=45, labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

        ax.xaxis.set_major_locator(plt.MaxNLocator(10))

    plt.tight_layout()

    if save:
        date = pd.Timestamp.now().strftime("%d-%m-%Y_%H-%M")
        save_path = f"../../figures/models/{prefixe}_{date}_wf_predictions.png"
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Plot saved to {save_path}")
    elif display:
        plt.show()
