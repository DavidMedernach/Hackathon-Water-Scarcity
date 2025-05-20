"""Preprocessing utilities for spatial-temporal interpolation and data merging.

This module provides functions to prepare k-d trees for spatial queries,
interpolate variables using inverse distance weighting, and merge interpolated
data with water flow observations. It also includes functions to handle
geometries and region-level data.
"""

import gc
import numpy as np
import pandas as pd  # type: ignore
from scipy.spatial import cKDTree  # type: ignore
from rioxarray.exceptions import NoDataInBounds
import xarray as xr


def prepare_kdtree(lats: np.ndarray, lons: np.ndarray) -> cKDTree:
    """Prepare a k-d tree for spatial queries using latitude and longitude.

    Args:
        lats (np.ndarray): Array of latitudes.
        lons (np.ndarray): Array of longitudes.

    Returns:
        cKDTree: A k-d tree for efficient nearest neighbor search.
    """
    coords = np.array([(lat, lon) for lat in lats for lon in lons])
    return cKDTree(coords)


def inverse_distance_weighting(values: np.ndarray, distances: np.ndarray) -> float:
    """Perform inverse distance weighting to interpolate a value.

    Args:
        values (np.ndarray): Array of values to be weighted.
        distances (np.ndarray): Array of distances corresponding to the values.

    Returns:
        float: The interpolated value.
    """
    weights = 1 / (distances + 1e-6)  # Avoid division by zero
    return np.sum(weights * values) / np.sum(weights)


def interpolate_variable(
    water_flows: pd.DataFrame, kdtree: cKDTree, variable_data, k: int = 4
) -> list:
    """Interpolate a variable to match the spatial locations in water_flows using a k-d tree.

    Args:
        water_flows (pd.DataFrame): DataFrame containing station coordinates
        and observation dates.
        kdtree (cKDTree): Precomputed k-d tree for spatial queries.
        variable_data: Dataset containing the variable to interpolate.
        k (int): Number of nearest neighbors to consider. Default is 4.

    Returns:
        list: Interpolated values for each row in water_flows.
    """
    results = []
    for _, row in water_flows.iterrows():
        obs_date = row["ObsDate"]
        station_coords = (row["latitude"], row["longitude"])

        # Find k nearest neighbors
        distances, indices = kdtree.query(station_coords, k=k)

        # Retrieve values at the specified time
        var_time = variable_data.sel(
            valid_time=obs_date, method="nearest"
        ).values.flatten()
        values = var_time[indices]

        # Compute interpolated value
        interpolated_value = inverse_distance_weighting(values, distances)
        results.append(interpolated_value)
    return results


def interpolate_geometry_time_optimized(
    water_flows: pd.DataFrame,
    kdtree: cKDTree,
    variable_data,
    geodf: pd.DataFrame,
    k: int = 4,
    agg_func=np.mean,
) -> pd.DataFrame:
    """Perform optimized spatial-temporal interpolation for geometries.

    Args:
        water_flows (pd.DataFrame): DataFrame with water flow observations
            and dates.
        kdtree (cKDTree): Precomputed k-d tree for spatial queries.
        variable_data: Dataset containing the variable to interpolate.
        geodf (pd.DataFrame): GeoDataFrame with geometries to interpolate.
        k (int): Number of nearest neighbors to consider. Default is 4.
        agg_func (callable): Aggregation function (e.g., mean).
        Default is np.mean.

    Returns:
        pd.DataFrame: Interpolated values aligned with geodf and water_flows.
    """
    # Compute centroids for geometries
    centroids = np.array(
        [[geom.centroid.y, geom.centroid.x] for geom in geodf.geometry]
    )

    # Precompute k nearest neighbors for all centroids
    distances, indices = kdtree.query(centroids, k=k)
    # Interpolate for each unique observation date
    results = []
    for obs_date in water_flows["ObsDate"].unique():
        var_time = variable_data.sel(
            valid_time=obs_date, method="nearest"
        ).values.flatten()
        values = var_time[indices]
        aggregated_values = agg_func(values, axis=1)
        results.append(aggregated_values)

    # Convert results to DataFrame
    results_df = pd.DataFrame(
        np.array(results).T,
        columns=water_flows["ObsDate"].unique(),
        index=geodf.index,
    ).T
    return results_df


def prepare_and_merge_region_data_optimized(
    water_flows: pd.DataFrame,
    region_data: pd.DataFrame,
    value_name: str,
    dtype_column: str,
) -> pd.DataFrame:
    """Prepare and merge region-level data into the water_flows DataFrame.

    Args:
        water_flows (pd.DataFrame): DataFrame containing water flow data.
        region_data (pd.DataFrame): DataFrame with region-level
        interpolated values.
        value_name (str): Name of the value column to merge.
        dtype_column (str): Name of the region type column.

    Returns:
        pd.DataFrame: Updated water_flows DataFrame with merged region data.
    """
    region_data.index.name = "ObsDate"
    region_data_reset = region_data.reset_index()

    # Transform region data into long format
    region_long = region_data_reset.melt(
        id_vars="ObsDate", var_name=dtype_column, value_name=value_name
    )

    # Align data types
    region_long[dtype_column] = region_long[dtype_column].astype(
        water_flows[dtype_column].dtype
    )

    # Merge with water flow data
    return water_flows.merge(region_long, how="left", on=["ObsDate", dtype_column])


def interpolate_and_merge_optimized(
    water_flows: pd.DataFrame,
    kdtree: cKDTree,
    variable_data,
    geodf: pd.DataFrame,
    value_name: str,
    dtype_column: str,
    k: int = 4,
    agg_func=np.mean,
) -> pd.DataFrame:
    """Interpolate a variable and merge the results into the water_flows DataFrame.

    Args:
        water_flows (pd.DataFrame): DataFrame containing water flow data.
        kdtree (cKDTree): Precomputed k-d tree for spatial queries.
        variable_data: Dataset containing the variable to interpolate.
        geodf (pd.DataFrame): GeoDataFrame with geometries for interpolation.
        value_name (str): Name of the value column to add.
        dtype_column (str): Name of the geometry type column.
        k (int): Number of nearest neighbors to consider. Default is 4.
        agg_func (callable): Aggregation function (e.g., mean).
        Default is np.mean.

    Returns:
        pd.DataFrame: Updated water_flows DataFrame with interpolated data.
    """
    # Perform interpolation
    interpolated = interpolate_geometry_time_optimized(
        water_flows, kdtree, variable_data, geodf, k=k, agg_func=agg_func
    )

    # Merge interpolated data with water flow data
    water_flows = prepare_and_merge_region_data_optimized(
        water_flows, interpolated, value_name, dtype_column
    )

    # Free up memory
    del interpolated
    gc.collect()

    return water_flows


def get_altitude(lat: float, lon: float, dem: xr.DataArray) -> float:
    """Get the altitude for a given latitude and longitude using DEM.

    Args:
        lat (float): Latitude coordinate.
        lon (float): Longitude coordinate.
        dem (xr.DataArray): Digital elevation model.

    Returns:
        float: The altitude at the given coordinates.
    """
    try:
        return dem.sel(x=lon, y=lat, method="nearest").values.item()
    except NoDataInBounds:
        return np.nan
