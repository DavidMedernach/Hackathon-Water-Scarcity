"""Preprocessing utilities for spatial-temporal interpolation and data merging."""
import os
import re
import random
from math import sqrt
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .helpers import save_or_create
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


def standardize_values(
    y: np.ndarray, stations: np.ndarray, station_stats: pd.DataFrame
) -> np.ndarray:
    """Standardize values based on station-level statistics.

    Args:
        y (np.ndarray): The values to standardize.
        stations (np.ndarray): The station codes.
        station_stats (pd.DataFrame): The station-level statistics.

    Returns:
        np.ndarray: The standardized values.
    """
    out = np.empty_like(y, dtype=float)
    for s in np.unique(stations):
        idx = stations == s
        min = station_stats.loc[s, "min"]
        max = station_stats.loc[s, "max"]
        out[idx] = (y[idx] - min) * 100.0 / (max - min)
    return out


def split_dataset(
    ds: pd.DataFrame,
    p: float = 0.75,
    time: str = None,
    seed: int = 42,
    test_stations: np.ndarray = None,
):
    """Split the dataset into training and testing sets.

    Args:
        ds (pd.DataFrame): The dataset containing data.
        p (float): The proportion of stations
        for the training set.
        time (str, optional): The timestamp for temporal split.
        seed (int): Random seed for reproducibility (default is 42).
        test_stations (np.ndarray): The list of test stations
        (for the Spatio-Temporal split).

    Returns:
        tuple: X_train, y_train, X_test, y_test, train_stations, test_stations
    """
    station_code = ds["station_code"].unique()
    random.seed(seed)
    random.shuffle(station_code)

    if test_stations is not None:
        train_stations = [s for s in station_code if s not in test_stations]
    else:
        test_stations = station_code[int(len(station_code) * p):]
        train_stations = station_code[: int(len(station_code) * p)]

    test_temporal = ds[pd.to_datetime(ds.index) >= pd.to_datetime(time)]
    test_temporal = test_temporal[test_temporal["station_code"].isin(train_stations)]

    train = ds[ds["station_code"].isin(train_stations)]
    test_spatio_temporal = ds[ds["station_code"].isin(test_stations)]
    train = train[pd.to_datetime(train.index) < pd.to_datetime(time)]
    test_spatio_temporal = test_spatio_temporal[
        pd.to_datetime(test_spatio_temporal.index) >= pd.to_datetime(time)
    ]
    return train, test_spatio_temporal, test_temporal


def get_station_stats(y: np.ndarray, station_code: np.ndarray) -> pd.DataFrame:
    """Compute station-level statistics for the given data.

    Args:
        y (np.ndarray): A NumPy array of numeric measurements or target values.
        station_code (np.ndarray): A NumPy array of station codes.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds
            to a unique station code and includes statistics.
    """
    df = pd.DataFrame({"y": y, "station_code": station_code})
    station_stats = df.groupby("station_code")["y"].agg(["mean", "std", "min", "max"])
    return station_stats


def standardize_prediction_intervals(
    y_pred_intervals: np.ndarray, stations: np.ndarray, station_stats: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Standardizes the prediction interval values for a set of stations.

    Args:
        y_pred_intervals (np.ndarray): the predicted interval values.
        stations (np.ndarray): station codes.
        station_stats (pd.DataFrame): statistics for each station.

    Returns:
        Tuple[np.ndarray, np.ndarray]: standardized lower and
            upper prediction interval values.
    """
    if y_pred_intervals is None:
        return None, None

    if len(y_pred_intervals.shape) == 3:
        y_pred_lower_std = standardize_values(
            y_pred_intervals[:, 0, 0], stations, station_stats
        )
        y_pred_upper_std = standardize_values(
            y_pred_intervals[:, 1, 0], stations, station_stats
        )
    else:
        y_pred_lower_std = standardize_values(
            y_pred_intervals[:, 0], stations, station_stats
        )
        y_pred_upper_std = standardize_values(
            y_pred_intervals[:, 1], stations, station_stats
        )

    return y_pred_lower_std, y_pred_upper_std


def compute_per_station_metrics(
    y_true_std: np.ndarray,
    y_pred_std: np.ndarray,
    stations: np.ndarray,
    y_pred_lower_std: np.ndarray = None,
    y_pred_upper_std: np.ndarray = None,
) -> pd.DataFrame:
    """Compute station-level performance metrics including.

    Args:
        y_true_std (np.ndarray): Standardized ground truth.
        y_pred_std (np.ndarray): Array of predicted standardized predictions.
        stations (np.ndarray): Station codes.
        y_pred_lower_std (np.ndarray): lower prediction interval values.
        y_pred_upper_std (np.ndarray): upper prediction interval values.

    Returns:
    pd.DataFrame
        Dataframe with the following metrics:
            - station_code: Identifier for the station.
            - scaled_rmse: Scaled Root Mean Squared Error for the station.
            - scaled_mae: Scaled Mean Absolute Error for the station.
            - coverage
            - scaled_interval_size: Average size of the prediction interval
            - log_likelihood: Gaussian negative log-likelihood.
    """
    station_list = np.unique(stations)

    records = []

    has_intervals = (y_pred_lower_std is not None) and (y_pred_upper_std is not None)

    for s in station_list:
        idx = stations == s
        y_true_s = y_true_std[idx]
        y_pred_s = y_pred_std[idx]

        rmse_s = sqrt(mean_squared_error(y_true_s, y_pred_s))
        mae_s = mean_absolute_error(y_true_s, y_pred_s)

        if has_intervals:
            y_lower_s = y_pred_lower_std[idx]
            y_upper_s = y_pred_upper_std[idx]

            # Estimate sigma using the 95% confidence interval approximation
            sigma_s = (y_upper_s - y_lower_s) / 3.29

            # Compute Gaussian negative log-likelihood
            nll_s = (1 / len(y_true_s)) * np.sum(
                np.log(sigma_s) + abs((y_true_s - y_pred_s)) / abs(2 * sigma_s)
            )

            coverage_s = np.mean((y_true_s >= y_lower_s) & (y_true_s <= y_upper_s))
            interval_size_s = np.mean(y_upper_s - y_lower_s)
        else:
            sigma_s = np.std(y_true_s - y_pred_s)  # Fallback estimation
            sigma_s = max(sigma_s, 1e-6)  # Ensure non-zero, positive sigma

            nll_s = (1 / len(y_true_s)) * np.sum(
                np.log(sigma_s) + abs((y_true_s - y_pred_s)) / abs(2 * sigma_s)
            )

            coverage_s = np.nan
            interval_size_s = np.nan

        # Collect station-level metrics
        records.append(
            {
                "station_code": s,
                "scaled_rmse": rmse_s,
                "scaled_mae": mae_s,
                "coverage": coverage_s,
                "scaled_interval_size": interval_size_s,
                "log_likelihood": nll_s,
            }
        )

    return pd.DataFrame(records)


def summarize_metrics(
    metrics: pd.DataFrame, model_name: str, dataset_type: str
) -> pd.DataFrame:
    """Given a station-level metrics DataFrame, compute average and a final score.

    Args:
        metrics (pd.DataFrame): station-level metrics.
        model_name (str): The name of the model.
        dataset_type (str): The type of dataset used (e.g., "test").

    Returns:
    pd.DataFrame
        A DataFrame containing the final model-level metrics
        (scaled RMSE, log-likelihood, scaled MAE, coverage,
        scaled interval size).
    """
    rmse_final = np.nanmean(metrics["scaled_rmse"])
    mae_final = np.nanmean(metrics["scaled_mae"])
    log_likelihood = np.nanmean(metrics["log_likelihood"])

    if metrics["coverage"].count() == 0:
        coverage_final = np.nan
        interval_size_final = np.nan
    else:
        coverage_final = np.nanmean(metrics["coverage"])
        interval_size_final = np.nanmean(metrics["scaled_interval_size"])

    data = {
        "model": [model_name],
        "dataset": [dataset_type],
        "scaled_rmse": [rmse_final],
        "log_likelihood": [log_likelihood],
        "scaled_mae": [mae_final],
        "coverage": [coverage_final],
        "scaled_interval_size": [interval_size_final],
    }
    return pd.DataFrame(data)


def print_summary_table(summary_df: pd.DataFrame):
    """Print a summary of the model-level metrics using tabulate."""
    row = summary_df.iloc[0]
    table_data = [
        ["Model", row["model"]],
        ["Dataset Type", row["dataset"]],
        ["Average Scaled RMSE", row["scaled_rmse"]],
        ["Average Scaled MAE", row["scaled_mae"]],
        ["Average Coverage", row["coverage"]],
        ["Average Scaled Interval Size", row["scaled_interval_size"]],
        ["Final Score", row["final_score"]],
    ]
    print(
        "\n"
        + tabulate(table_data, headers=["Metric", "Value"], tablefmt="pretty")
        + "\n"
    )


def generate_boxplots(
    station_metrics_df: pd.DataFrame,
    column_to_display: str,
    prefix: str,
    title: str,
    save: bool = False,
    display: bool = True,
):
    """Generate and save boxplots based on the station-level metrics.

    If RMSE_mode is True, only shows a scaled RMSE boxplot.
    Otherwise, shows coverage & interval size boxplots.
    """
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    station_metrics_df.boxplot(column=column_to_display, by="model", ax=ax1)
    ax1.set_title("Per-Station Scaled Gaussian Log-Likelihood")
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Scaled GLL")
    fig1.suptitle(title)
    plt.tight_layout()
    if display:
        plt.show()

    if save:
        current_date = pd.Timestamp.now().strftime("%d-%m-%Y_%H-%M")
        path = f"../../figures/models/{prefix}_{current_date}_boxplots.png"
        save_or_create(plt, path)


def compare_models_per_station(
    y: np.ndarray,
    predictions: List[dict],
    station_code: np.ndarray,
    prefix: str = "",
    column_to_display: str = True,
    title: str = "Model Evaluation",
    save: bool = False,
    display: bool = True,
):
    """Evaluate the performance of one or multiple models at the station level.

    Args:
        y : np.ndarray
            Ground truth values for the test set.
        predictions : List[dict]
            A list of prediction dictionaries. Each dictionary must include:
            - "model": A string with the model name.
            - "dataset": Either "train" or "test".
            - "prediction": A 1D array of predicted values.
            - "prediction_interval" (optional): interval bounds.
        station_code : np.ndarray
            An array of station identifiers corresponding to each entry in y.
        prefix : str, optional
            A string prefix for saving figures.
        column_to_display : str, optional
            The column to display in the boxplots.
        title : str, optional
            The title of the boxplots.
        save : bool, optional
            If True, the boxplots are saved.
        display : bool, optional
            If True, the boxplots are displayed.
    """
    all_station_metrics = []

    for pred in predictions:
        model_name = pred["model"]

        station_stats = get_station_stats(y, station_code)

        y_pred = pred["prediction"]
        y_pred_intervals = pred.get("prediction_interval", None)

        y_true_std = standardize_values(y, station_code, station_stats)
        y_pred_std = standardize_values(y_pred, station_code, station_stats)

        y_pred_lower_std, y_pred_upper_std = standardize_prediction_intervals(
            y_pred_intervals, station_code, station_stats
        )

        station_metrics_df = compute_per_station_metrics(
            y_true_std=y_true_std,
            y_pred_std=y_pred_std,
            stations=station_code,
            y_pred_lower_std=y_pred_lower_std,
            y_pred_upper_std=y_pred_upper_std,
        )

        station_metrics_df["model"] = model_name
        all_station_metrics.append(station_metrics_df)

    station_metrics_df = pd.concat(all_station_metrics, ignore_index=True)

    generate_boxplots(
        station_metrics_df, column_to_display, prefix, title, save, display
    )


def load_models_auto(mn: str, dir: str = "../../models/") -> List[any]:
    """Auto-load the latest models for week0, week1, and week2...

    Args:
        mn (str): The base model name to search for
        (e.g., "mapie_quantile").
        dir (str): Directory where models are stored.

    Returns:
    - List of loaded models in the order [week0, week1, week2].
    """
    p = rf"^{mn}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}})_week_([0-9]).pkl$"
    pattern = re.compile(p)
    latest_paths = {}

    for fname in os.listdir(dir):
        match = pattern.match(fname)
        if match:
            date_str, week_str = match.groups()
            week_num = int(week_str)
            date_obj = datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S")
            if week_num not in latest_paths:
                latest_paths[week_num] = (date_obj, fname)
            else:
                current_latest_date, _ = latest_paths[week_num]
                if date_obj > current_latest_date:
                    latest_paths[week_num] = (date_obj, fname)

    loaded_mapie = []
    for i in [0, 1, 2, 3]:
        if i not in latest_paths:
            raise ValueError(f"No mapie_quantile model found for week{i} in {dir}.")
        model_file = latest_paths[i][1]
        full_path = os.path.join(dir, model_file)
        loaded_mapie.append(joblib.load(full_path))

    return loaded_mapie


def custom_log_likelihood(estimator, X, y_true, cv_data, station_stats, alpha=0.1):
    """Log-likelihood scoring function.

    Args:
        estimator : The fitted estimator with a .predict method.
        X : DataFrame of predictor variables.
        y_true : True target values.
        cv_data : Full DataFrame that includes extra columns
        (e.g., "station_code").
        station_stats : Station-level statistics needed for standardization.
        alpha : Significance level (default from ALPHA).

    Returns:
        nll_s : Computed log-likelihood score.
    """
    # Align y_true with X.
    y_true = pd.Series(y_true.values, index=X.index)

    # Get predictions.
    y_pred = estimator.predict(X)

    # Get quantile predictions.
    y_quantiles = estimator.predict(X, quantiles=[alpha / 2, 1 - alpha / 2])

    # Retrieve station codes from cv_data using X's indices.
    current_stations = cv_data.loc[X.index, "station_code"].to_numpy()

    # Standardize the values.
    y_true_std = standardize_values(y_true.to_numpy(), current_stations, station_stats)
    y_pred_std = standardize_values(y_pred, current_stations, station_stats)
    y_lower_std, y_upper_std = standardize_prediction_intervals(
        y_quantiles, current_stations, station_stats
    )

    # Compute sigma from the prediction interval.
    sigma_std = (y_upper_std - y_lower_std) / 3.29
    sigma_std = np.maximum(sigma_std, 1e-6)

    # Compute the negative log-likelihood.
    nll_s = (1 / len(y_true_std)) * np.sum(
        np.log(sigma_std) + np.abs(y_true_std - y_pred_std) / (2 * sigma_std)
    )

    # Optionally, print some diagnostics.
    cov = np.mean((y_true_std >= y_lower_std) & (y_true_std <= y_upper_std))
    i_size = np.mean(y_upper_std - y_lower_std)
    print(f"Fold: coverage = {cov:.3f}, interval size = {i_size:.3f}")

    return nll_s


def create_deep_model(input_shape: Tuple[int]):
    """Define a simple deep learning model for regression."""
    model = Sequential(
        [
            Input(shape=input_shape),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dense(1, activation="linear"),  # regression output
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model
