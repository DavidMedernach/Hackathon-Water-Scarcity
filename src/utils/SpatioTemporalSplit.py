"""A custom cross-validator that splits the data along two dimensions."""
import numpy as np
from sklearn.model_selection import BaseCrossValidator


class SpatioTemporalSplit(BaseCrossValidator):
    """A custom cross-validator that splits the data along two dimensions.

    This class implements a spatio-temporal split for cross-validation.
      - Temporal: a cutoff date is determined so that training data includes
        only observations with dates before that cutoff.
      - Spatial: the unique stations are randomly split so that training data
        includes only a fraction (e.g. 75%) of the stations
        while the remaining stations appear in test.

    Parameters:
      n_splits: number of folds.
      date_col: column name for the observation date.
      station_col: column name for the station code.
      temporal_frac: fraction (between 0 and 1) of the timeline
      to use for training.
      The cutoff date is chosen as the quantile of the unique dates.
      spatial_frac: fraction (between 0 and 1) of stations
      to include in training.
      random_state: random seed for reproducibility.
    """

    def __init__(
        self,
        n_splits=10,
        date_col="ObsDate",
        station_col="station_code",
        temporal_frac=0.75,
        spatial_frac=0.75,
        random_state=None,
    ):
        """Initialize the SpatioTemporalSplit.

        Args:
            n_splits (int, optional): _description_. Defaults to 10.
            date_col (str, optional): _description_. Defaults to "ObsDate".
            station_col (str, optional): _description_. Defaults to "station_code".
            temporal_frac (float, optional): _description_. Defaults to 0.75.
            spatial_frac (float, optional): _description_. Defaults to 0.75.
            random_state (_type_, optional): _description_. Defaults to None.
        """
        self.n_splits = n_splits
        self.date_col = date_col
        self.station_col = station_col
        self.temporal_frac = temporal_frac
        self.spatial_frac = spatial_frac
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test sets."""
        # Ensure X contains the needed columns.
        if self.date_col not in X.columns:
            raise ValueError(f"Column '{self.date_col}' not found in X")
        if self.station_col not in X.columns:
            raise ValueError(f"Column '{self.station_col}' not found in X")

        # Sort by the date.
        X_sorted = X.sort_values(by=self.date_col)
        unique_dates = X_sorted[self.date_col].unique()
        if len(unique_dates) == 0:
            raise ValueError("No dates found in X.")

        # Determine the cutoff date based on the temporal fraction.
        cutoff_index = int(len(unique_dates) * self.temporal_frac)
        cutoff_date = unique_dates[cutoff_index]

        # Get the unique stations.
        unique_stations = X[self.station_col].unique()
        rng = np.random.RandomState(self.random_state)

        for i in range(self.n_splits):
            # Randomly shuffle and split stations.
            stations_shuffled = rng.permutation(unique_stations)
            n = int(len(unique_stations) * self.spatial_frac)
            if (n < 1) or ((len(unique_stations) - n) < 1):
                raise ValueError(
                    """The provided spatial_frac does
                                 not leave stations for both training
                                 and testing."""
                )

            train_stations = stations_shuffled[:n]
            test_stations = stations_shuffled[n:]

            # Training indices: rows with date before cutoff AND station in train_stations.
            train_idx = X[
                (X[self.date_col] < cutoff_date)
                & (X[self.station_col].isin(train_stations))
            ].index
            # Test indices: rows with date on/after cutoff AND station in test_stations.
            test_idx = X[
                (X[self.date_col] >= cutoff_date)
                & (X[self.station_col].isin(test_stations))
            ].index

            if len(train_idx) == 0 or len(test_idx) == 0:
                print(
                    f"""Warning: Fold {i} has {len(train_idx)} training rows 
                    and {len(test_idx)} test rows. Skipping this fold."""
                )
                continue

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator."""
        return self.n_splits
