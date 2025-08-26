import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.abspath('..'))
from classes import Paths, MyLoggerSetup
import numpy as np
import pandas as pd
import logging

paths = Paths()


def setup_logging(log_filename: str, max_lines: int = 20):
    """
    Sets up the logging configuration with a given log filename.
    """
    logging_path = paths.get_log_path(log_filename)

    logging.basicConfig(
        filename=logging_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    MyLoggerSetup.trim_log_file(logging_path, max_lines=max_lines)
    logging.info("Logging setup complete.")
    

def load_data(path: str) -> pd.DataFrame:
    """
    Loads the Santander Customer Satisfaction dataset.
    Logs basic information.
    """
    df = pd.read_csv(path)
    logging.info(f"Loaded data from {path} with shape {df.shape}")
    return df


def initial_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning: remove constant columns and duplicates.
    Logs what was removed.
    """
    # Remove constant features
    nunique = df.nunique()
    constant_cols = nunique[nunique == 1].index.tolist()
    if constant_cols:
        logging.info(f"Removed constant columns: {constant_cols}")
        print(f"Removed constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)
    else:
        logging.info("No constant columns found.")

    # Remove duplicates
    initial_shape = df.shape
    df = df.drop_duplicates()
    n_removed = initial_shape[0] - df.shape[0]
    if n_removed > 0:
        logging.info(f"Removed {n_removed} duplicate rows.")
        print(f"Removed {n_removed} duplicate rows.")
    else:
        logging.info("No duplicate rows found.")

    return df


# Some feature values are present in train_dataframe and absent in test_dataframe and vice-versa.
def intersect_features(train_dataframe, test_dataframe):
    """    Returns the intersection of features between train and test DataFrames.
    Logs the common features found.
    """
    logging.info("Finding common features between train and test datasets.")
    print("Finding common features between train and test datasets.")
    if train_dataframe.empty or test_dataframe.empty:
        logging.warning("One of the DataFrames is empty. Returning empty DataFrame.")
        return pd.DataFrame(), pd.DataFrame()
    if not isinstance(train_dataframe, pd.DataFrame) or not isinstance(test_dataframe, pd.DataFrame):
        logging.error("Both inputs must be pandas DataFrames.")
        raise ValueError("Both inputs must be pandas DataFrames.")
    if train_dataframe.shape[1] == 0 or test_dataframe.shape[1] == 0:
        logging.warning("One of the DataFrames has no columns. Returning empty DataFrame.")
        return pd.DataFrame(), pd.DataFrame()
    common_feat = list(set(train_dataframe.columns) & set(test_dataframe.columns))
    logging.info(f"Common features found: {common_feat}")
    return train_dataframe[common_feat], test_dataframe[common_feat]


class FeaturesUtils:
    """
    Class containing utility functions for feature engineering.
    """

    @staticmethod
    def find_categorical_candidates(df_train: pd.DataFrame, threshold: float = 0.05, max_unique_values: int = 50):
        """
        Identifies potential categorical columns in the DataFrame based on unique value counts.
        Logs the number of candidates found.

        Parameters:
        - df_train: DataFrame to analyze.
        - threshold: Maximum ratio of unique values to total rows to consider a column categorical.
        - max_unique_values: Maximum number of unique values for a column to be considered categorical.

        Returns:
        - List of column names that are potential categorical candidates.
        """
        logging.info("Finding categorical candidates in the DataFrame.")

        if not isinstance(df_train, pd.DataFrame):
            logging.error("Input must be a pandas DataFrame.")
            raise ValueError("Input must be a pandas DataFrame.")

        if df_train.empty or df_train.shape[0] == 0:
            logging.warning("The DataFrame is empty or has no rows. Returning an empty list.")
            return []

        n_rows = df_train.shape[0]
        categorical_candidates = []

        # Adjust threshold dynamically
        tmp = max_unique_values / n_rows
        if tmp < threshold:
            threshold = tmp
            logging.info(
                f"Threshold adjusted to {threshold*100:.2f}% "
                f"based on max_unique_values={max_unique_values} and n_rows={n_rows}"
            )

        for col in df_train.columns:
            n_unique = df_train[col].nunique(dropna=True)
            unique_ratio = n_unique / n_rows

            if n_unique <= max_unique_values or unique_ratio < threshold:
                categorical_candidates.append(col)

        logging.info(f"Found {len(categorical_candidates)} potential categorical candidates.")
        return categorical_candidates

    @staticmethod
    def detect_outliers(column_name: str, dataframe: pd.DataFrame, method: str = "iqr", k: float = 1.5, n: float = 3):
        """
        Detect outliers in a DataFrame column using IQR or 3-sigma rule.

        Parameters:
        - column_name: str, column to analyze
        - dataframe: DataFrame
        - method: "iqr" or "3sigma"
        - k: float, multiplier for IQR rule (default=1.5)
        - n: float, multiplier for sigma rule (default=3)

        Returns:
        - DataFrame of outliers, or None if none detected
        """
        if not pd.api.types.is_numeric_dtype(dataframe[column_name]):
            logging.info("Column %s is not numeric and will be skipped.", column_name)
            return None

        series = dataframe[column_name].dropna()

        if method == "iqr":
            Q1, Q3 = series.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - k * IQR, Q3 + k * IQR
        elif method == "3sigma":
            mean, std = series.mean(), series.std()
            lower, upper = mean - n * std, mean + n * std
        else:
            raise ValueError("method must be 'iqr' or '3sigma'")

        outliers = dataframe[(series < lower) | (series > upper)]

        if not outliers.empty:
            logging.info(
                "Column {col:>30}: {count:>5} outliers detected ({percent:.2f}% of data) using {method:>10} method".format(
                    col=column_name,
                    count=len(outliers),
                    percent=100 * len(outliers) / len(dataframe),
                    method=method,
                )
            )

            print(
                "Column {col:>30}: {count:>5} outliers detected ({percent:.2f}% of data) using {method:>10} method".format(
                    col=column_name,
                    count=len(outliers),
                    percent=100 * len(outliers) / len(dataframe),
                    method=method,
                )
            )
            return outliers
        else:
            logging.info("Column %s: no outliers detected (%s method)", column_name, method)
            return None
