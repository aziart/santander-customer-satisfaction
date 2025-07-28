import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.abspath('..'))
from classes import Paths
import pandas as pd
import logging

paths= Paths()
logging_path = paths.get_log_path("01_eda.log")

# Logging configuration
logging.basicConfig(
    filename=logging_path, #'01_eda.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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