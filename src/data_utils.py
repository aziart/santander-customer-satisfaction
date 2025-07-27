
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Loads the Santander Customer Satisfaction dataset.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    df = pd.read_csv(path)
    return df


def initial_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning: remove constant columns and duplicates.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    # Remove constant features
    nunique = df.nunique()
    constant_cols = nunique[nunique == 1].index.tolist()
    df = df.drop(columns=constant_cols)

    # Remove duplicates
    df = df.drop_duplicates()

    return df
