import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def impute_categorical_mode(data: pd.DataFrame, categorical_columns: list):
    """
    Impute missing values in categorical features based on mode (most frequent value).

    Parameters:
        data (pd.DataFrame): Input DataFrame.
        categorical_columns (list): List of column names that are categorical and need imputation.

    Returns:
        pd.DataFrame: DataFrame with imputed values for the categorical features.
    """
    data_imputed = data.copy()

    for col in categorical_columns:
        # Calculate the mode of the column, excluding missing values
        mode_value = data_imputed[col].mode().iloc[0]

        # Fill missing values with the mode
        data_imputed[col].fillna(mode_value, inplace=True)

    return data_imputed


def normalize_continuous_features(data: pd.DataFrame, continuous_columns):
    """
    This function normalizes continuous (numeric) columns in a pandas DataFrame
    using Min-Max scaling. The Min-Max scaling transforms the data such that
    each value is scaled to the range [0, 1].

    Parameters:
    data (pd.DataFrame): The pandas DataFrame containing the data to be normalized.
    continuous_columns (list): A list of column names (as strings) representing
                                the continuous (numeric) features to be normalized.

    Returns:
    None: The function modifies the original DataFrame by replacing the values in
          the specified columns with the normalized values.
    """

    # Create a MinMaxScaler to scale the continuous features
    scaler = MinMaxScaler()

    # Apply the MinMaxScaler to the specified columns in the DataFrame
    # The values in these columns are scaled to the range [0, 1]
    data[continuous_columns] = scaler.fit_transform(data[continuous_columns])
