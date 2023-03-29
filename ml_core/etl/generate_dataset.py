"""
This script loads the Vertical Mill data, performs a dataframe preprocessing pipeline (removes the date and time column
and inputes missing values with 0 or the mean value) and saves the resulting dataframe.
Any other further transformations should be added here.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


def generate(raw_data_loc: str, transformed_data_loc: str) -> None:
    """
    Load Mill data, remove datetime column and stores in disk.

    Parameters:
        raw_data_loc (str): the location of the raw data

        transformed_data_loc (str): the location of the generated transformed data to save
    """
    logger.info(f"Loading dataset from: {raw_data_loc}")

    df = pd.read_csv(raw_data_loc)

    # preprocess the dataframe
    df = process_dataframe_pipeline(df)

    # TODO: It should be nice to give some overview about the generated dataset here... but let's keep the focus.

    # save data frame into disk
    df.to_csv(transformed_data_loc, index=False)
    logger.info("Training data has been saved into disk!")


def process_dataframe_pipeline(df: pd.DataFrame, fillna="mean") -> pd.DataFrame:
    """
    Performs all dataframe-related preprocessing

    Parameters:
        df (pd.DataFrame): data frame from VRM

    Returns:
        df_processed (pd.DataFrame): data frame preprocessed
    """
    df_processed = df.copy(deep=True)

    # 1 - Drop the first column containing datetime
    df_processed = df_processed.iloc[:, 1:]

    # 2 - Fill empty values with the mean (for the Isolation Forest it could be 0 since it is a tree-based algorithm)
    # Since there are no outliers detected in the columns with missing values, we can use the mean value to fill
    # the missing values
    df_processed = df_processed.fillna(df_processed.mean())

    return df_processed
