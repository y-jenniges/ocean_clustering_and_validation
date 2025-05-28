import pandas as pd
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer


def impute_data(csv_path, parameters, drop_columns, output_dir):
    """
    Impute missing values of specified parameters by first scaling them to [0, 1] and then imputing using KNN.

    Args:
        csv_path (str): Path to data CSV table.
        parameters (list<str>): Column names of parameters to impute.
        drop_columns (list<str>): Columns to drop from dataframe.
        output_dir (str): Directory where to save imputed table.
    Returns:
        imputed_table_path (str): Path to imputed CSV table file.
    """
    logging.info(f"Imputing data for {parameters}...")

    # Load CSV data
    df = pd.read_csv(csv_path)
    df_scaled = df.copy()

    # Drop non-desired columns
    if drop_columns:
        df_scaled = df.drop(columns=drop_columns, axis=1)

    # Add geographic information to support imputation (ensure that column names exist in df)
    spatial_cols = [c for c in ["LATITUDE", "LONGITUDE", "LEV_M"] if c in df_scaled.columns]
    time_col = [c for c in ["DATEANDTIME"] if c in df_scaled.columns]
    time_cols = []

    # Scale spatial parameters
    scaler_spatial = MinMaxScaler().fit(df_scaled[spatial_cols])
    df_scaled[spatial_cols] = scaler_spatial.transform(df_scaled[spatial_cols])

    # Scale parameters
    scaler_params = RobustScaler().fit(df_scaled[parameters])
    df_scaled[parameters] = scaler_params.transform(df_scaled[parameters])

    # Scale time (create cyclical time features)
    if time_col:
        logging.ERROR("Time support not entirely implemented yet!")
        # Ensure correct format
        df_scaled["DATEANDTIME"] = pd.to_datetime(df_scaled["DATEANDTIME"])

        # Extract components
        df_scaled["year"] = df_scaled["DATEANDTIME"].dt.year
        df_scaled["month"] = df_scaled["DATEANDTIME"].dt.month
        df_scaled["day"] = df_scaled["DATEANDTIME"].dt.day
        df_scaled["hour"] = df_scaled["DATEANDTIME"].dt.hour
        df_scaled["minute"] = df_scaled["DATEANDTIME"].dt.minute
        df_scaled["second"] = df_scaled["DATEANDTIME"].dt.second

        # Cyclic encoding
        df_scaled["month_sin"] = np.sin(2 * np.pi * df_scaled["month"] / 12)
        df_scaled["month_cos"] = np.cos(2 * np.pi * df_scaled["month"] / 12)
        df_scaled["day_sin"] = np.sin(2 * np.pi * df_scaled["day"] / 31)
        df_scaled["day_cos"] = np.cos(2 * np.pi * df_scaled["day"] / 31)
        df_scaled["hour_sin"] = np.sin(2 * np.pi * df_scaled["hour"] / 24)
        df_scaled["hour_cos"] = np.cos(2 * np.pi * df_scaled["hour"] / 24)
        df_scaled["minute_sin"] = np.sin(2 * np.pi * df_scaled["minute"] / 60)
        df_scaled["minute_cos"] = np.cos(2 * np.pi * df_scaled["minute"] / 60)
        df_scaled["second_sin"] = np.sin(2 * np.pi * df_scaled["second"] / 60)
        df_scaled["second_cos"] = np.cos(2 * np.pi * df_scaled["second"] / 60)

        # Scale the linear "year" column
        df_scaled["year_scaled"] = MinMaxScaler.fit_transform(df_scaled[["year"]])

        time_cols = ["year_scaled", "month_sin", "month_cos", "day_sin", "day_cos", "hour_sin", "hour_cos",
                     "minute_sin", "minute_cos", "second_sin", "second_cos"]

        # Drop raw components
        df_scaled.drop(columns=["year", "month", "day", "hour", "minute", "second", "DATEANDTIME"], inplace=True)

    # Impute missing values using KNN
    imputed = KNNImputer(n_neighbors=5, weights="distance", add_indicator=True).fit_transform(
        df_scaled[spatial_cols + parameters + time_cols])

    # Undo scaling
    df_unscaled = df_scaled.copy()
    df_unscaled[spatial_cols] = pd.DataFrame(scaler_spatial.inverse_transform(imputed[:, :len(spatial_cols)]),
                                             columns=spatial_cols).reset_index(drop=True)
    df_unscaled[parameters] = pd.DataFrame(scaler_params.inverse_transform(
        imputed[:, len(spatial_cols): len(spatial_cols) + len(parameters)]),
        columns=parameters).reset_index(drop=True)

    # Undo time scaling

    # Add imputation information and columns from original df
    df_unscaled["imputed"] = np.round(imputed[:, len(spatial_cols) + len(parameters):
                                                 len(spatial_cols) + len(parameters) * 2].sum(axis=1)
                                      / len(parameters) * 100, 2)
    df_unscaled[[x for x in df.columns if x not in df_unscaled.columns]] = df[
        [x for x in df.columns if x not in df_unscaled.columns]]

    # Print how many grid cells were partly/entirely imputed
    all_imputed_cells = np.round(len(df_unscaled[df_unscaled.imputed == 100]) / len(df_unscaled) * 100, 1)
    partly_imputed_cells = np.round(
        len(df_unscaled[(df_unscaled.imputed < 100) & (df_unscaled.imputed > 0)]) / len(df_unscaled) * 100, 1)
    total_missingness = np.round(df_unscaled.imputed.sum() / len(df_unscaled), 1)
    logging.info(f"Proportion of grid cells that were entirely imputed: {all_imputed_cells}%")
    logging.info(f"Proportion of grid cells that were partly imputed: {partly_imputed_cells}%")
    logging.info(f"Proportion total missingness: {total_missingness}%")

    # Store imputed table
    imputed_table_path = output_dir + "wide_table_knn.csv"
    df_unscaled.to_csv(imputed_table_path, index=False)
    logging.info(f"Stored imputed wide table as {imputed_table_path}")

    return imputed_table_path
