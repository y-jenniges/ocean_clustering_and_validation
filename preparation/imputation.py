import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler
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

    # Drop non-desired columns
    if drop_columns:
        df = df.drop(columns=drop_columns, axis=1)

    # Add geographic information to support imputation (ensure that column names exist in df)
    parameters = parameters + ["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME"]
    parameters = [p for p in parameters if p in df.columns]

    # Scale parameters
    scaler = MinMaxScaler().fit(df[parameters])
    scaled = scaler.transform(df[parameters])

    # Impute missing values using KNN
    imputer = KNNImputer(n_neighbors=5, weights="distance", add_indicator=True).fit(scaled)
    imputed = imputer.transform(scaled)

    # Undo scaling
    df_unscaled = pd.DataFrame(scaler.inverse_transform(imputed[:, :scaled.shape[1]]),
                               columns=parameters).reset_index(drop=True)

    # Merge columns from original df and imputed data
    df_final = df.copy()
    df_final[parameters] = df_unscaled[parameters]
    df_final[[x + "_imputed" for x in parameters]] = pd.DataFrame(imputed[:, scaled.shape[1]:],
                                                                  columns=[x + "_imputed" for x in parameters])

    # Store imputed table
    imputed_table_path = output_dir + "wide_table_knn.csv"
    df_final.to_csv(imputed_table_path, index=False)
    logging.info(f"Stored imputed wide table as {imputed_table_path}")

    return imputed_table_path
