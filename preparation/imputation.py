import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer


def impute_data(csv_path, parameters, drop_columns, output_dir):
    """ Impute missing values of specified parameters by first scaling them to [0, 1] and then imputing using KNN.
    :param csv_path: Path to data CSV table
    :param parameters: Column names of parameters to impute
    :param drop_columns: Columns to drop from dataframe
    :param output_dir: Directory where to save imputed table
    :return: Path to imputed table
    """
    logging.info(f"Imputing data for {parameters}...")
    # Load CSV data
    df = pd.read_csv(csv_path)

    # Drop desired columns
    if drop_columns:
        df = df.drop(columns=drop_columns, axis=1)

    # Scale parameters
    scaler = MinMaxScaler().fit(df[parameters])
    scaled = scaler.transform(df[parameters])

    # Impute missing values using KNN
    imputer = KNNImputer(n_neighbors=5, weights="distance").fit(scaled)
    imputed = imputer.transform(scaled)

    # Undo scaling
    df_unscaled = pd.DataFrame(scaler.inverse_transform(imputed), columns=df[parameters].columns).reset_index(drop=True)

    # Merge columns from original df and imputed data
    df_unscaled = pd.concat([df[[x for x in df.columns if x not in parameters]].reset_index(drop=True), df_unscaled],
                            axis=1)

    # Store imputed table
    imputed_table_path = output_dir + "wide_table_knn.csv"
    df_unscaled.to_csv(imputed_table_path, index=False)
    logging.info(f"Stored imputed wide table as {imputed_table_path}")

    return imputed_table_path
