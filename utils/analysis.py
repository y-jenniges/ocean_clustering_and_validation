from scipy.spatial.transform import Rotation as R
import pandas as pd
from pathlib import Path
from visualisation.plotting import plot_embedding
import hyp_config
import config


# *********************************************************************************************************************#
""" Functions to work with UMAP space. """
# *********************************************************************************************************************#


def rotate_umap(u, x, y, z, factor=1):
    """ Rotating the embedding around x, y and z axis. """
    rx = R.from_euler('x', x, degrees=True)
    ry = R.from_euler('y', y, degrees=True)
    rz = R.from_euler('z', z, degrees=True)
    factor = float(factor)
    u_r = rx.apply(ry.apply(rz.apply(factor * u)))
    return u_r


def get_user_input(u):
    """ Ask user whether to rotate the embedding. """
    plot_embedding(u, figsize=(4, 4))
    user_input = input("Rotate? y/n")
    if user_input == "y":
        print("  y")
        rotx = input("x rotation: ")
        roty = input("y rotation: ")
        rotz = input("z rotation: ")
        factor = int(input("factor: "))
        umap_temp = rotate_umap(u, rotx, roty, rotz, factor=factor)
        return get_user_input(umap_temp)
    elif user_input == "n":
        print("  n")
        return u
    else:
        print("  repeating...")
        get_user_input(u)


# *********************************************************************************************************************#
""" Functions to work with the labels dataframes. """
# *********************************************************************************************************************#


def prepare_labels_df(labels_df, iteration=0):
    temp = labels_df.copy()

    # Filter for labels from the given iteration
    if "iteration" in temp.columns:
        temp = temp[temp["iteration"] == iteration]
        temp = temp.drop("iteration", axis=1)

    # If a column is entirely NaN, remove it
    temp = temp.dropna(axis=1, how="all")

    # Make the labels of the 2 preprocessing methods different columns
    umap_cols = [f"e{i}" for i in range(hyp_config.umap_hyps["n_components"])]
    labels_pivoted = temp.drop(columns=umap_cols).pivot(
        index=[x for x in temp if x not in ["preprocessing", "label"] + umap_cols],
        columns="preprocessing",
        values="label"
    ).reset_index()

    # Merge back UMAP columns
    df_final = pd.merge(
        labels_pivoted,
        temp[[x for x in temp if x not in ["preprocessing", "label"]]].dropna(),
        on=[x for x in temp if x not in ["preprocessing", "label"] + umap_cols],
        how="left"
    )

    return df_final


def load_dbscan_labels(iteration):
    """ Load labels of one iteration from DBSCAN. Only for iteration 0, a dataframe was stored since storing all DBSCAN
    labels in one file results in a too large file (~24GB) due to added columns. """
    if iteration == 0:
        return pd.read_csv(f"{config.output_dir_clustering}labels_dbscan_iteration0.csv")
    else:
        # All other labels have not been concatenated, so we do it on the fly here
        dfs = []
        for file in Path(config.output_dir_clustering).glob(f"labels_iteration{iteration}_*_dbscan_*.csv"):
            # Load labels
            t = pd.read_csv(file)

            # Add parameters to temporary dataframe
            t["iteration"] = iteration
            t["preprocessing"] = str(file).split(f"iteration{iteration}_")[1].split("_dbscan")[0]

            # Store each hyperparameter in a separate column
            for hyp in config.algorithms_and_hyps["dbscan"][1].keys():
                t[hyp] = str(file).split(hyp)[1].split("_")[0].rstrip(".csv")

            dfs.append(t)
        return pd.concat(dfs, ignore_index=True, axis=0)


# *********************************************************************************************************************#
""" Functions to work with the internal validation/scores dataframes. """
# *********************************************************************************************************************#


def get_query_string_hyps_filter(hyp_dict, preproc=None, iteration=None, algorithm=None, float_tol=1e-8):
    """ Helper function to assemble a query for a dataframe to filter for hyperparameters from a given dict. """
    query_parts = []
    for k, v in hyp_dict.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            query_parts.append(f"{k}.isnull()")  # Handle NaN values
        elif isinstance(v, float):
            query_parts.append(f"({k} >= {v - float_tol} & {k} <= {v + float_tol})")  # Handle float precision
        else:
            query_parts.append(f"{k} == {repr(v)}")  # Handle integers & strings

    if preproc:  # Add preprocessing condition only if provided
        query_parts.append(f"preprocessing == {repr(preproc)}")

    if iteration:  # Add iteration condition only if provided
        query_parts.append(f"iteration == {repr(iteration)}")

    if algorithm:  # Add algorithm condition only if provided
        query_parts.append(f"clustering == {repr(algorithm)}")

    return " & ".join(query_parts)
