import math
from tqdm import tqdm
import numpy as np
from pathlib import Path

import config
from visualisation.plotting import color_code_labels


def compute_volume(row, dlat, dlon, depths):
    """ Computes the volume of a grid cell, which is specified as a row from a pandas dataframe. """
    lat = row["LATITUDE"]
    depth = row["LEV_M"]

    # Find depth step size
    idx = np.argwhere(depths == depth)
    next_depth = depths[idx + 1]
    ddepth = (next_depth - depth).flatten()[0]

    R = 6378137.0  # Earth radius in meters (WGS84 standard)

    lat_length = 111320  # Average meters per degree of latitude
    lon_length = (math.pi / 180) * R * math.cos(
        math.radians(lat))  # Length of 1 degree of longitude in meters at given latitude

    volume = dlat * lat_length * dlon * lon_length * ddepth
    return volume


def compute_volumetric_nemi(nemi_pack, base_id: int = 0, max_clusters=None):
    """ From an ensemble of cluster sets, compute a final labelling and quantify uncertainty. Adapted from
    https://github.com/maikejulie/NEMI.

    Args:
        nemi_pack (list<list<int, pandas.Dataframe>>): List containing cluster iteration number and the cluster set.
        base_id (int, optional): Index (starting at 0) of ensemble member to use as the base comparison.
        max_clusters (int): Number of cluster sets to compare the base cluster set against.
    """
    base_id = base_id

    # List of ensemble members we are comparing to the base
    compare_ids = [i for i in range(len(nemi_pack))]
    compare_ids.pop(base_id)

    # Identify clusters from the base ensemble member
    base_labels = [x for i, x in nemi_pack if int(i) == int(base_id)][0].sorted_label  # .label.unique())
    base_volumes = [x for i, x in nemi_pack if int(i) == int(base_id)][0].volume

    # Number of clusters
    num_clusters = int(np.max(base_labels) + 1)

    # If not pre-set, set max number of clusters to total number of clusters in the base
    if max_clusters is None:
        max_clusters = num_clusters

    sortedOverlap = np.zeros((len(compare_ids) + 1, max_clusters, base_labels.shape[0])) * np.nan

    # print(num_clusters, max_clusters)
    summaryStats = np.zeros((num_clusters, max_clusters))

    # Compile sorted cluster data
    # TODO: add assert statement to make sure that the clusters have been sorted?
    dataVector = [(nemi[1].sorted_label, nemi[1].volume) for id, nemi in enumerate(nemi_pack) if id != base_id]

    # Loop over ensemble members, not including the base member
    for compare_cnt, compare_id in tqdm(enumerate(compare_ids)):
        # Grab clusters of ensemble member
        compare_labels, compare_volumes = dataVector[compare_cnt]

        # Go through each cluster in the base and assess the percentage overlap
        # For every cluster in the ensemble member (overlap / total coverage area)
        for c1 in range(max_clusters):
            # Initialize dummy array to mark location of the cluster for the base member
            data1_M = np.zeros(base_labels.shape, dtype=int)
            # Mark where the considered cluster is in the member that is being used as the baseline
            data1_M[np.where(c1 == base_labels)] = 1
            # Count numer of entries [Why?]
            summaryStats[0, c1] = np.sum(data1_M)
            # Mark volumes of the base cluster
            data1_volumes = data1_M * base_volumes

            # Go through each cluster
            # k = 0
            for c2 in range(num_clusters):
                # Initialize dummy array to mark where the cluster is in the comparison member
                data2_M = np.zeros(base_labels.shape, dtype=int)

                # Mark where the considered cluster is in the member that is being used as the comparison
                data2_M[np.where(c2 == compare_labels)] = 1

                # Mark volumes of the comparison cluster
                data2_volumes = data2_M * compare_volumes

                # Sum of flags where the two datasets of that cluster are both present
                shared_cells = data1_M * data2_M
                volume_overlap = np.sum((data1_volumes + data2_volumes) * shared_cells)

                # Sum of where they overlap
                volume_total = np.sum(data1_volumes + data2_volumes)

                # Collect the number that is largest of k and the num_overlap/num_total
                # k = max(k, num_overlap / num_total)
                summaryStats[c2, c1] = (volume_overlap / volume_total) * 100  # volumetric percentage of coverage

            # Filled in 'summaryStatistics' matrix results of percentage overlaps

        usedClusters = set()  # Used to mak sure clusters don't get selected twice
        # Clusters are already sorted by size

        sortedOverlapForOneCluster = np.zeros(base_labels.shape, dtype=int) * np.nan
        # Go through clusters from (biggest to smallest since they are sorted)
        for c1 in range(max_clusters):
            sortedOverlapForOneCluster = np.zeros(base_labels.shape, dtype=int) * np.nan
            # print('cluster number ', c1, summaryStats.shape, summaryStats[1:,c1-1].shape)

            # Find biggest cluster in first column, making sure it has not been used
            sortedClusters = np.argsort(summaryStats[:, c1])[::-1]
            biggestCluster = [ele for ele in sortedClusters if ele not in usedClusters][0]

            # Record it for later
            usedClusters.add(biggestCluster)

            # Initialize dummy array
            data2_M = np.zeros(base_labels.shape, dtype=int)

            # Select which country is being assessed
            data2_M[np.where(biggestCluster == compare_labels)] = 1  # Select cluster being assessed

            sortedOverlapForOneCluster[np.where(data2_M == 1)] = 1
            sortedOverlap[compare_id, c1, :] = sortedOverlapForOneCluster

    # Fill in the base entry in the sorted overlap
    for c1 in range(max_clusters):
        sortedOverlap[base_id, c1, :] = 1 * (base_labels == c1)

    # Majority vote
    aggOverlaps = np.nansum(sortedOverlap, axis=0)
    voteOverlaps = np.argmax(aggOverlaps, axis=0)

    # Save clusters estimated from the ensemble
    clusters = voteOverlaps

    # Compute how uncertain the prediction is
    uncertainty = 1 - np.max(aggOverlaps, axis=0)

    return clusters, uncertainty


# @delayed
def compute_and_store_nemi_cluster_set(df, pack, base_id, prefix="volume_"):
    """
    Compute volumetric NEMI for a given ensemble of cluster sets and store output as CSV file.

    Args:
        df (pandas.DataFrame): Used to store all information form the original dataframe along with the new cluster
        labels.
        pack (list<list<int, pandas.Dataframe>>): List containing cluster iteration number and the cluster set.
        base_id (int): Base Id used for the NEMI method.
        prefix (str): Prefix used for filenames of the new cluster sets.
    """
    filename = f"{config.output_dir_uncertainty}{prefix}nemi_iteration{base_id}_uncertainty.csv"

    # Only do computations if file does not yet exist
    if Path(filename).exists():
        # Compute NEMI labels and uncertainty
        final_labels, uncertainty = compute_volumetric_nemi(nemi_pack=pack, base_id=base_id)

        # Store
        df["final_label"] = final_labels
        df["uncertainty"] = uncertainty * 100
        df = color_code_labels(df, column_name="final_label").rename({"color": "label_color"}, axis=1)
        df.to_csv(filename, index=False)
