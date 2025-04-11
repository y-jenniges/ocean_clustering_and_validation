# Validation scripts
Scripts in this folder support evaluating internal and external validation to determine hyperparameters and a suitable clustering method. 

| File                   | Description                                                                                                                | Required data                                                                                                              | 
|------------------------|----------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| dashboard_dbscan.py    | Interactive dashboard to explore various pre-computed hyperparameter combinations for DBSCAN in geographic and UMAP spaces | labels_dbscan_iteration0.csv, internal_validation_dbscan.csv (from clustering experiments)                                 |
| dashboard_kmeans.py    | Interactive dashboard to explore various pre-computed hyperparameter combinations for KMeans in geographic and UMAP spaces | labels_kmeans.csv, internal_validation_kmeans.csv (from clustering experiments)                                            |
| dashboard_ward.py      | Interactive dashboard to explore various pre-computed hyperparameter combinations for Ward in geographic and UMAP spaces   | labels_ward.csv, internal_validation_ward.csv (from clustering experiments)                                                |
| emus.ipynb             | Explore EMU regions                                                                                                        | Longhurst_world_v4_2010.shp [1, 2]                                                                                         |
| longhurst.ipynb        | Explore Longhurst provinces                                                                                                | emu_v5_20200914.nc [3, 4]                                                                                                  |
| score_evaluation.ipynb | Explore and plot the cluster validity indexes computed for the various clustering experiments                              | internal_validation_kmeans.csv, internal_validation_ward.csv, internal_validation.dbscan.csv (from clustering experiments) |

[1] Flanders Marine Institute (2009). Longhurst Provinces. Available online at https://www.marineregions.org/. Consulted on 2025-04-11.

[2] Longhurst, A.R. (2006). Ecological Geography of the Sea. 2nd Edition. Academic Press, San Diego, 560p.

[3] https://esri.maps.arcgis.com/home/group.html?id=6c78a5125d3244f38d1bc732ef0ee743#overview (last access: 2025-04-11)

[4] Sayre, Roger G. et al. (2017). A Three-Dimensional Mapping of the Ocean Based on Environmental Data. Oceanography, 30(1), 90-103. doi:https://doi.org/10.5670/oceanog.2017.116
