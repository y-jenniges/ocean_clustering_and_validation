# Scripts for clustering experiments and feature importances
Scripts in this folder prepare data from the COMFORT database from raw measurements to a fully specified grid product. 

| File                      | Description                                                                           | Required data                                                      | 
|---------------------------|---------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| clustering_experiments.py | Description of the final, gridded and imputed data set                                |                                                                    |
| feature_importance.py     | Functions and classes to grid data                                                    | labels_kmeans.csv, labels_ward.csv, labels_dbscan.csv from main.py |
| ---------------           | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |