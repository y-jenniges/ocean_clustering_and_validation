# Uncertainty scripts
Scripts in this folder analyse clustering uncertainties and perform NEMI [1]. Here, an adapted version of NEMI code [2] was used.

| File                            | Description                                                                            | Required data                                    | 
|---------------------------------|----------------------------------------------------------------------------------------|--------------------------------------------------|
| main_uncertainty_experiments.py | Main functions to compute uncertainty experiments                                      | wide_table_knn.csv (from data preparation)       |
| uncertainty_metrics.py          | Helper functions to compute uncertainty metrics like overlap                           |                                                  |
| uncertainty_nemi.py             | Helper functions to compute NEMI cluster sets and uncertainties                        |                                                  |
| uncertainty_nemi_analysis.ipynb | Script to analyse NEMI runs and uncertainties and determine a final cluster set        |                                                  |
| uncertainty_umap.ipynb          | Compute multiple embeddings using UMAP (fixed hyperparameters) and analyse variability | umap_dbscan_*.csv (from uncertainty experiments) |

[1] Sonnewald, M., in review. A hierarchical ensemble manifold methodology for new knowledge on spatial data: An application to ocean physics. Journal of Advances in Modeling Earth Systems. Available: ESSOAr.

[2] https://github.com/maikejulie/NEMI (last access: 2025-04-11)
