# Ocean clustering and validation

Code accompanying the paper ***Jenniges et al.: Systematic definition and validation of 3-dimensional physical and biogeochemical ocean clusters using unsupervised machine learning.***

**Note:** Please be aware that running the code requires enough disc space (at least 100 GB) and may take some time to run. 
On a computer with 200GB free disc space, 32GB RAM, an Intel(R) Core(TM) i7-11800H and 8 cores, 
preparation and imputation with the given configuration took ~5.5h 
and one iteration of the clustering experiments ~15.5h.

<br>

**File explanations**

| File            | Description                                                                                | Required data                                    | 
|-----------------|--------------------------------------------------------------------------------------------|--------------------------------------------------|
| main.py         | Prepare the database, grid and impute data, conduct clustering and uncertainty experiments | comfort.sqlite[1], GEBCO_2022_sub_ice_topo.nc[2] |
| config.py       | Parameter settings for database preparation, gridding and clustering                       |                                                  |
| hyp_config.py   | Final hyperparameter settings of UMAP and clustering methods                               |                                                  |
| environment.yml | Required Python libraries to run the code                                                  |                                                  |

<br>

For an **interactive dashboard visualisation** of the final clustering, refer to:
- Code: https://github.com/y-jenniges/ocean_cluster_dashboard
- Dashboard: https://ocean-cluster-dashboard.onrender.com

[1] Korablev, A., Olsen, A., Geophysical Institute, University of Bergen, Bjerknes Centre for Climate Research (2022).COMFORT Dataset [Data set]. Norstore. https://doi.org/10.11582/2022.00039

[2] GEBCO Bathymetric Compilation Group 2022(2022). The GEBCO_2022 Grid - a continuous terrain model of the global oceans and land. NERC EDS British Oceanographic Data Centre NOC. doi:10.5285/e0f0bb80-ab44-2739-e053-6c86abc0289c
