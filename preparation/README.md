# Data preparation scripts
Scripts in this folder prepare data from the COMFORT database from raw measurements to a fully specified grid product. 

| File                   | Description                                                                                                    | Required data  | 
|------------------------|----------------------------------------------------------------------------------------------------------------|----------------|
| gridding.py            | Helper functions and classes to grid data                                                                      |                |
| imputation.py          | Helper function to apply KNN imputation to fill in missing values in a grid                                    |                |
| preparation.py         | Main functions to prepare the COMFORT database, e.g. by unifying units and averaging over equal space and time | COMFORT.sqlite |
| units.py               | Helper functions and classes to convert units                                                                  |                |
