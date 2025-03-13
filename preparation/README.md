# Data preparation scripts
Scripts in this folder prepare data from the COMFORT database from raw measurements to a fully specified grid product. 

| File                   | Description                                                                           | Required data                      | 
|------------------------|---------------------------------------------------------------------------------------|------------------------------------|
| data.ipynb             | Description of the final, gridded and imputed data set                                | wide_table_knn.csv from main.py    |
| gridding.py            | Functions and classes to grid data                                                    |                                    |
| imputation.py          | Function to apply KNN imputation to fill in missing values in a grid                  |                                    |
| preparation.py         | Prepares the database, e.g. by unifying units and averaging over equal space and time |                                    |
| units.py               | Functions and classes to convert units                                                |                                    |
| ---------------        | ------------------------------------------------------------------------------------- | ---------------------------------- |