# Walmart Sales Forecast

### Short description
This repository trains a Random Forest regressor to predict weekly sales for stores/departments using historical data and store/features metadata. Primary work is in the Jupyter notebook [notebook/notebook.ipynb](notebook/notebook.ipynb).

### Project structure
- data/
  - train.csv — historical training data ([data/train.csv](data/train.csv))
  - test.csv — test set to predict ([data/test.csv](data/test.csv))
  - features.csv — weekly features (weather, CPI, etc.) ([data/features.csv](data/features.csv))
  - stores.csv — store metadata (Type, Size) ([data/stores.csv](data/stores.csv))
  - processed_data/
    - Clean_train_data.csv — cleaned training data ([data/processed_data/Clean_train_data.csv](data/processed_data/Clean_train_data.csv))
    - submission.csv — sample submission ([data/processed_data/submission.csv](data/processed_data/submission.csv))
- model/
  - random_forest_model.pkl — trained model artifact ([model/random_forest_model.pkl](model/random_forest_model.pkl))
- notebook/
  - notebook.ipynb — main preprocessing, training, prediction notebook ([notebook/notebook.ipynb](notebook/notebook.ipynb))
  - Data_visualization.ipynb — exploratory plots ([notebook/Data_visualization.ipynb](notebook/Data_visualization.ipynb))

### How the notebook works (high level)
- Loads files: [data/train.csv](data/train.csv), [data/test.csv](data/test.csv), [data/features.csv](data/features.csv), [data/stores.csv](data/stores.csv).
- Merges features and stores into the train and test sets.
- Creates date features (Year, Month, Week).
- Converts boolean columns to strings and label-encodes categorical columns using combined train+test values. Encoders are stored in the notebook variable [`encoders`](notebook/notebook.ipynb).
- Drops MarkDown1..5 and removes zero Weekly_Sales entries from training.
- Defines feature set [`Features`](notebook/notebook.ipynb) and target [`target`](notebook/notebook.ipynb).
- Trains RandomForestRegressor stored in [`rf`](notebook/notebook.ipynb) and evaluates MAE / MSE / R2.
- Retrains on full training set and writes predictions to data/processed_data/submission.csv and saves model to [model/random_forest_model.pkl](model/random_forest_model.pkl).

### Quick start
1. Install dependencies:
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
2. Open and run:
   - [notebook/notebook.ipynb](notebook/notebook.ipynb)
   - Or run the preprocessing + training cells in order.

### Notes & tips
- Categorical encoding uses LabelEncoder fitted on combined train+test to avoid unseen labels at prediction time (see variable [`encoders`](notebook/notebook.ipynb)).
- MarkDown columns are removed; missing Weekly_Sales rows with zero are treated as NaN and dropped in training.
- Final submission is written to data/processed_data/submission.csv.

### Contact
- Work is organized inside [notebook/notebook.ipynb](notebook/notebook.ipynb). Inspect the notebook for variable definitions (e.g. [`train_data`](notebook/notebook.ipynb), [`test_data`](notebook/notebook.ipynb), [`Features`](notebook/notebook.ipynb), [`rf`](notebook/notebook.ipynb)) to reproduce steps.

### Data Source
    Data is from Walmart Sales Forecast on kaggle (https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast).