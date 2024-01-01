# TODO: replace them with config
data_folder = 'data'  # Data subfolder
data_file = 'census.csv'  # Data filename
slices_perf_filename = 'slice_output.txt'  # Text file to keep performance evaluation on each slice
label = 'salary'  # Label of dataset

model_folder = 'model'
model_encoder = 'encoder.joblib'
label_bin = 'binarizer.joblib'
model_file = 'model_cat.pickle'  # Model filename


# Categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]