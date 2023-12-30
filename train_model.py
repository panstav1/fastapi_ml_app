# Script to train machine learning model.
# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split
from ml.data import process_data, save_encoders, load_encoders
from ml.model import train_model, save_model, inference, compute_model_metrics

import pandas as pd
import os

# Variable set
# TODO: replace them with config
data_folder = 'data'  # Data subfolder
data_file = 'census.csv'  # Data filename
model_path = './model/model_cat.pickle'  # Model filename
slices_perf_filename = 'slice_output.txt'  # Text file to keep performance evaluation on each slice
label = 'salary'  # Label of dataset

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

# Keep hashmap of results for evaluation of each slice on categorical features
slices_perf_dict = {}

# Add code to load in the data.
input_dataset = pd.read_csv(os.path.join(data_folder, data_file ))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(input_dataset, test_size=0.20)



# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label=label, training=True
)

# Train and save a model.
model = train_model(X_train, y_train)

# Save model
save_model(model, model_path)

save_encoders(encoder, lb, './model/encoder.joblib', './model/binarizer.joblib')




# Slice evaluation
for cat_feat in cat_features:
    for cur_feat_unique in test[cat_feat].unique():
        cur_sample = test[test[cat_feat] == cur_feat_unique]

        # Proces the test data with the process_data function.
        X_test, y_test, encoder, lb = process_data (
            cur_sample, categorical_features=cat_features, label=label, training=False, encoder=encoder, lb=lb
        )
        y_pred = inference(model, X_test)
        cur_prec, cur_rec, cur_fbeta = compute_model_metrics(y_test, y_pred)

    slices_perf_dict[cat_feat] = {
                                      'Precision': cur_prec,
                                      'Recall': cur_rec,
                                      'F1_Beta': cur_fbeta
                                      }

# Writing dictionary to a text file
with open (slices_perf_filename, 'w') as file:
    for key, value in slices_perf_dict.items():
        file.write (f'{key}: {value}\n')



