# Script to train machine learning model.
from sklearn.model_selection import train_test_split
from ml.data import process_data, save_encoders, load_encoders
from ml.model import train_model, save_model, inference, compute_model_metrics
import constants
import pandas as pd
import os


# Keep hashmap of results for evaluation of each slice on categorical features
slices_perf_dict = {}

# Add code to load in the data.
input_dataset = pd.read_csv(os.path.join(constants.data_folder, constants.data_file))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(input_dataset, test_size=0.20)



# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=constants.cat_features, label=constants.label, training=True
)

# Train and save a model.
model = train_model(X_train, y_train)

# Save model
save_model(model, os.path.join(constants.model_folder, constants.model_file))

save_encoders(encoder, lb, os.path.join(constants.model_folder, constants.model_encoder),
              os.path.join(constants.model_folder, constants.label_bin))


# Slice evaluation
for cat_feat in constants.cat_features:
    for cur_feat_unique in test[cat_feat].unique():
        cur_sample = test[test[cat_feat] == cur_feat_unique]

        # Proces the test data with the process_data function.
        X_test, y_test, encoder, lb = process_data (
            cur_sample, categorical_features=constants.cat_features, label=constants.label, training=False, encoder=encoder, lb=lb
        )
        y_pred = inference(model, X_test)
        cur_prec, cur_rec, cur_fbeta = compute_model_metrics(y_test, y_pred)

    slices_perf_dict[cat_feat] = {
                                      'Precision': cur_prec,
                                      'Recall': cur_rec,
                                      'F1_Beta': cur_fbeta
                                      }

# Writing dictionary to a text file
with open (constants.slices_perf_filename, 'w') as file:
    for key, value in slices_perf_dict.items():
        file.write (f'{key}: {value}\n')



