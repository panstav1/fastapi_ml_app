import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import joblib


def process_data(
    input_data, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    input_data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `input_data`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    features : np.array
        Processed data.
    dep_var : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        dep_var = input_data[label]
        features = input_data.drop([label], axis=1)
    else:
        dep_var = np.array([])
        features = input_data

    x_categorical = features[categorical_features].values
    x_continuous = features.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        x_categorical = encoder.fit_transform(x_categorical)
        dep_var = lb.fit_transform(dep_var.values).ravel()
    else:
        x_categorical = encoder.transform(x_categorical)
        try:
            dep_var = lb.transform(dep_var.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    features = np.concatenate([x_continuous, x_categorical], axis=1)
    return features, dep_var, encoder, lb


def save_encoders(encoder, lb, encoder_filepath, lb_filepath):
    """ Save the trained OneHotEncoder and LabelBinarizer to disk.

    This function saves the trained OneHotEncoder and LabelBinarizer objects to disk,
    allowing them to be reused for processing new data in inference or validation. This
    ensures that the same preprocessing steps used during training are applied consistently.

    Inputs
    ------
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder object to be saved.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer object to be saved.
    encoder_filepath : str
        Filepath for saving the OneHotEncoder object.
    lb_filepath : str
        Filepath for saving the LabelBinarizer object.

    Returns
    -------
    None
    """

    joblib.dump(encoder, encoder_filepath)
    joblib.dump(lb, lb_filepath)


def load_encoders(encoder_filepath, lb_filepath):
    """ Load the trained OneHotEncoder and LabelBinarizer from disk.

    This function loads the OneHotEncoder and LabelBinarizer objects from disk,
    which were previously saved during the training phase. Loading these objects
    allows for consistent preprocessing steps to be applied to new data during
    inference or validation.

    Inputs
    ------
    encoder_filepath : str
        Filepath from where to load the OneHotEncoder object.
    lb_filepath : str
        Filepath from where to load the LabelBinarizer object.

    Returns
    -------
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Loaded OneHotEncoder object.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Loaded LabelBinarizer object.
    """

    encoder = joblib.load(encoder_filepath)
    lb = joblib.load(lb_filepath)
    return encoder, lb
