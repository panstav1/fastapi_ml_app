from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import load_encoders, process_data
import catboost as cb
import optuna
import pickle

def train_model(x_train, y_train):
    """
    Trains a machine learning model with hyperparameter tuning and returns it.

    Inputs
    ------
    x_train : np.array
        Training data.
    y_train : np.array
        Labels.

    Returns
    -------
    model
        Trained machine learning model.
    """

    def objective(trial):
        # Define hyperparameter search space
        param = {
            'objective': 'CrossEntropy',
            'iterations': trial.suggest_int('iterations', 50, 1000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'random_strength': trial.suggest_int('random_strength', 0, 100),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'od_type': 'Iter',
            'od_wait': 50
        }

        # Create and fit the model
        model = cb.CatBoostClassifier(**param, verbose=0)
        model.fit(x_train, y_train)

        # Evaluate the model
        preds = inference(model,x_train)
        f1_score = compute_model_metrics(y_train, preds)[-1]

        return f1_score

    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1)

    # Train the final model with the best hyperparameters
    best_params = study.best_params
    model = cb.CatBoostClassifier(**best_params)
    model.fit(x_train, y_train)

    return model


def save_model(model, filename):
    """
    Saves a machine learning model to a file.

    Parameters
    ----------
    model : object
        The machine learning model to be saved.
    filename : str
        The file path where the model should be saved.
    """

    # Check if the model is a CatBoost model
    try:
        model.save_model(filename)
    except AttributeError as e:
        # Use pickle for other model types
        with open(filename, 'wb') as file:
            pickle.dump (model, file)


def load_model(filename):
    """
    Loads a machine learning model from a file.

    Parameters
    ----------
    filename : str
        The file path from where the model should be loaded.

    Returns
    -------
    model : object
        The loaded machine learning model.
    """
    try:
        # Attempt to load as a CatBoost model
        model = cb.CatBoostClassifier()
        model.load_model(filename)
    except Exception as e:
        # If loading as a CatBoost model fails, try loading with pickle
        with open(filename, 'rb') as file:
            model = pickle.load(file)

    return model

def compute_model_metrics(y_ground, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y_ground : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y_ground, preds, beta=1, zero_division=1)
    precision = precision_score(y_ground, preds, zero_division=1)
    recall = recall_score(y_ground, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, input_data):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : pickle
        Trained machine learning model.
    input_data : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    return model.predict(input_data)
