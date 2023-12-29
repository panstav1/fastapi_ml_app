from sklearn.metrics import fbeta_score, precision_score, recall_score
import catboost as cb
import optuna

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
        score = model.get_best_score()

        return score

    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    # Train the final model with the best hyperparameters
    best_params = study.best_params
    model = cb.CatBoostClassifier(**best_params)
    model.fit(x_train, y_train)

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


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    pass
