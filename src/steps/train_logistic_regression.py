import logging
from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from core.step import Step


class LogisticRegressionStep(Step):
    name = "logistic_regression"
    description = "Trains a logistic regression classifier"
    def __init__(self, grid_search: bool = False, param_grid: dict = None, **params):
        self.grid_search = grid_search
        self.param_grid = param_grid or {
            "penalty": ["l2"],
            "C": [0.01, 0.1, 1, 10],
            "solver": ["liblinear", "lbfgs", "sag", "saga", "newton-cg"],
            "max_iter": [100, 200, 300],
        }
        self.model = LogisticRegression(**params)

    def set_stats(self, data: dict):
        data["stats"]["time"].append((self.name, datetime.now()))
        data["stats"]["model"] = "Logistic Regression"

    def run(self, data: dict) -> dict:

        """
        Trains a Losigstic Regression model and stores it in the data dictionary.
        """

        if "X_train" not in data:
            raise ValueError("No train data found in the data dictionary.")

        if "y_train" not in data:
            raise ValueError("No training labels found in the data dictionary.")

        if "dataset" not in data:
            raise ValueError("No dataset found in the data dictionary.")

        X = data["X_train"]
        y = data["y_train"]

        if self.grid_search:
            self.step_log(f"Running grid search for {self.name}...")
            grid_search = GridSearchCV(
                LogisticRegression(),
                self.param_grid,
                cv=5,
                scoring="accuracy",
                n_jobs=-1,
            )
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            data["hyperparameters"] = grid_search.best_params_
            self.step_log(
                f"Best parameters from grid search: {grid_search.best_params_}"
            )
            data['stats']['hyperparameters_used'] = grid_search.best_params_
            data['stats']['hyperparameters_searched'] = self.param_grid
        else:
            data["hyperparameters"] = "default"
            logging.info(f"Training {self.name} model...")
            self.model.fit(X, y)

        # Store the model in data for later use
        data["model"] = self.model

        return data
