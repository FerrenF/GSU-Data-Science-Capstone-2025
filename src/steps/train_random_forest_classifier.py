import logging
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from core.step import Step


class RandomForestClassificationStep(Step):
    name = "random_forest_classification"

    def __init__(self, grid_search: bool = False, param_grid: dict = None, **params):
        """
        Random forests are overpowered. (A tad boring in how they work, however.)

        :param grid_search: Whether to perform a GridSearchCV to optimize hyperparameters.
        :param param_grid: Grid of parameters for GridSearchCV.
        :param params: Default model parameters if not using grid search.
        """
        self.grid_search = grid_search
        self.param_grid = param_grid or {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        }
        self.model = RandomForestClassifier(**params)

    def set_stats(self, data: dict):
        data["stats"]["time"].append((self.name, datetime.now()))
        data["stats"]["model"] = "Random Forest Classifier"

    def run(self, data: dict) -> dict:
        if "dataset" not in data:
            raise ValueError("No dataset found in the data dictionary.")
        if "X_train" not in data or "y_train" not in data:
            raise ValueError("Training data or labels missing from the data dictionary.")

        X = data["X_train"]
        y = data["y_train"]

        if self.grid_search:
            logging.info(f"Running grid search for {self.name}...")
            grid_search = GridSearchCV(RandomForestClassifier(), self.param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            data["hyperparameters"] = grid_search.best_params_
            logging.info(f"Best parameters from grid search: {grid_search.best_params_}")
        else:
            data["hyperparameters"] = "default"
            logging.info(f"Training {self.name} model...")
            self.model.fit(X, y)

        data["model"] = self.model
        logging.info(f"Training complete for {self.name}.")

        return data
