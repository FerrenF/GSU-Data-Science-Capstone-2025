import logging
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from core.step import Step


class SVRStep(Step):

    name="svr"
    def __init__(self, name: str = "svr", kernel='rbf', C=1.0, epsilon=0.1):
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)

    def run(self, data: dict) -> dict:
        """
        Trains a Support Vector Regression model using vectorized text data and stores it in the data dictionary.
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in the data dictionary.")

        df = data["dataset"]
        X = df["vector"]
        y = df["label"]  # Assuming 'target' column exists

        logging.info(f"Training {self.name} model...")
        self.model.fit(X, y)

        # Store the model in data for later use
        data["model"] = self.model
        logging.info(f"Training complete for {self.name}.")

        return data
