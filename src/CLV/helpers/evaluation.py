import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class Evaluation:
    def __init__(self, y, predictions):
        super(Evaluation, self).__init__()
        self.y = y
        self.predictions = predictions
        self.evaluation = {}

    def root_mean_squared_error(self, actual, predictions):
        """
        calculate RMSE
        """
        self.evaluation["RMSE"] = np.sqrt(mean_squared_error(actual, predictions))

    def mae_error(self, actual, predictions):
        """
        calculate MAE (mean_absolute_error)
        """
        self.evaluation["MAE"] = mean_absolute_error(actual, predictions)

    def mse_error(self, actual, predictions):
        """
        calculate MSE (mean_squared_error)
        """
        self.evaluation["MSE"] = mean_squared_error(actual, predictions)

    def r_2_score(self, actual, predictions):
        """
        calculate r2_score
        """
        self.evaluation["R2"] = r2_score(actual, predictions)

    def calculate_all_scores(self):
        self.root_mean_squared_error(self.y, self.predictions)
        self.mae_error(self.y, self.predictions)
        self.mse_error(self.y, self.predictions)
        self.r_2_score(self.y, self.predictions)
        eval_df = pd.DataFrame([self.evaluation])
        return eval_df
