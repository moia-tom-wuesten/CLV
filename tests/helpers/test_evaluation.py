import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.CLV.helpers.evaluation import Evaluation


def test_root_mean_squared_error():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    e = Evaluation(y_true, y_pred)
    e.root_mean_squared_error(y_true, y_pred)
    expected_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    assert e.evaluation["RMSE"] == expected_rmse


def test_mae_error():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    e = Evaluation(y_true, y_pred)
    e.mae_error(y_true, y_pred)
    expected_mae = mean_absolute_error(y_true, y_pred)
    assert e.evaluation["MAE"] == expected_mae


def test_mse_error():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    e = Evaluation(y_true, y_pred)
    e.mse_error(y_true, y_pred)
    expected_mse = mean_squared_error(y_true, y_pred)
    assert e.evaluation["MSE"] == expected_mse


def test_r_2_score():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    e = Evaluation(y_true, y_pred)
    e.r_2_score(y_true, y_pred)
    expected_r2 = r2_score(y_true, y_pred)
    assert e.evaluation["R2"] == expected_r2


def test_calculate_all_scores():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    e = Evaluation(y_true, y_pred)
    eval_df = e.calculate_all_scores()
    assert isinstance(eval_df, pd.DataFrame)
    assert eval_df.shape == (1, 4)
    assert np.all(eval_df.columns == ["RMSE", "MAE", "MSE", "R2"])
