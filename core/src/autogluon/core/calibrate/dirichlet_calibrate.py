import numpy as np
from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def dirichlet_calibrate(y_val_probs: np.ndarray, y_val: np.ndarray):
    reg = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    calibrator = FullDirichletCalibrator(reg_lambda=reg, reg_mu=None)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    grid_search_cv = GridSearchCV(calibrator, param_grid={'reg_lambda': reg,
                                                          'reg_mu': [None]},
                                  cv=skf, scoring='neg_log_loss')

    grid_search_cv.fit(y_val_probs, y_val)

    return grid_search_cv
