"""
Contain some metrics
"""
import numpy as np
# from lifelines.utils import concordance_index
# from pysurvival.utils._metrics import _concordance_index
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import integrated_brier_score


def c_index(true_T, true_E, pred_risk, include_ties=True):
    """
    Calculate c-index for survival prediction downstream task
    """
    # Ordering true_T, true_E and pred_score in descending order according to true_T
    order = np.argsort(-true_T)

    true_T = true_T[order]
    true_E = true_E[order]
    pred_risk = pred_risk[order]

    # Calculating the c-index
    # result = concordance_index(true_T, -pred_risk, true_E)
    # result = _concordance_index(pred_risk, true_T, true_E, include_ties)[0]
    result = concordance_index_censored(true_E.astype(bool), true_T, pred_risk)[0]

    return result


def ibs(true_T, true_E, pred_survival, time_points):
    """
    Calculate integrated brier score for survival prediction downstream task
    """
    true_E_bool = true_E.astype(bool)
    true = np.array([(true_E_bool[i], true_T[i]) for i in range(len(true_E))], dtype=[('event', np.bool_), ('time', np.float32)])

    # time points must be within the range of T
    min_T = true_T.min()
    max_T = true_T.max()
    valid_index = []
    for i in range(len(time_points)):
        if min_T <= time_points[i] <= max_T:
            valid_index.append(i)
    time_points = time_points[valid_index]
    pred_survival = pred_survival[:, valid_index]

    result = integrated_brier_score(true, true, pred_survival, time_points)

    return result
