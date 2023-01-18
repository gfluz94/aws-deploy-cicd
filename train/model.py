from typing import Dict, List, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(
    level=logging.DEBUG, format="[%(asctime)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger()


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    precision_score,
    recall_score,
    average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
import xgboost


class TrainingOrchestrator(object):

    def __init__(
        self,
        target_col_name: str,
        unique_id_col_name: str,
        selected_cols: List[str],
        merchant_groups: List[str],
        test_set_size: float,
        n_folds: int,
        calibrate: bool,
        xgboost_parameters: Dict[str, Union[int, float]],
        random_seed: int = 99,
        save_eval_artifacts: bool = False,
        eval_artifacts_path: str = "../media/",
        verbose: bool = False
    ) -> None:
        pass

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass

    def _train_xgboost(self) -> xgboost.XGBClassifier:
        pass

    def _get_metrics(self) -> None:
        pass

    def _plot_curves(self) -> None:
        pass

    def _plot_distirbution(self) -> None:
        pass

    def _generate_band_analysis(self) -> None:
        pass

    def fit(self, df: pd.DataFrame) -> xgboost.XGBClassifier:
        pass

    def evaluate_performance(self) -> None:
        pass

