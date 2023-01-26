from typing import Dict, List, Tuple, Union
import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
# import seaborn as sns
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
    average_precision_score,
)
from sklearn.calibration import CalibratedClassifierCV
import xgboost


class TrainingOrchestrator(object):
    """Orchestrator for the whole training loop. The main methods output the classifier and its performance evaluation.

    Parameters:
        target_col_name (str): Name of the target variable in the dataframe
        unique_id_col_name (str): Name of the ID field in the dataframe
        selected_cols (List[str]): List of selected features for the model
        log_transform_cols (List[str]): List of features to which we must apply log1p
        merchant_groups (List[str]): List of considered entries for `merchant group`
        test_set_size (float): Percentage of fataframe to be used as test set (0-1)
        calibrate (bool): Whether or not xgboost model should have its probabilities calibrated
        n_folds (int): Number of folds to be used for calibration, if `calibrate` is set to True
        xgboost_parameters (Dict[str, Union[int, float]]): XGBoost's parameters for training
        save_eval_artifacts (bool, optional): Whether or not evaluation outputs should be saved. Defaults to False.
        eval_artifacts_path (str, optional): Path to where evaluation artifacts should be dumped. Defaults to "../media/".
        verbose (bool, optional): Whether or not logging information should be displayed. Defaults to False.
        random_seed (int, optional): Seed to ensure reproducibility. Defaults to 99.
    """

    def __init__(
        self,
        target_col_name: str,
        unique_id_col_name: str,
        selected_cols: List[str],
        log_transform_cols: List[str],
        merchant_groups: List[str],
        test_set_size: float,
        calibrate: bool,
        n_folds: int,
        xgboost_parameters: Dict[str, Union[int, float]],
        save_eval_artifacts: bool = False,
        eval_artifacts_path: str = "../media/",
        verbose: bool = False,
        random_seed: int = 99,
    ) -> None:
        """Constructor for TrainingOrchestrator class.

        Args:
            target_col_name (str): Name of the target variable in the dataframe
            unique_id_col_name (str): Name of the ID field in the dataframe
            selected_cols (List[str]): List of selected features for the model
            log_transform_cols (List[str]): List of features to which we must apply log1p
            merchant_groups (List[str]): List of considered entries for `merchant group`
            test_set_size (float): Percentage of fataframe to be used as test set (0-1)
            calibrate (bool): Whether or not xgboost model should have its probabilities calibrated
            n_folds (int): Number of folds to be used for calibration, if `calibrate` is set to True
            xgboost_parameters (Dict[str, Union[int, float]]): XGBoost's parameters for training
            save_eval_artifacts (bool, optional): Whether or not evaluation outputs should be saved. Defaults to False.
            eval_artifacts_path (str, optional): Path to where evaluation artifacts should be dumped. Defaults to "../media/".
            verbose (bool, optional): Whether or not logging information should be displayed. Defaults to False.
            random_seed (int, optional): Seed to ensure reproducibility. Defaults to 99.
        """
        self._target_col_name = target_col_name
        self._unique_id_col_name = unique_id_col_name
        self._selected_cols = selected_cols
        self._log_transform_cols = log_transform_cols
        self._merchant_groups = merchant_groups
        self._test_set_size = test_set_size
        self._calibrate = calibrate
        self._n_folds = n_folds
        self._xgboost_parameters = xgboost_parameters
        self._save_eval_artifacts = save_eval_artifacts
        self._eval_artifacts_path = eval_artifacts_path
        self._verbose = verbose
        self._random_seed = random_seed
        self._model = None
        self._fitted = False

    @property
    def target_col_name(self) -> str:
        return self._target_col_name

    @property
    def unique_id_col_name(self) -> str:
        return self._unique_id_col_name

    @property
    def selected_cols(self) -> List[str]:
        return self._selected_cols

    @property
    def log_transform_cols(self) -> List[str]:
        return self._log_transform_cols

    @property
    def merchant_groups(self) -> List[str]:
        return self._merchant_groups

    @property
    def test_set_size(self) -> float:
        return self._test_set_size

    @property
    def calibrate(self) -> bool:
        return self._calibrate

    @property
    def n_folds(self) -> int:
        return self._n_folds

    @property
    def xgboost_parameters(self) -> Dict[str, Union[int, float]]:
        return self._xgboost_parameters

    @property
    def save_eval_artifacts(self) -> bool:
        return self._save_eval_artifacts

    @property
    def eval_artifacts_path(self) -> str:
        return self._eval_artifacts_path

    @property
    def verbose(self) -> str:
        return self._verbose

    @property
    def random_seed(self) -> int:
        return self._random_seed

    def _get_feature_groups(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        all_fields = list(
            filter(
                lambda x: x not in [self._target_col_name, self._unique_id_col_name],
                df.columns,
            )
        )

        cat_features = list(filter(lambda x: "status" in x, all_fields))
        bool_features = list(
            filter(lambda x: x.startswith("is_") or x.startswith("has_"), all_fields)
        )
        string_features = list(
            filter(lambda x: df[x].dtype.name == "object", all_fields)
        )
        numeric_features = list(
            set(all_fields)
            - set(cat_features)
            - set(bool_features)
            - set(string_features)
        )

        return {
            "category": cat_features,
            "bool": bool_features,
            "str": string_features,
            "float64": numeric_features,
        }

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # DROP MISSING TAREGT
        df = df.dropna(subset=[self._target_col_name])

        # SELECTING COLUMNS
        df = df.loc[
            :, self._selected_cols + [self._target_col_name, self._unique_id_col_name]
        ]

        # GET FEATURE TYPES
        feature_groups = self._get_feature_groups(df)
        for feat_type, feat_group in feature_groups.items():
            df[feat_group] = df[feat_group].astype(feat_type)

        # REPLACE ALL STATUS WITH SINGLE ONE FOR ENTIRE PERIOD
        status_cols = filter(lambda x: x.startswith("account_worst_status"), df.columns)
        df["account_worst_status_all"] = np.max(df.loc[:, status_cols], axis=1)
        df = df.drop(columns=[status_cols])

        # CONVERT BOOLEAN
        df[feature_groups["bool"]] = df[feature_groups["bool"]].astype(int)

        # APPLY LOG
        for col in self._log_transform_cols:
            df[col] = np.log1p(df[col])

        # CONVERTING BACK CATEGORIES TO FLOAT TYPE
        cat_cols = list(set(feature_groups["category"]) & set(df.columns))
        df[cat_cols] = df[cat_cols].astype(float)

        # MERCHANT GROUP TRANSFORMATION
        merchant_group_column = "merchant_group"
        for cat in self._merchant_groups:
            df[cat] = (df[merchant_group_column] == cat).astype(float)
        df = df.drop(columns=[merchant_group_column])

        return df

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_test_split(
            df, test_size=self._test_set_size, random_state=self._random_seed
        )

    def _get_features_and_targets(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        return (
            df.drop(columns=[self._unique_id_col_name, self._target_col_name]),
            df[self._target_col_name],
        )

    def _get_model(self) -> Union[xgboost.XGBClassifier, CalibratedClassifierCV]:
        if self._fitted:
            return self._model
        self._model = xgboost.XGBClassifier(
            objective="binary:logistic", seed=self._random_seed
        )
        self._model.set_params(self._xgboost_parameters)
        if self._calibrate:
            self._model = CalibratedClassifierCV(
                self._model, method="sigmoid", cv=self._n_folds
            )
        return self._model

    def _get_metrics(self, df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
        df_ = self._preprocess_data(df.copy())
        X, y = self._get_features_and_targets(df_)
        y_proba = self._get_model().predict_proba(X)
        y_pred = (y_proba > threshold) * 1.0
        recall = recall_score(y, y_pred)
        precision = precision_score(y, y_pred)
        f1 = 2 * recall * precision / (recall + precision)
        auc = roc_auc_score(y, y_proba)
        avg_p = average_precision_score(y, y_proba)
        return {
            "RECALL": recall,
            "PRECISION": precision,
            "F1": f1,
            "ROC-AUC": auc,
            "AVERAGE PRECISION": avg_p
        }

    def _plot_curves(self) -> None:
        pass

    def _plot_distribution(self) -> None:
        pass

    def _generate_band_analysis(self) -> None:
        pass

    def fit(
        self, df: pd.DataFrame
    ) -> Union[xgboost.XGBClassifier, CalibratedClassifierCV]:
        # PREPROCESS
        df_ = self._preprocess_data(df.copy())

        # SPLIT DATA
        df_train, _ = self._split_data(df_)
        X_train, y_train = self._get_features_and_targets(df_train)

        # FIT MODEL
        self._model = self._get_model().fit(X_train, y_train)
        self._fitted = True
        return self._model

    def evaluate_performance(self) -> None:
        pass
