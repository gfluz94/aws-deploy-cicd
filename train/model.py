from typing import Dict, List, Tuple, Union
import os
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
    average_precision_score,
)
from sklearn.calibration import CalibratedClassifierCV
import xgboost

from exceptions import ModelNotFitted


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

    def _convert_probabilities_to_score(self, y_proba: np.ndarray) -> np.ndarray:
        double_decrease_factor = 20 / np.log(2)
        constant = 600 - np.log(50) * double_decrease_factor
        return constant - np.log(y_proba / (1 - y_proba)) * double_decrease_factor

    def _get_metrics(
        self, y_proba: np.ndarray, y_true: pd.Series, threshold: float = 0.5
    ) -> Dict[str, float]:
        y_pred = (y_proba > threshold) * 1.0
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        f1 = 2 * recall * precision / (recall + precision)
        auc = roc_auc_score(y_true, y_proba)
        avg_p = average_precision_score(y_true, y_proba)
        return {
            "RECALL": recall,
            "PRECISION": precision,
            "F1": f1,
            "ROC-AUC": auc,
            "AVERAGE PRECISION": avg_p,
        }

    def _plot_curves(self, y_proba: np.ndarray, y_true: pd.Series, label: str) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4), sharey=True)
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ax1.plot(
            fpr, tpr, color="red", label=f"(AUC = {roc_auc_score(y_true, y_proba):.3f})"
        )
        ax1.plot([0, 1], [0, 1], color="navy")
        ax1.set_xlabel("FPR")
        ax1.set_ylabel("TPR")
        ax1.set_xlim((0, 1))
        ax1.set_ylim((0, 1.001))
        ax1.legend(loc=4)
        ax1.grid(alpha=0.15)
        ax1.set_title(f"{label.upper()} - ROC", fontsize=13)

        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ax2.plot(
            recall,
            precision,
            color="red",
            label=f"(AUC = {average_precision_score(y_true, y_proba):.3f}",
        )
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_xlim((0, 1))
        ax2.set_ylim((0, 1.001))
        ax2.legend(loc=4)
        ax2.grid(alpha=0.15)
        ax2.set_title(f"{label.upper()} - Precision-Recall", fontsize=13)
        plt.show()

        if self._save_eval_artifacts:
            fig.savefig(os.path.join(self._eval_artifacts_path, f"roc_pr_{label}.png"))

    def _plot_distribution(
        self, y_true: np.ndarray, scores: np.ndarray, label: str
    ) -> None:
        df = pd.DataFrame({"Label": y_true, "Predicted Score": scores})
        default = df[df.Label == 1.0]
        non_default = df[df.Label == 0.0]
        fig, ax = plt.subplots(1, 1, figsize=(15, 4))
        sns.distplot(
            default["Predicted Score"], bins=30, label="Default", color="red", ax=ax
        )
        sns.distplot(
            non_default["Predicted Score"],
            bins=30,
            label="Non-Default",
            color="blue",
            ax=ax,
        )
        ax.set_xlabel("Credit Score")
        ax.grid(alpha=0.15)
        ax.legend()
        ax.set_title(f"{label.upper()} - Score Distribution", fontsize=13)
        plt.show()

        if self._save_eval_artifacts:
            fig.savefig(
                os.path.join(self._eval_artifacts_path, f"distribution_{label}.png")
            )

    def _get_bands(
        self, score_min: float, score_max: float, step: float = 20 / np.log(2)
    ) -> List[Tuple[float]]:
        limits = np.arange(score_min, score_max, step, dtype=np.int32)
        limits = np.concatenate((limits, np.array([score_max])), axis=0)
        return list(zip(limits, limits[1:]))

    def _assign_bands(
        self, scores: np.ndarray, true_labels: np.ndarray
    ) -> pd.DataFrame:
        df = pd.DataFrame({"Score": scores, "Default": true_labels})
        score_min = np.int32(scores.min())
        score_max = np.int32(scores.max())
        bands = self._get_bands(
            score_min=(score_min - score_min % 10),
            score_max=(score_max + (10 - score_max % 10)),
        )
        df["Band"] = np.nan
        for i, (lower, upper) in enumerate(bands):
            right = ")"
            if i == len(bands) - 1:
                right = "]"
            df.loc[
                (df.Score >= lower) & (df.Score < upper), "Band"
            ] = f"[{lower}, {upper}{right}"
        return df

    def _run_band_analysis(
        self, scores: np.ndarray, true_labels: np.ndarray
    ) -> pd.DataFrame:
        df = self._assign_bands(scores, true_labels)
        df_ = (
            df.groupby("Band")
            .agg({"Score": "count", "Default": "sum"})
            .rename(columns={"Score": "Band Size", "Default": "# Default"})
        )
        df_["# Default"] = df_["# Default"].astype(int)
        df_["cumulative_size"] = df_["Band Size"].cumsum()
        df_["cumulative_true_positive"] = df_["# Default"].cumsum()

        # % TOTAL
        df_["% Total Population"] = np.round(
            df_["cumulative_size"] / df_["cumulative_size"].iloc[-1], 4
        )

        # BAND PRECISION
        df_["Band Precision"] = np.round(df_["# Default"] / df_["Band Size"], 2)

        # CUMULATIVE PRECISION
        df_["Cumulative Precision"] = np.round(
            df_["cumulative_true_positive"] / df_["cumulative_size"], 2
        )

        # CUMULATIVE RECALL
        df_["Cumulative Recall"] = np.round(
            df_["cumulative_true_positive"] / df_["cumulative_true_positive"].iloc[-1],
            2,
        )

        # CUMULATIVE F1-SCORE
        df_["Cumulative F1-Score"] = np.round(
            2
            * df_["Cumulative Recall"]
            * df_["Cumulative Precision"]
            / (df_["Cumulative Recall"] + df_["Cumulative Precision"]),
            2,
        )

        return df_[
            [
                "Band Size",
                "# Default",
                "% Total Population",
                "Band Precision",
                "Cumulative Precision",
                "Cumulative Recall",
                "Cumulative F1-Score",
            ]
        ]

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

    def evaluate_performance(
        self, df: pd.DataFrame, label: str, threshold: float
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        if not self._fitted:
            raise ModelNotFitted("Model needs to be fitted first!")

        df_ = self._preprocess_data(df.copy())
        X, y = self._get_features_and_targets(df_)

        # PREDICTIONS
        y_proba = self._get_model().predict(X)
        scores = self._convert_probabilities_to_score(y_proba)

        # METRICS
        metrics = self._get_metrics(y_proba, y, threshold=threshold)

        # CURVES
        self._plot_curves(y_proba, y, label)
        self._plot_distribution(y_proba, scores, label)

        # BANDS
        df_bands = self._run_band_analysis(scores, y)

        return (metrics, df_bands)
