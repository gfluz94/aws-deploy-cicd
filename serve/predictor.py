from typing import Any, Dict, List
import os
import numpy as np
import pandas as pd
import dill
from xgboost import XGBClassifier


class PredictorService(object):
    """Trained model wrapper to predict default probability during inference time.
    Parameters:
        model_path (str, optional): Path within Docker container where model is located. Defaults to "/app/models".
        user_id_col (str, optional): Field name for user id. Defaults to "uuid".
        merchant_groups (List[str], optional): Set of merchant groups of interest for preprocessing.
        log_transform_cols (List[str], optional): Features that require log transformation.
    """

    def __init__(
        self,
        model_path: str = "models",
        user_id_col: str = "uuid",
        merchant_groups: List[str] = [
            "Clothing & Shoes",
            "Intangible products",
            "Food & Beverage",
            "Erotic Material",
            "Entertainment",
        ],
        log_transform_cols: List[str] = [
            "max_paid_inv_0_24m", "sum_capital_paid_account_0_12m"
        ],
    ):
        """Constructor method for PredictorService
        Args:
            model_path (str, optional): Path within Docker container where model is located. Defaults to "/app/models".
            user_id_col (str, optional): Field name for user id. Defaults to "uuid".
            merchant_groups (List[str], optional): Set of merchant groups of interest for preprocessing.
            log_transform_cols (List[str], optional): Features that require log transformation.
        """
        self._model_path = model_path
        self._user_id_col = user_id_col
        self._merchant_groups = merchant_groups
        self._log_transform_cols = log_transform_cols
        self._model = None

    @property
    def model_path(self) -> str:
        """(str) Path within Docker container where model is located."""
        return self._model_path

    @property
    def user_id_col(self) -> str:
        """(str, optional): Field name for user id."""
        return self._user_id_col

    @property
    def merchant_groups(self) -> str:
        """(List[str], optional) Set of merchant groups of interest for preprocessing."""
        return self._merchant_groups

    @property
    def log_transform_cols(self) -> str:
        """(List[str], optional) Features that require log transformation."""
        return self._log_transform_cols

    def _get_model(self) -> Dict[str, XGBClassifier]:
        """Method that allows for loading and instantiation of trained models, like a singleton.
        Returns:
            XGBClassifier: Trained classifier
        """
        if self._model is None:
            with open(
                os.path.join(self._model_path, "default_classifier.pkl"), "rb"
            ) as f:
                self._model = dill.load(f)

        return self._model

    def _convert_to_pandas(self, request: Dict[str, Any]) -> pd.DataFrame:
        """Method to convert raw request to a dataframe
        Args:
            request (Dict[str, Any]): Request containing features to be fed to the model
        Returns:
            pandas.DataFrame: Dataframe containing features
        """
        return pd.DataFrame([request])

    def _preprocess_input_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Method to preprocess input dataframe in a format accepted by the model
        Args:
            df (pandas.DataFrame): Input dataframe containing variables
        Returns:
            pandas.DataFrame: Preprocessed dataframe to be fed to the model
        """

        # DROP USER ID
        df = df.drop(columns=[self._user_id_col])

        # SINGLE FEATURE FOR WORST STATUS EVER
        status_cols = list(
            filter(lambda x: x.startswith("account_worst_status"), df.columns)
        )
        df["account_worst_status_all"] = np.max(df.loc[:, status_cols], axis=1)
        df = df.drop(columns=status_cols)

        # CONVERTING BOOLEAN FEATURE INTO INTEGER
        bool_features = list(
            filter(lambda x: x.startswith("is_") or x.startswith("has_"), df.columns)
        )
        df.loc[:, bool_features] = df.loc[:, bool_features].astype(int)

        # CONVERTING BOOLEAN FEATURE INTO INTEGER
        merchant_group_column = "merchant_group"
        for cat in self._merchant_groups:
            df[cat] = (df[merchant_group_column] == cat).astype(float)
        df = df.drop(columns=[merchant_group_column])

        # APPLYING LOG
        for col in self._log_transform_cols:
            df[col] = np.log1p(df[col])

        return df

    def _convert_probability_to_score(self, probability: float) -> float:
        """Method to convert output probability into credit score
        Args:
            probability (float): Default probability output by the model
        Returns:
            float: Final credit score
        """
        double_decrease_factor = 20 / np.log(2)
        constant = 600 - np.log(50) * double_decrease_factor
        probability = np.clip(probability, 1e-8, 0.99999999)
        return constant - np.log(probability / (1 - probability)) * double_decrease_factor

    def predict(self, request_inputs: Dict[str, Any]) -> Dict[str, float]:
        """Method for prediction whenever a request invokes the model in production
        Args:
            request_inputs (Dict[str, Any]): Raw input from the server
        Returns:
            Dict[str, float]: Final prediction for default probability and credit score
        """
        predictions = {"uuid": request_inputs["uuid"]}
        X = self._convert_to_pandas(request_inputs)
        X = self._preprocess_input_df(X)
        predictions["default_probability"] = (
            self._get_model().predict_proba(X)[:, 1].tolist()[0]
        )
        predictions["credit_score"] = self._convert_probability_to_score(
            predictions["default_probability"]
        )
        return predictions
