from typing import Any, Dict
import os
import pandas as pd
import dill
from xgboost import XGBClassifier


class PredictorService(object):
    """Trained model wrapper to predict default probability during inference time.
    Parameters:
        model_path (str, optional): Path within Docker container where model is located. Defaults to "/app/models".
    """

    def __init__(self, model_path: str = "models"):
        """Constructor method for PredictorService
        Args:
            model_path (str, optional): Path within Docker container where model is located. Defaults to "/app/models".
        """
        self._model_path = model_path

    @property
    def model_path(self) -> str:
        """(str) Path within Docker container where model is located."""
        return self._model_path

    def _get_model(self) -> Dict[str, XGBClassifier]:
        """Method that allows for loading and instantiation of trained models, like a singleton.
        Returns:
            XGBClassifier: Trained classifier
        """
        if not hasattr(self, "_model"):
                with open(os.path.join(self._model_path, f"default_classifier.pkl"), "rb") as f:
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

    def _preprocess_input_df(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Method to preprocess input dataframe in a format accepted by the model
        Args:
            df (pandas.DataFrame): Input dataframe containing variables
        Returns:
            pandas.DataFrame: Preprocessed dataframe to be fed to the model
        """
        pass

    def _convert_probability_to_score(
        self, probability: float
    ) -> float:
        """Method to convert output probability into credit score
        Args:
            probability (float): Default probability output by the model
        Returns:
            float: Final credit score
        """
        pass

    def predict(self, request_inputs: Dict[str, Any]) -> Dict[str, float]:
        """Method for prediction whenever a request invokes the model in production
        Args:
            request_inputs (Dict[str, Any]): Raw input from the server
        Returns:
            Dict[str, float]: Final prediction for default probability and credit score
        """
        predictions = {
            "uuid": request_inputs["uuid"]
        }
        X = self._convert_to_pandas(request_inputs)
        X = self._preprocess_input_df(X)
        predictions["default_probability"] = self._get_model().predict_proba(X)[:, 1].tolist()[0]
        predictions["credit_score"] = self._convert_probability_to_score(predictions["default_probability"])
        return predictions