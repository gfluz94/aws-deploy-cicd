from typing import Any, Dict
from serve.app import USERID_COLUMN, MERCHANT_GROUPS, LOG_TRANSFORM_COLUMNS
from serve.predictor import PredictorService


class TestPredictorService(object):

    service = PredictorService(
        model_path="output",
        user_id_col=USERID_COLUMN,
        merchant_groups=MERCHANT_GROUPS,
        log_transform_cols=LOG_TRANSFORM_COLUMNS,
    )

    def test_predict(
        self,
        prediction_input_json: Dict[str, Any],
        expected_prediction_output: Dict[str, Any],
    ):
        # OUTPUT
        output = self.service.predict(prediction_input_json)

        # EXPECTED
        expected = expected_prediction_output

        # ASSERT
        assert output == expected
