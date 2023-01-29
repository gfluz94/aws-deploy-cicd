from typing import Dict
import pandas as pd
import pytest

from train.model import TrainingOrchestrator
from train.exceptions import ModelNotFitted


class TestTrainingOrchestrator(object):
    def test_evaluate_performanceRaisesExceptionModelNotFitted(
        self, input_raw_data: pd.DataFrame
    ) -> None:
        training_orchestrator = TrainingOrchestrator(
            target_col_name="default",
            unique_id_col_name="uuid",
            selected_cols=[
                "max_paid_inv_0_24m",
                "avg_payment_span_0_12m",
                "sum_capital_paid_account_0_12m",
                "time_hours",
                "recovery_debt",
                "sum_capital_paid_account_12_24m",
                "num_active_div_by_paid_inv_0_12m",
                "sum_paid_inv_0_12m",
                "account_days_in_rem_12_24m",
                "num_arch_ok_0_12m",
                "account_amount_added_12_24m",
                "merchant_group",
                "has_paid",
                "account_status",
                "account_worst_status_0_24m",
                "status_last_archived_0_24m",
                "status_2nd_last_archived_0_24m",
                "status_3rd_last_archived_0_24m",
                "status_max_archived_0_6_months",
                "status_max_archived_0_12_months",
                "status_max_archived_0_24_months",
            ],
            log_transform_cols=["max_paid_inv_0_24m", "sum_capital_paid_account_0_12m"],
            merchant_groups=[
                "Clothing & Shoes",
                "Intangible products",
                "Food & Beverage",
                "Erotic Material",
                "Entertainment",
            ],
            test_set_size=0.25,
            calibrate=True,
            n_folds=3,
            xgboost_parameters={"n_estimators": 200},
            save_eval_artifacts=False,
            eval_artifacts_path=".",
            verbose=False,
            random_seed=99,
        )
        with pytest.raises(ModelNotFitted):
            training_orchestrator.evaluate_performance(input_raw_data, threshold=0.50)

    def test_fit_evaluate_performanceReturnsExpectedMetricsAndDataFrames(
        self,
        input_raw_data: pd.DataFrame,
        metrics_output: Dict[str, float],
        train_bands_output: pd.DataFrame,
        test_bands_output: pd.DataFrame,
    ) -> None:
        training_orchestrator = TrainingOrchestrator(
            target_col_name="default",
            unique_id_col_name="uuid",
            selected_cols=[
                "max_paid_inv_0_24m",
                "avg_payment_span_0_12m",
                "sum_capital_paid_account_0_12m",
                "time_hours",
                "recovery_debt",
                "sum_capital_paid_account_12_24m",
                "num_active_div_by_paid_inv_0_12m",
                "sum_paid_inv_0_12m",
                "account_days_in_rem_12_24m",
                "num_arch_ok_0_12m",
                "account_amount_added_12_24m",
                "merchant_group",
                "has_paid",
                "account_status",
                "account_worst_status_0_3m",
                "account_worst_status_12_24m",
                "account_worst_status_3_6m",
                "account_worst_status_6_12m",
                "status_last_archived_0_24m",
                "status_2nd_last_archived_0_24m",
                "status_3rd_last_archived_0_24m",
                "status_max_archived_0_6_months",
                "status_max_archived_0_12_months",
                "status_max_archived_0_24_months",
            ],
            log_transform_cols=["max_paid_inv_0_24m", "sum_capital_paid_account_0_12m"],
            merchant_groups=[
                "Clothing & Shoes",
                "Intangible products",
                "Food & Beverage",
                "Erotic Material",
                "Entertainment",
            ],
            test_set_size=1 / 3.0,
            calibrate=False,
            n_folds=2,
            xgboost_parameters={"n_estimators": 200},
            save_eval_artifacts=False,
            eval_artifacts_path=".",
            verbose=False,
            random_seed=99,
        )
        _ = training_orchestrator.fit(input_raw_data)
        (train_metrics, train_bands), (
            test_metrics,
            test_bands,
        ) = training_orchestrator.evaluate_performance(input_raw_data, threshold=0.50)
        print(test_bands.to_dict())
        assert train_metrics == metrics_output and test_metrics == metrics_output
        pd.testing.assert_frame_equal(train_bands, train_bands_output)
        pd.testing.assert_frame_equal(test_bands, test_bands_output)
