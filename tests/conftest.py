from typing import Any, Dict
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def input_raw_data() -> pd.DataFrame:
    """Example dataframe for testing purposes

    Returns:
        pandas.DataFrame: Mocked dataframe containing raw data from database
    """
    return pd.DataFrame(
        {
            "uuid": [
                "6a3b044e-2203-49a7-917e-0c4050023c6a",
                "ff3f1d1b-ddd5-4fdd-a10e-d5fe87b17fd0",
                "3f264b33-d392-493f-ad24-43006710de77",
                "575cebe3-2807-4dcb-8047-cf4ca9476a4d",
                "320f1544-064f-4568-a5f0-4ed54e91483a",
                "adc8c99d-2d3f-401a-8b8f-24d43f34ee70",
            ],
            "default": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "account_amount_added_12_24m": [24785, 0, 39270, 57229, 148922, 6383],
            "account_days_in_dc_12_24m": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "account_days_in_rem_12_24m": [60.0, 110.0, 0.0, 0.0, 47.0, 0.0],
            "account_days_in_term_12_24m": [20.0, 29.0, 0.0, 0.0, 0.0, 0.0],
            "account_incoming_debt_vs_paid_0_24m": [
                1.02887892747216,
                0.487131661442006,
                1.94328198871435,
                0.232244230969046,
                0.969055089046784,
                0.0,
            ],
            "account_status": [2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            "account_worst_status_0_3m": [2.0, 2.0, 2.0, 1.0, 2.0, 1.0],
            "account_worst_status_12_24m": [3.0, 3.0, 1.0, 1.0, 2.0, 1.0],
            "account_worst_status_3_6m": [2.0, 3.0, 2.0, 1.0, 2.0, 1.0],
            "account_worst_status_6_12m": [1.0, 2.0, 2.0, 1.0, 2.0, 1.0],
            "age": [35, 33, 18, 34, 40, 41],
            "avg_payment_span_0_12m": [
                45.0,
                42.0,
                28.1428571428571,
                26.9302325581395,
                33.7272727272727,
                14.25,
            ],
            "avg_payment_span_0_3m": [
                59.0,
                35.0,
                20.0,
                25.8666666666667,
                37.5714285714286,
                13.0,
            ],
            "merchant_category": [
                "Youthful Shoes & Clothing",
                "Dietary supplements",
                "Youthful Shoes & Clothing",
                "Diversified entertainment",
                "Diversified entertainment",
                "Diversified entertainment",
            ],
            "merchant_group": [
                "Clothing & Shoes",
                "Health & Beauty",
                "Clothing & Shoes",
                "Entertainment",
                "Entertainment",
                "Entertainment",
            ],
            "has_paid": [True, True, True, True, True, True],
            "max_paid_inv_0_12m": [2920.0, 10330.0, 7190.0, 8655.0, 6075.0, 18385.0],
            "max_paid_inv_0_24m": [2920.0, 17358.0, 7190.0, 9645.0, 9090.0, 18385.0],
            "name_in_email": ["no_match", "Nick", "F+L", "F", "Nick", "F+L"],
            "num_active_div_by_paid_inv_0_12m": [
                2.5,
                0.166666666666667,
                0.125,
                0.0833333333333333,
                0.818181818181818,
                0.225,
            ],
            "num_active_inv": [5, 1, 1, 20, 9, 9],
            "num_arch_dc_0_12m": [0, 1, 0, 0, 0, 0],
            "num_arch_dc_12_24m": [0, 2, 0, 0, 0, 0],
            "num_arch_ok_0_12m": [0, 1, 5, 215, 3, 33],
            "num_arch_ok_12_24m": [0, 1, 6, 257, 2, 45],
            "num_arch_rem_0_12m": [2, 2, 2, 0, 3, 1],
            "num_arch_written_off_0_12m": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "num_arch_written_off_12_24m": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "num_unpaid_bills": [23, 6, 29, 37, 23, 9],
            "status_last_archived_0_24m": [2, 1, 1, 1, 1, 1],
            "status_2nd_last_archived_0_24m": [2, 2, 1, 1, 2, 1],
            "status_3rd_last_archived_0_24m": [0, 3, 1, 1, 2, 1],
            "status_max_archived_0_6_months": [2, 0, 1, 1, 2, 2],
            "status_max_archived_0_12_months": [2, 3, 2, 1, 2, 2],
            "status_max_archived_0_24_months": [2, 3, 2, 1, 2, 2],
            "recovery_debt": [0, 0, 0, 0, 0, 0],
            "sum_capital_paid_account_0_12m": [7297, 12807, 27166, 42206, 104643, 0],
            "sum_capital_paid_account_12_24m": [12943, 22569, 5059, 35336, 32381, 3190],
            "sum_paid_inv_0_12m": [7890, 49467, 31845, 457257, 24390, 123306],
            "time_hours": [
                19.525,
                21.5055555555556,
                9.80444444444445,
                12.1927777777778,
                21.4111111111111,
                21.6991666666667,
            ],
            "worst_status_active_inv": [1.0, 2.0, 1.0, 1.0, 1.0, 2.0],
        }
    )


@pytest.fixture(scope="module")
def train_bands_output() -> pd.DataFrame:
    """Example output dataframe for testing purposes

    Returns:
        pandas.DataFrame: Mocked dataframe containing band analysis
    """
    df = pd.DataFrame(
        {
            "Band Size": [4],
            "# Default": [2],
            "% Total Population": [1.0],
            "Band Precision": [0.5],
            "Cumulative Precision": [0.5],
            "Cumulative Recall": [1.0],
            "Cumulative F1-Score": [0.67],
        },
        index=["[480, 490]"],
    )
    df.index.name = "Band"
    return df


@pytest.fixture(scope="module")
def test_bands_output() -> pd.DataFrame:
    """Example output dataframe for testing purposes

    Returns:
        pandas.DataFrame: Mocked dataframe containing band analysis
    """
    df = pd.DataFrame(
        {
            "Band Size": [2],
            "# Default": [1],
            "% Total Population": [1.0],
            "Band Precision": [0.5],
            "Cumulative Precision": [0.5],
            "Cumulative Recall": [1.0],
            "Cumulative F1-Score": [0.67],
        },
        index=["[480, 490]"],
    )
    df.index.name = "Band"
    return df


@pytest.fixture(scope="module")
def metrics_output() -> Dict[str, float]:
    """Example output for testing purposes

    Returns:
        Dict[str, float]: Dictionary containing metrics
    """
    return {
        "RECALL": 0.0,
        "PRECISION": 0.0,
        "F1": None,
        "ROC-AUC": 0.5,
        "AVERAGE PRECISION": 0.5,
    }


@pytest.fixture(scope="module")
def prediction_input_json() -> Dict[str, Any]:
    """Example output for testing purposes

    Returns:
        Dict[str, float]: Dictionary containing prediction request
    """
    return {
        "uuid": "1234",
        "max_paid_inv_0_24m": 10.0,
        "avg_payment_span_0_12m": 10.0,
        "sum_capital_paid_account_0_12m": 10.0,
        "time_hours": 10.0,
        "recovery_debt": 10.0,
        "sum_capital_paid_account_12_24m": 10.0,
        "num_active_div_by_paid_inv_0_12m": 10.0,
        "sum_paid_inv_0_12m": 10.0,
        "account_days_in_rem_12_24m": 10.0,
        "num_arch_ok_0_12m": 10.0,
        "account_amount_added_12_24m": 10.0,
        "has_paid": True,
        "account_status": 1.0,
        "account_worst_status_0_3m": 1.0,
        "account_worst_status_3_6m": 1.0,
        "account_worst_status_6_12m": 2.0,
        "account_worst_status_12_24m": 1.0,
        "status_last_archived_0_24m": 1.0,
        "status_2nd_last_archived_0_24m": 1.0,
        "status_3rd_last_archived_0_24m": 1.0,
        "status_max_archived_0_6_months": 1.0,
        "status_max_archived_0_12_months": 1.0,
        "status_max_archived_0_24_months": 1.0,
        "merchant_group": "Entertainment",
    }


@pytest.fixture(scope="module")
def expected_prediction_output() -> Dict[str, Any]:
    """Example output for testing purposes

    Returns:
        Dict[str, Any]: Dictionary containing prediction output
    """
    return {
        "uuid": "1234",
        "default_probability": 0.23453453928232193,
        "credit_score": 521.2536138390417,
    }
