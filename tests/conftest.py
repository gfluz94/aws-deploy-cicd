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
                "575cebe3-2807-4dcb-8047-cf4ca9476a4d",
                "320f1544-064f-4568-a5f0-4ed54e91483a",
                "adc8c99d-2d3f-401a-8b8f-24d43f34ee70",
            ],
            "default": [0.0, 0.0, 0.0],
            "account_amount_added_12_24m": [57229, 148922, 6383],
            "account_days_in_dc_12_24m": [0.0, 0.0, 0.0],
            "account_days_in_rem_12_24m": [0.0, 47.0, 0.0],
            "account_days_in_term_12_24m": [0.0, 0.0, 0.0],
            "account_incoming_debt_vs_paid_0_24m": [
                0.232244230969046,
                0.969055089046784,
                0.0,
            ],
            "account_status": [1.0, 1.0, 1.0],
            "account_worst_status_0_3m": [1.0, 2.0, 1.0],
            "account_worst_status_12_24m": [1.0, 2.0, 1.0],
            "account_worst_status_3_6m": [1.0, 2.0, 1.0],
            "account_worst_status_6_12m": [1.0, 2.0, 1.0],
            "age": [34, 40, 41],
            "avg_payment_span_0_12m": [26.9302325581395, 33.7272727272727, 14.25],
            "avg_payment_span_0_3m": [25.8666666666667, 37.5714285714286, 13.0],
            "merchant_category": [
                "Diversified entertainment",
                "Diversified entertainment",
                "Diversified entertainment",
            ],
            "merchant_group": ["Entertainment", "Entertainment", "Entertainment"],
            "has_paid": [True, True, True],
            "max_paid_inv_0_12m": [8655.0, 6075.0, 18385.0],
            "max_paid_inv_0_24m": [9645.0, 9090.0, 18385.0],
            "name_in_email": ["F", "Nick", "F+L"],
            "num_active_div_by_paid_inv_0_12m": [
                0.0833333333333333,
                0.818181818181818,
                0.225,
            ],
            "num_active_inv": [20, 9, 9],
            "num_arch_dc_0_12m": [0, 0, 0],
            "num_arch_dc_12_24m": [0, 0, 0],
            "num_arch_ok_0_12m": [215, 3, 33],
            "num_arch_ok_12_24m": [257, 2, 45],
            "num_arch_rem_0_12m": [0, 3, 1],
            "num_arch_written_off_0_12m": [0.0, 0.0, 0.0],
            "num_arch_written_off_12_24m": [0.0, 0.0, 0.0],
            "num_unpaid_bills": [37, 23, 9],
            "status_last_archived_0_24m": [1, 1, 1],
            "status_2nd_last_archived_0_24m": [1, 2, 1],
            "status_3rd_last_archived_0_24m": [1, 2, 1],
            "status_max_archived_0_6_months": [1, 2, 2],
            "status_max_archived_0_12_months": [1, 2, 2],
            "status_max_archived_0_24_months": [1, 2, 2],
            "recovery_debt": [0, 0, 0],
            "sum_capital_paid_account_0_12m": [42206, 104643, 0],
            "sum_capital_paid_account_12_24m": [35336, 32381, 3190],
            "sum_paid_inv_0_12m": [457257, 24390, 123306],
            "time_hours": [12.1927777777778, 21.4111111111111, 21.6991666666667],
            "worst_status_active_inv": [1.0, 1.0, 2.0],
        }
    )
