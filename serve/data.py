from typing import Optional
from pydantic import BaseModel


class FeatureStoreDataRequest(BaseModel):
    """Data model for incoming request from a user from a Feature Store, in the ideal scenario.

    Parameters:
        uuid (str): User ID for identification purposes and integration with policies
        max_paid_inv_0_24m (float): Maximum amount of paid invoices during the last 24 months
        avg_payment_span_0_12m (float): Average payment span over the last 12 months
        sum_capital_paid_account_0_12m (float): Total sum of paid amount with account over last 12 months
        time_hours (float): Time of the day for transaction
        recovery_debt (float): Recovery debt amount
        sum_capital_paid_account_12_24m (float): Total sum of paid amount with account from M-24 to M-12
        num_active_div_by_paid_inv_0_12m (float): Number of active invoices over paid ones in the last 12 months
        sum_paid_inv_0_12m (float): Total sum of invoices in the last 12 months
        account_days_in_rem_12_24m (float): Number of days, between M-12 and M-24, that account was in debt
        num_arch_ok_0_12m (float): Number of "ok" transactions archived during the last 12 months
        account_amount_added_12_24m (float): Total amount added to account between M-24 and M-12
        has_paid (bool): Whether or not user has already paid any invoice
        account_status (float): Current account status
        account_worst_status_0_3m (float): Worst account status in the last 3 months
        account_worst_status_3_6m (float): Worst account status between M-6 and M-3
        account_worst_status_6_12m (float): Worst account status between M-12 and M-6
        account_worst_status_12_24m (float): Worst account status between M-24 and M-12
        status_last_archived_0_24m (float): Status at last time invoice was archived, over last 24 months
        status_2nd_last_archived_0_24m (float): Status at 2nd last time invoice was archived, over last 24 months
        status_3rd_last_archived_0_24m (float): Status at 3rd last time invoice was archived, over last 24 months
        status_max_archived_0_6_months (float): Worst account status over last 6 months, when any invoice was archived
        status_max_archived_0_12_months (float): Worst account status over last 12 months, when any invoice was archived
        status_max_archived_0_24_months (float): Worst account status over last 24 months, when any invoice was archived
        merchant_group (str): Group to which merchant belongs
    """

    uuid: str
    max_paid_inv_0_24m: Optional[float] = None
    avg_payment_span_0_12m: Optional[float] = None
    sum_capital_paid_account_0_12m: Optional[float] = None
    time_hours: Optional[float] = None
    recovery_debt: Optional[float] = None
    sum_capital_paid_account_12_24m: Optional[float] = None
    num_active_div_by_paid_inv_0_12m: Optional[float] = None
    sum_paid_inv_0_12m: Optional[float] = None
    account_days_in_rem_12_24m: Optional[float] = None
    num_arch_ok_0_12m: Optional[float] = None
    account_amount_added_12_24m: Optional[float] = None
    has_paid: Optional[bool] = None
    account_status: Optional[float] = None
    account_worst_status_0_3m: Optional[float] = None
    account_worst_status_3_6m: Optional[float] = None
    account_worst_status_6_12m: Optional[float] = None
    account_worst_status_12_24m: Optional[float] = None
    status_last_archived_0_24m: Optional[float] = None
    status_2nd_last_archived_0_24m: Optional[float] = None
    status_3rd_last_archived_0_24m: Optional[float] = None
    status_max_archived_0_6_months: Optional[float] = None
    status_max_archived_0_12_months: Optional[float] = None
    status_max_archived_0_24_months: Optional[float] = None
    merchant_group: Optional[str] = None
