from __future__ import annotations

import numpy as np
import pandas as pd

from insiders.config import FEATURE_COLUMNS


def split_purchase_and_returns(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    purchases = transactions.loc[transactions["quantity"] > 0].copy()
    returns = transactions.loc[transactions["quantity"] < 0].copy()
    return purchases, returns


def build_customer_features(transactions: pd.DataFrame, drop_missing: bool = True) -> pd.DataFrame:
    """Build one row per customer with the features used by the clustering notebooks."""

    purchases, returns = split_purchase_and_returns(transactions)
    purchases["gross_revenue"] = purchases["quantity"] * purchases["unit_price"]

    reference = transactions[["customer_id"]].drop_duplicates(ignore_index=True)

    aggregated = (
        purchases.groupby("customer_id")
        .agg(
            gross_revenue=("gross_revenue", "sum"),
            last_purchase_date=("invoice_date", "max"),
            qtde_invoices=("invoice_no", "nunique"),
            qtde_itens=("quantity", "sum"),
            qtde_products=("stock_code", "count"),
            avg_ticket=("gross_revenue", "mean"),
            first_purchase_date=("invoice_date", "min"),
        )
        .reset_index()
    )

    max_purchase_date = purchases["invoice_date"].max()
    aggregated["recency_days"] = (max_purchase_date - aggregated["last_purchase_date"]).dt.days
    active_days = (aggregated["last_purchase_date"] - aggregated["first_purchase_date"]).dt.days + 1
    aggregated["frequency"] = np.where(active_days > 0, aggregated["qtde_invoices"] / active_days, 0)

    avg_recency = _calculate_average_recency_days(purchases)
    returns_features = _calculate_returns(returns)
    basket_features = _calculate_basket_features(purchases)

    features = (
        reference.merge(aggregated.drop(columns=["first_purchase_date", "last_purchase_date"]), on="customer_id", how="left")
        .merge(avg_recency, on="customer_id", how="left")
        .merge(returns_features, on="customer_id", how="left")
        .merge(basket_features, on="customer_id", how="left")
    )

    features["qtde_returns"] = features["qtde_returns"].fillna(0)
    features["avg_ticket"] = features["avg_ticket"].round(2)
    features["basket_size"] = features["basket_size"].round(2)
    features["u_basket_size"] = features["u_basket_size"].round(2)

    ordered_columns = ["customer_id", *FEATURE_COLUMNS]
    features = features.loc[:, ordered_columns]
    if drop_missing:
        features = features.dropna().copy()

    return features.reset_index(drop=True)


def _calculate_average_recency_days(purchases: pd.DataFrame) -> pd.DataFrame:
    invoice_dates = (
        purchases[["customer_id", "invoice_no", "invoice_date"]]
        .drop_duplicates()
        .sort_values(["customer_id", "invoice_no", "invoice_date"])
    )
    invoice_dates["previous_customer_id"] = invoice_dates["customer_id"].shift()
    invoice_dates["previous_date"] = invoice_dates["invoice_date"].shift()
    invoice_dates["avg_recency_days"] = np.where(
        invoice_dates["customer_id"].eq(invoice_dates["previous_customer_id"]),
        (invoice_dates["invoice_date"] - invoice_dates["previous_date"]).dt.days,
        np.nan,
    )

    return (
        invoice_dates.dropna(subset=["avg_recency_days"])
        .groupby("customer_id", as_index=False)
        .agg(avg_recency_days=("avg_recency_days", "mean"))
    )


def _calculate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame(columns=["customer_id", "qtde_returns"])

    result = (
        returns.groupby("customer_id", as_index=False)
        .agg(qtde_returns=("quantity", "sum"))
    )
    result["qtde_returns"] = result["qtde_returns"] * -1
    return result


def _calculate_basket_features(purchases: pd.DataFrame) -> pd.DataFrame:
    basket = (
        purchases.groupby("customer_id")
        .agg(
            itens=("quantity", "sum"),
            unique_items=("stock_code", "count"),
            compras=("invoice_no", "nunique"),
        )
        .reset_index()
    )
    basket["basket_size"] = basket["itens"] / basket["compras"]
    basket["u_basket_size"] = basket["unique_items"] / basket["compras"]
    return basket[["customer_id", "basket_size", "u_basket_size"]]

