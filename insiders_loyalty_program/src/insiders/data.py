from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from insiders.config import (
    EXCLUDED_COUNTRIES,
    EXCLUDED_CUSTOMER_IDS,
    EXCLUDED_STOCK_CODES,
    MIN_UNIT_PRICE,
    MISSING_CUSTOMER_ID_START,
    RAW_COLUMNS,
    RAW_DATA_PATH,
)


def load_transactions(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw e-commerce transactions and normalize column names."""

    df = pd.read_csv(path, encoding="unicode_escape")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    if len(df.columns) != len(RAW_COLUMNS):
        raise ValueError(f"Expected {len(RAW_COLUMNS)} columns, found {len(df.columns)}")

    df.columns = RAW_COLUMNS
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], format="%d-%b-%y", errors="coerce")
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    return df


def assign_missing_customer_ids(
    transactions: pd.DataFrame,
    start_id: int = MISSING_CUSTOMER_ID_START,
) -> pd.DataFrame:
    """Create deterministic synthetic customer IDs for invoices without one."""

    df = transactions.copy()
    missing_invoices = df.loc[df["customer_id"].isna(), "invoice_no"].drop_duplicates()
    synthetic_ids = pd.DataFrame(
        {
            "invoice_no": missing_invoices.to_numpy(),
            "synthetic_customer_id": np.arange(start_id, start_id + len(missing_invoices)),
        }
    )

    df = df.merge(synthetic_ids, on="invoice_no", how="left")
    df["customer_id"] = df["customer_id"].combine_first(df["synthetic_customer_id"])
    return df.drop(columns=["synthetic_customer_id"])


def clean_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Apply the business filters used before feature engineering."""

    df = assign_missing_customer_ids(transactions)
    df = df.dropna(subset=["invoice_no", "stock_code", "quantity", "invoice_date", "unit_price", "customer_id"])
    df = df.loc[df["unit_price"] >= MIN_UNIT_PRICE].copy()
    df = df.loc[~df["customer_id"].isin(EXCLUDED_CUSTOMER_IDS)].copy()
    df = df.loc[~df["stock_code"].isin(EXCLUDED_STOCK_CODES)].copy()
    df = df.loc[~df["country"].isin(EXCLUDED_COUNTRIES)].copy()
    return df.drop(columns=["description"], errors="ignore")


def load_clean_transactions(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw data and apply cleaning in one call."""

    return clean_transactions(load_transactions(path))

