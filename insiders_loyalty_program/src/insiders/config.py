from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "Ecommerce" / "Ecommerce.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_COLUMNS = [
    "invoice_no",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "unit_price",
    "customer_id",
    "country",
]

EXCLUDED_STOCK_CODES = {
    "POST",
    "D",
    "DOT",
    "M",
    "S",
    "AMAZONFEE",
    "m",
    "DCGSSBOY",
    "DCGSSGIRL",
    "PADS",
    "B",
    "CRUK",
}

EXCLUDED_COUNTRIES = {"European Community", "Unspecified"}
EXCLUDED_CUSTOMER_IDS = {16446}

MIN_UNIT_PRICE = 0.04
MISSING_CUSTOMER_ID_START = 19000

FEATURE_COLUMNS = [
    "gross_revenue",
    "recency_days",
    "qtde_invoices",
    "qtde_itens",
    "qtde_products",
    "avg_ticket",
    "avg_recency_days",
    "frequency",
    "qtde_returns",
    "basket_size",
    "u_basket_size",
]


@dataclass(frozen=True)
class TrainingConfig:
    """Default model choices taken from the final modeling notebook."""

    algorithm: str = "gmm"
    n_clusters: int = 9
    random_state: int = 42

