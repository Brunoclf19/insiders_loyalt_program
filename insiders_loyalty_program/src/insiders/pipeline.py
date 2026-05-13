from __future__ import annotations

import argparse
from pathlib import Path

from insiders.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_PATH, TrainingConfig
from insiders.data import load_clean_transactions
from insiders.features import build_customer_features
from insiders.modeling import evaluate_cluster_counts, save_model, train_clustering_model


def run_training_pipeline(
    input_path: str | Path = RAW_DATA_PATH,
    processed_dir: str | Path = PROCESSED_DATA_DIR,
    models_dir: str | Path = MODELS_DIR,
    config: TrainingConfig = TrainingConfig(),
) -> dict[str, Path | float]:
    processed_dir = Path(processed_dir)
    models_dir = Path(models_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    transactions = load_clean_transactions(input_path)
    features = build_customer_features(transactions)
    scores = evaluate_cluster_counts(features, random_state=config.random_state)
    model, labels, silhouette = train_clustering_model(features, config=config)

    clustered = features.copy()
    clustered["cluster"] = labels

    features_path = processed_dir / "customer_features.csv"
    scores_path = processed_dir / "cluster_scores.csv"
    clustered_path = processed_dir / "clustered_customers.csv"
    model_path = models_dir / "clustering_model.pkl"

    features.to_csv(features_path, index=False)
    scores.to_csv(scores_path, index=False)
    clustered.to_csv(clustered_path, index=False)
    save_model(model, model_path)

    return {
        "features_path": features_path,
        "scores_path": scores_path,
        "clustered_path": clustered_path,
        "model_path": model_path,
        "silhouette": float(silhouette),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Insiders customer clustering pipeline.")
    parser.add_argument("--input", default=str(RAW_DATA_PATH), help="Path to raw Ecommerce.csv")
    parser.add_argument("--processed-dir", default=str(PROCESSED_DATA_DIR), help="Directory for generated CSV files")
    parser.add_argument("--models-dir", default=str(MODELS_DIR), help="Directory for generated model artifacts")
    parser.add_argument("--algorithm", default="gmm", choices=["gmm", "kmeans"], help="Clustering algorithm")
    parser.add_argument("--clusters", default=9, type=int, help="Number of clusters for final model")
    parser.add_argument("--random-state", default=42, type=int, help="Random seed for reproducible training")
    args = parser.parse_args()

    result = run_training_pipeline(
        input_path=args.input,
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        config=TrainingConfig(
            algorithm=args.algorithm,
            n_clusters=args.clusters,
            random_state=args.random_state,
        ),
    )

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
