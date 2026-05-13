from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from insiders.config import FEATURE_COLUMNS, TrainingConfig


def build_training_matrix(features: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    columns = feature_columns or FEATURE_COLUMNS
    missing = set(columns) - set(features.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    return features.loc[:, columns].copy()


def evaluate_cluster_counts(
    features: pd.DataFrame,
    clusters: range | list[int] = range(2, 25),
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare KMeans, GMM and hierarchical clustering with silhouette score."""

    X = MinMaxScaler().fit_transform(build_training_matrix(features))
    rows = []

    for n_clusters in clusters:
        kmeans_labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit_predict(X)
        gmm_labels = GaussianMixture(n_components=n_clusters, random_state=random_state).fit_predict(X)
        hierarchical_model = hierarchy.linkage(X, method="ward")
        hierarchical_labels = hierarchy.fcluster(hierarchical_model, n_clusters, criterion="maxclust")

        rows.append(
            {
                "n_clusters": n_clusters,
                "kmeans_silhouette": silhouette_score(X, kmeans_labels),
                "gmm_silhouette": silhouette_score(X, gmm_labels),
                "hierarchical_silhouette": silhouette_score(X, hierarchical_labels),
            }
        )

    return pd.DataFrame(rows)


def train_clustering_model(
    features: pd.DataFrame,
    config: TrainingConfig = TrainingConfig(),
) -> tuple[Pipeline, pd.Series, float]:
    """Train the final clustering model and return labels plus silhouette score."""

    estimator = _build_estimator(config)
    pipeline = Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
            ("model", estimator),
        ]
    )

    X = build_training_matrix(features)
    labels = pipeline.fit_predict(X)
    score = silhouette_score(pipeline.named_steps["scaler"].transform(X), labels)
    return pipeline, pd.Series(labels, index=features.index, name="cluster"), score


def save_model(model: Pipeline, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(model, file)


def load_model(path: str | Path) -> Pipeline:
    with Path(path).open("rb") as file:
        return pickle.load(file)


def _build_estimator(config: TrainingConfig):
    if config.algorithm == "gmm":
        return GaussianMixture(n_components=config.n_clusters, random_state=config.random_state)
    if config.algorithm == "kmeans":
        return KMeans(n_clusters=config.n_clusters, n_init=10, random_state=config.random_state)

    raise ValueError(f"Unsupported algorithm: {config.algorithm}")

