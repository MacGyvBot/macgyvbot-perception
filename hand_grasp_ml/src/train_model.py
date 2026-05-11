from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from feature_extractor import FEATURE_COUNT


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "hand_grasp_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "hand_grasp_model.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42
ACTIVE_LABELS = {"open", "grasp"}
MIN_RECOMMENDED_SAMPLES = 400


def load_dataset() -> tuple[pd.DataFrame, pd.Series] | None:
    if not DATASET_PATH.exists():
        print(
            f"Error: Dataset not found at {DATASET_PATH}. "
            "Run python src/collect_dataset.py first."
        )
        return None

    try:
        dataset = pd.read_csv(DATASET_PATH, header=None)
    except EmptyDataError:
        print(
            f"Error: Dataset file is empty at {DATASET_PATH}. "
            "Collect samples before training."
        )
        return None

    expected_columns = FEATURE_COUNT + 1
    if dataset.shape[1] != expected_columns:
        print(
            f"Error: Expected {expected_columns} columns "
            f"({FEATURE_COUNT} features + label), got {dataset.shape[1]}."
        )
        return None

    x = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1].astype(str)

    inactive_mask = ~y.isin(ACTIVE_LABELS)
    if inactive_mask.any():
        skipped_counts = y[inactive_mask].value_counts().to_dict()
        print(f"Warning: Ignoring inactive labels during training: {skipped_counts}")
        dataset = dataset.loc[~inactive_mask].reset_index(drop=True)
        x = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1].astype(str)

    if len(dataset) < MIN_RECOMMENDED_SAMPLES:
        print(
            f"Warning: Only {len(dataset)} active samples found. "
            "Recommended baseline is at least 200 samples per class."
        )

    class_counts = y.value_counts()
    if len(class_counts) < 2:
        print("Error: Need at least 2 classes to train a classifier.")
        return None

    if class_counts.min() < 2:
        print(
            "Error: Each class needs at least 2 samples for stratified "
            f"train/test split. Current counts: {class_counts.to_dict()}"
        )
        return None

    return x, y


def train_model(x: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    x_values = x.astype(float).to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        x_values,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=sorted(y.unique())))
    print(f"Label order: {sorted(y.unique())}")

    return model


def save_model(model: RandomForestClassifier) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


def main() -> int:
    loaded = load_dataset()
    if loaded is None:
        return 1

    x, y = loaded
    model = train_model(x, y)
    save_model(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
