"""Predict MSI/MSS from processed scRNA 18+MLH1 feature files."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("joblib is required to load scRNA model artifacts.") from exc


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FINAL_MODEL_ROOT = PACKAGE_ROOT.parent
DEFAULT_MODEL_DIR = PACKAGE_ROOT / "models" / "scRNA_model"
DEFAULT_RESULT_ROOT = FINAL_MODEL_ROOT / "data" / "scRNA" / "result"
MODEL_CONFIG = "model_config.json"
MODEL_FILENAME = "trained_msi_model.joblib"
PORTABLE_MODEL_FILENAME = "trained_msi_model_portable.json"


class PortableBinaryGradientBoosting(object):
    """Version-stable predictor for the exported binary sklearn GBM trees."""

    def __init__(self, payload):
        if payload.get("model_type") != "binary_gradient_boosting_trees":
            raise ValueError("Unsupported portable model type: %s" % payload.get("model_type"))
        self.classes_ = np.asarray(payload.get("classes", [0, 1]))
        self.learning_rate = float(payload.get("learning_rate", 1.0))
        init_raw_score = np.asarray(payload.get("init_raw_score", 0.0), dtype=float).reshape(-1)
        if init_raw_score.size != 1:
            raise ValueError("Portable binary GBM must contain one init raw score.")
        self.init_raw_score = float(init_raw_score[0])
        self.estimators = list(payload.get("estimators", []))

    @staticmethod
    def _tree_predict(tree, matrix):
        children_left = np.asarray(tree["children_left"], dtype=np.int64)
        children_right = np.asarray(tree["children_right"], dtype=np.int64)
        features = np.asarray(tree["feature"], dtype=np.int64)
        thresholds = np.asarray(tree["threshold"], dtype=float)
        values = np.asarray(tree["value"], dtype=float)
        output = np.empty(matrix.shape[0], dtype=float)
        for row_index in range(matrix.shape[0]):
            node = 0
            while children_left[node] != children_right[node]:
                feature_index = features[node]
                if matrix[row_index, feature_index] <= thresholds[node]:
                    node = children_left[node]
                else:
                    node = children_right[node]
            output[row_index] = values[node]
        return output

    def decision_function(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("Portable GBM expects a 2D feature matrix.")
        raw = np.full(matrix.shape[0], self.init_raw_score, dtype=float)
        for tree in self.estimators:
            raw += self.learning_rate * self._tree_predict(tree, matrix)
        return raw

    def predict_proba(self, matrix):
        raw = self.decision_function(matrix)
        positive = 1.0 / (1.0 + np.exp(-np.clip(raw, -709.0, 709.0)))
        return np.column_stack([1.0 - positive, positive])


def load_portable_model(portable_path):
    portable_path = Path(portable_path)
    if not portable_path.exists():
        raise FileNotFoundError("Missing portable model file: %s" % portable_path)
    payload = json.loads(portable_path.read_text(encoding="utf-8"))
    return PortableBinaryGradientBoosting(payload)


def _is_legacy_gradient_boosting_pickle_error(exc):
    return "sklearn.ensemble._gb_losses" in str(exc)


def load_model_artifacts(model_dir=DEFAULT_MODEL_DIR):
    model_dir = Path(model_dir)
    config_path = model_dir / MODEL_CONFIG
    if not config_path.exists():
        raise FileNotFoundError("Missing model config: %s" % config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_path = model_dir / str(config.get("model_filename", MODEL_FILENAME))
    if not model_path.exists():
        raise FileNotFoundError("Missing model file: %s" % model_path)
    try:
        model = joblib.load(model_path)
    except ModuleNotFoundError as exc:
        if not _is_legacy_gradient_boosting_pickle_error(exc):
            raise
        portable_path = model_dir / str(config.get("portable_model_filename", PORTABLE_MODEL_FILENAME))
        model = load_portable_model(portable_path)
    return model, config


def _scale_group(values, method):
    if method == "dataset_minmax":
        min_values = values.min(axis=0)
        ranges = (values.max(axis=0) - min_values).replace(0, 1.0)
        return (values - min_values) / ranges
    if method == "dataset_zscore":
        means = values.mean(axis=0)
        stds = values.std(axis=0, ddof=0).replace(0, 1.0)
        return (values - means) / stds
    raise ValueError("Unsupported dataset scale method: %s" % method)


def _scale_with_training_minmax(values, features, scale_params):
    if not isinstance(scale_params, dict):
        raise ValueError("training_minmax requires scale_params in the model config.")
    minimum = pd.Series(scale_params.get("min", {}), dtype=float).reindex(features)
    ranges = pd.Series(scale_params.get("range", {}), dtype=float).reindex(features)
    missing = minimum[minimum.isna()].index.tolist() + ranges[ranges.isna()].index.tolist()
    if missing:
        raise ValueError("training_minmax is missing parameters for: %s" % sorted(set(missing)))
    ranges = ranges.replace(0.0, 1.0)
    return (values - minimum) / ranges


def scale_feature_matrix(frame, features, scale_method, dataset_col=None, scale_params=None):
    values = frame[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    if scale_method == "none":
        return values
    if scale_method == "training_minmax":
        return _scale_with_training_minmax(values, features, scale_params).astype(float)
    if scale_method in {"dataset_minmax", "dataset_zscore"}:
        scaled = pd.DataFrame(index=values.index, columns=features, dtype=float)
        if dataset_col and dataset_col in frame.columns:
            for _, index in frame.groupby(dataset_col, sort=False).groups.items():
                scaled.loc[index, features] = _scale_group(values.loc[index, features], scale_method)
        else:
            scaled.loc[:, features] = _scale_group(values, scale_method)
        return scaled.astype(float)
    raise ValueError("Unsupported scale method: %s" % scale_method)


def _resolve_sample_col(frame, configured, override=None):
    for candidate in [override, configured, "sampleID", "SampleID", "sample", "Sample"]:
        if candidate and candidate in frame.columns:
            return str(candidate)
    raise ValueError("Processed feature table must contain a sample column.")


def _predict_positive_probability(model, matrix):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(matrix)
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            positive_index = classes.index(1)
        elif "MSI" in classes:
            positive_index = classes.index("MSI")
        else:
            positive_index = probabilities.shape[1] - 1
        return probabilities[:, positive_index]
    scores = model.decision_function(matrix)
    return 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype=float)))


def _prediction_filename(processed_path):
    stem = Path(processed_path).stem
    if stem.endswith("_processed"):
        stem = stem[: -len("_processed")]
    return "%s_predictions.csv" % stem


def predict_scrna_processed_file(
    processed_path,
    model_dir=DEFAULT_MODEL_DIR,
    output_dir=DEFAULT_RESULT_ROOT / "prediction",
    sample_col=None,
    label_col=None,
    dataset_col=None,
):
    model, config = load_model_artifacts(model_dir)
    frame = pd.read_csv(processed_path)
    features = [str(feature) for feature in config["features"]]
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise ValueError("Processed feature table is missing model columns: %s" % missing)

    sample_col = _resolve_sample_col(frame, str(config.get("sample_col", "sampleID")), sample_col)
    label_col = str(label_col if label_col is not None else config.get("label_col", "MMR"))
    dataset_col = str(dataset_col if dataset_col is not None else config.get("dataset_col", "dataset"))
    if dataset_col not in frame.columns:
        dataset_col = None

    matrix = scale_feature_matrix(
        frame,
        features,
        str(config.get("scale_method", "dataset_minmax")),
        dataset_col=dataset_col,
        scale_params=config.get("scale_params"),
    )
    probabilities = _predict_positive_probability(model, matrix.to_numpy())
    threshold = float(config.get("threshold", 0.5))
    positive_label = str(config.get("positive_label", "MSI"))
    negative_label = str(config.get("negative_label", "MSS"))
    predictions = pd.DataFrame(
        {
            sample_col: frame[sample_col].astype(str).to_numpy(),
            "predicted_MMR": np.where(probabilities >= threshold, positive_label, negative_label),
            "MSI_probability": probabilities,
        }
    )
    if label_col in frame.columns and frame[label_col].notna().any():
        predictions[label_col] = frame[label_col].to_numpy()
    for column in [str(col) for col in config.get("metadata_columns", [])]:
        if column in frame.columns and column not in predictions.columns:
            predictions[column] = frame[column].to_numpy()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _prediction_filename(processed_path)
    predictions.to_csv(output_path, index=False)
    return output_path


def predict_all_processed_files(
    process_dir=DEFAULT_RESULT_ROOT / "process",
    model_dir=DEFAULT_MODEL_DIR,
    output_dir=DEFAULT_RESULT_ROOT / "prediction",
    continue_on_error=True,
):
    rows = []
    for processed_path in sorted(Path(process_dir).glob("*_processed.csv")):
        try:
            prediction_path = predict_scrna_processed_file(processed_path, model_dir=model_dir, output_dir=output_dir)
            rows.append(
                {
                    "dataset": processed_path.stem.replace("_processed", ""),
                    "status": "ok",
                    "prediction_path": str(prediction_path),
                    "error": "",
                }
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            rows.append(
                {
                    "dataset": processed_path.stem.replace("_processed", ""),
                    "status": "failed",
                    "prediction_path": "",
                    "error": "%s: %s" % (type(exc).__name__, exc),
                }
            )
    summary = pd.DataFrame(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "scrna_prediction_summary.csv", index=False)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=None)
    parser.add_argument("--process-dir", type=Path, default=DEFAULT_RESULT_ROOT / "process")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_ROOT / "prediction")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.processed:
        path = predict_scrna_processed_file(args.processed, model_dir=args.model_dir, output_dir=args.output_dir)
        print(json.dumps({"prediction_path": str(path)}, indent=2))
    else:
        summary = predict_all_processed_files(
            process_dir=args.process_dir,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            continue_on_error=not args.fail_fast,
        )
        print(summary.to_json(orient="records", force_ascii=False, indent=2))


if __name__ == "__main__":
    main()
