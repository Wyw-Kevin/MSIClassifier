"""Convenience pipeline for scRNA migration preprocessing and MSI prediction."""

import argparse
import json
from pathlib import Path

from .predict import DEFAULT_MODEL_DIR, predict_scrna_processed_file
from .preprocess import DEFAULT_INPUT_ROOT, DEFAULT_REFERENCE_DIR, DEFAULT_RESULT_ROOT, process_scrna_feature_table, process_scrna_h5ad


def run_scrna_pipeline(
    dataset,
    h5ad_path=None,
    feature_table=None,
    input_root=DEFAULT_INPUT_ROOT,
    result_root=DEFAULT_RESULT_ROOT,
    reference_dir=DEFAULT_REFERENCE_DIR,
    model_dir=DEFAULT_MODEL_DIR,
    selected_features=None,
    known_features_path=None,
    reference_mlh1_path=None,
    reference_mlh1_fallback_path=None,
    major_cells_path=None,
    batch_size=1024,
    limit_cells=None,
    device="cuda:0",
    sample_col=None,
    tumor_col="Class",
    tumor_value="Tumor",
    prefix=None,
    h5ad_backed="r",
    use_raw=False,
):
    if feature_table is not None:
        processed_path = process_scrna_feature_table(
            dataset,
            feature_table,
            result_root=result_root,
            selected_features=selected_features,
        )
    elif h5ad_path is not None:
        processed_path = process_scrna_h5ad(
            dataset,
            h5ad_path,
            input_root=input_root,
            result_root=result_root,
            reference_dir=reference_dir,
            selected_features=selected_features,
            known_features_path=known_features_path,
            reference_mlh1_path=reference_mlh1_path,
            reference_mlh1_fallback_path=reference_mlh1_fallback_path,
            major_cells_path=major_cells_path,
            batch_size=batch_size,
            limit_cells=limit_cells,
            device=device,
            sample_col=sample_col,
            tumor_col=tumor_col,
            tumor_value=tumor_value,
            prefix=prefix,
            h5ad_backed=h5ad_backed,
            use_raw=use_raw,
        )
    else:
        raise ValueError("Either h5ad_path or feature_table is required.")
    prediction_path = predict_scrna_processed_file(
        processed_path,
        model_dir=model_dir,
        output_dir=Path(result_root) / "prediction",
    )
    return {"processed_path": processed_path, "prediction_path": prediction_path}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--h5ad", type=Path, default=None)
    parser.add_argument("--feature-table", type=Path, default=None)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--selected-features", type=Path, default=None)
    parser.add_argument("--known-features-path", type=Path, default=None)
    parser.add_argument("--reference-mlh1-path", type=Path, default=None)
    parser.add_argument("--reference-mlh1-fallback-path", type=Path, default=None)
    parser.add_argument("--major-cells", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--limit-cells", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-col", default=None)
    parser.add_argument("--tumor-col", default="Class")
    parser.add_argument("--tumor-value", default="Tumor")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--in-memory", action="store_true")
    parser.add_argument("--use-raw", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = run_scrna_pipeline(
        args.dataset,
        h5ad_path=args.h5ad,
        feature_table=args.feature_table,
        input_root=args.input_root,
        result_root=args.result_root,
        reference_dir=args.reference_dir,
        model_dir=args.model_dir,
        selected_features=args.selected_features,
        known_features_path=args.known_features_path,
        reference_mlh1_path=args.reference_mlh1_path,
        reference_mlh1_fallback_path=args.reference_mlh1_fallback_path,
        major_cells_path=args.major_cells,
        batch_size=args.batch_size,
        limit_cells=args.limit_cells,
        device=args.device,
        sample_col=args.sample_col,
        tumor_col=args.tumor_col,
        tumor_value=args.tumor_value,
        prefix=args.prefix,
        h5ad_backed=None if args.in_memory else "r",
        use_raw=args.use_raw,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
