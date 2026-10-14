"""Reusable scRNA MSI/MSS preprocessing and prediction helpers."""

# Packaging-only path resolution; archived originals retain exact source bytes.
from pathlib import Path as _MSIPackagePath
import sys as _MSIPackageSys
for _MSIPackageParent in _MSIPackagePath(__file__).resolve().parents:
    if (_MSIPackageParent / "project_paths.py").is_file():
        _MSIPackageSys.path.insert(0, str(_MSIPackageParent))
        break
from project_paths import project_root

__all__ = [
    "process_scrna_h5ad",
    "process_scrna_feature_table",
    "predict_scrna_processed_file",
    "run_scrna_pipeline",
]


def process_scrna_h5ad(*args, **kwargs):
    from .preprocess import process_scrna_h5ad as _process_scrna_h5ad

    return _process_scrna_h5ad(*args, **kwargs)


def process_scrna_feature_table(*args, **kwargs):
    from .preprocess import process_scrna_feature_table as _process_scrna_feature_table

    return _process_scrna_feature_table(*args, **kwargs)


def predict_scrna_processed_file(*args, **kwargs):
    from .predict import predict_scrna_processed_file as _predict_scrna_processed_file

    return _predict_scrna_processed_file(*args, **kwargs)


def run_scrna_pipeline(*args, **kwargs):
    from .pipeline import run_scrna_pipeline as _run_scrna_pipeline

    return _run_scrna_pipeline(*args, **kwargs)


def __getattr__(name):
    if name in __all__:
        return globals()[name]
    raise AttributeError("module 'scRNA_msi' has no attribute %r" % name)
