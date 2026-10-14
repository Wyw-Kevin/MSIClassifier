"""Helpers for packaged read-only entry points; no classifier fitting."""
from pathlib import Path
import os

def project_root(value=None):
    return Path(value or os.environ.get('MSI_PROJECT_ROOT') or Path(__file__).resolve().parent/'project').expanduser().resolve()

def final_root(value=None):
    return project_root(value)/'最终模型'

def artifact_path(value=None):
    return final_root(value)/'model/reference/shared18_weighted_mlh1_2026/model.joblib'
