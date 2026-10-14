"""Resolve the optional full project root, defaulting to the bundled portable tree."""
from pathlib import Path
import os
def project_root():
    value = os.environ.get("MSI_PROJECT_ROOT")
    return Path(value).expanduser().resolve() if value else Path(__file__).resolve().parent
