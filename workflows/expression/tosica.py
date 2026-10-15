"""Frozen hierarchical annotation using exact archived gene orders and masks."""
from pathlib import Path
import json
import sys
import types
import numpy as np
import pandas as pd


def labels(folder):
    return pd.read_csv(Path(folder) / "label_dictionary.csv", index_col=0).iloc[:, 0].astype(str).tolist()


def genes(folder):
    folder = Path(folder)
    if (folder / "train_genes.csv").exists():
        values = pd.read_csv(folder / "train_genes.csv", header=None).iloc[:, 0].astype(str).tolist()
        return values[1:] if values[0].lower() in {"gene", "genes"} else values
    return pd.read_csv(folder / "genes_used.csv").iloc[:, 0].astype(str).tolist()


def predict(a, rows, folder, weight, source, device, batch_size):
    import torch
    folder = Path(folder)
    pkg = types.ModuleType("TOSICA")
    pkg.__path__ = [str(Path(source) / "TOSICA")]
    sys.modules.setdefault("TOSICA", pkg)
    from TOSICA.TOSICA_model import scTrans_model
    names = genes(folder); classes = labels(folder)
    cfg = {"embed_dim": 48, "depth": 2, "num_heads": 4}
    meta = folder / "model_manifest.json"
    if meta.exists():
        cfg.update({k: v for k, v in json.loads(meta.read_text()).get("config", {}).items() if k in cfg})
    model = scTrans_model(num_classes=len(classes), num_genes=len(names), mask=np.load(folder / "mask.npy"),
                          has_logits=False, **cfg)
    model.load_state_dict(torch.load(folder / weight, map_location="cpu", weights_only=True))
    model.to(device).eval()
    positions = a.var_names.get_indexer(names)
    present = np.flatnonzero(positions >= 0)
    if len(present) == 0:
        raise ValueError("No frozen TOSICA input genes are measured")
    result = []; probabilities = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start + batch_size]
            matrix = np.zeros((len(selected), len(names)), dtype=np.float32)
            matrix[:, present] = a[selected, positions[present]].X.toarray()
            z = model.feature_embed(torch.from_numpy(matrix).to(device))
            z = torch.cat((model.cls_token.expand(z.shape[0], -1, -1), z), dim=1)
            for block in model.blocks:
                z, _ = block(z)
            logits = model.head(model.pre_logits(model.norm(z)[:, 0]))
            values, indices = torch.softmax(logits, dim=1).max(dim=1)
            result.extend(np.array(classes, dtype=object)[indices.cpu().numpy()].tolist())
            probabilities.extend(values.cpu().numpy().tolist())
    info = {"folder": str(folder), "weight": weight, "n_measured_genes": len(present), "n_model_genes": len(names),
            "missing_genes": [name for name, pos in zip(names, positions) if pos < 0],
            "missing_gene_policy": "exact historical gene-order alignment; missing model genes are zero-padded and reported; subtype coverage is checked separately"}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.array(result), np.array(probabilities), info
