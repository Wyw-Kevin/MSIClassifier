#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train hierarchical TOSICA migration models from GSE178341."""

from __future__ import annotations

import json
import math
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from scipy import sparse
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT = Path("/home/wyw/rstudio/MSI")
FINAL = ROOT / "最终模型"
OUT_DIR = FINAL / "Migrate"
TOSICA_ROOT = ROOT / "TOSICA"
tosica_pkg = types.ModuleType("TOSICA")
tosica_pkg.__path__ = [str(TOSICA_ROOT / "TOSICA")]
sys.modules.setdefault("TOSICA", tosica_pkg)

from TOSICA.TOSICA_model import scTrans_model as create_model  # noqa: E402
from TOSICA.train import create_pathway_mask, get_gmt, read_gmt, set_seed  # noqa: E402


FULL_GSE178341 = ROOT / "GSE178341.h5ad"
COMMON_GENE_REFERENCE = (
    ROOT
    / "modle2"
    / "outputs_decode_bulk"
    / "reference_noncompressed_min50"
    / "gse178341_decode_reference_tumor_full_protein_coding.h5ad"
)

MAJOR_COL = "clTopLevel"
SUBTYPE_COL = "cl295v11SubShort"
TRAIN_LABEL_COL = "tosica_training_label"
SUBTYPE_MAJORS = ["TNKILC", "Myeloid", "Strom", "Plasma"]
TARGET_SUBTYPES_BY_MAJOR = {
    "TNKILC": ["cTNI02", "cTNI04", "cTNI05", "cTNI07", "cTNI10", "cTNI11", "cTNI13", "cTNI26"],
    "Myeloid": ["cM01", "cM02", "cM04", "cM05", "cM06"],
    "Strom": ["cS17", "cS18", "cS27", "cS32"],
    "Plasma": ["cP1"],
}
MIN_SUBTYPE_CELLS = 50
ACCURACY_THRESHOLD = 0.95
RANDOM_STATE = 20260723


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 64
    lr: float = 0.001
    lrf: float = 0.01
    max_g: int = 300
    max_gs: int = 120
    n_unannotated: int = 64
    embed_dim: int = 32
    depth: int = 2
    num_heads: int = 4
    test_size: float = 0.2
    major_cap_per_class: int = 5000
    subtype_cap_per_class: int | None = None
    gmt_path: str = "human_gobp"
    balanced_sampler: bool = True


CONFIG = TrainConfig()


class SparseExpressionDataset(Dataset):
    def __init__(self, x, y: np.ndarray, obs_names: np.ndarray):
        if sparse.issparse(x):
            self.x = x.tocsr()
            self.is_sparse = True
        else:
            self.x = np.asarray(x, dtype=np.float32)
            self.is_sparse = False
        self.y = np.asarray(y, dtype=np.int64)
        self.obs_names = np.asarray(obs_names, dtype=str)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int):
        if self.is_sparse:
            row = self.x[index].toarray().ravel().astype(np.float32, copy=False)
        else:
            row = self.x[index].astype(np.float32, copy=False)
        return torch.from_numpy(row), torch.tensor(self.y[index], dtype=torch.long), self.obs_names[index]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "models").mkdir(exist_ok=True)
    (OUT_DIR / "tables").mkdir(exist_ok=True)


def common_genes() -> tuple[list[str], list[int]]:
    full = sc.read_h5ad(FULL_GSE178341, backed="r")
    ref = sc.read_h5ad(COMMON_GENE_REFERENCE, backed="r")
    full_var = [str(v) for v in full.var_names]
    first_index = {}
    for idx, gene in enumerate(full_var):
        if gene not in first_index:
            first_index[gene] = idx
    genes = [str(g) for g in ref.var_names]
    missing = [gene for gene in genes if gene not in first_index]
    full.file.close()
    ref.file.close()
    if missing:
        raise ValueError("Common gene reference has genes missing from GSE178341: %s" % missing[:10])
    return genes, [first_index[gene] for gene in genes]


def read_obs() -> pd.DataFrame:
    adata = sc.read_h5ad(FULL_GSE178341, backed="r")
    obs = adata.obs.copy()
    obs["_row_index"] = np.arange(adata.n_obs)
    adata.file.close()
    for col in [MAJOR_COL, SUBTYPE_COL, "sample", "MMR", "MLH1"]:
        if col in obs.columns:
            obs[col] = obs[col].astype(str)
    return obs


def sample_rows(
    obs: pd.DataFrame,
    label_col: str,
    allowed_labels: list[str],
    cap_per_class: int | None,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    summary = []
    for label in allowed_labels:
        idx = obs.index[obs[label_col].astype(str).eq(str(label))].to_numpy()
        n_total = int(idx.shape[0])
        n_take = n_total if cap_per_class is None else min(n_total, int(cap_per_class))
        if n_take <= 0:
            continue
        selected = rng.choice(idx, size=n_take, replace=False)
        rows.append(selected)
        summary.append({"label": label, "n_total": n_total, "n_used": int(n_take)})
    if not rows:
        raise ValueError("No rows selected for %s" % label_col)
    selected_index = np.concatenate(rows)
    rng.shuffle(selected_index)
    out = obs.loc[selected_index, "_row_index"].to_numpy(dtype=int)
    return out, pd.DataFrame(summary)


def load_subset(row_positions: np.ndarray, gene_indices: list[int], gene_names: list[str], label_col: str) -> sc.AnnData:
    adata_backed = sc.read_h5ad(FULL_GSE178341, backed="r")
    adata = adata_backed[row_positions, gene_indices].to_memory()
    adata_backed.file.close()
    adata.var_names = pd.Index(gene_names)
    adata.var_names_make_unique()
    adata.obs[label_col] = adata.obs[label_col].astype(str)
    adata.X = adata.X.astype(np.float32)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def build_pathway_mask(gene_names: list[str], project_dir: Path, config: TrainConfig) -> tuple[np.ndarray, pd.DataFrame]:
    gmt_path = config.gmt_path
    if ".gmt" not in gmt_path:
        gmt_path = str(get_gmt(gmt_path))
    pathway_dict = read_gmt(gmt_path, min_g=0, max_g=config.max_g)
    mask, pathway = create_pathway_mask(
        feature_list=np.asarray(gene_names),
        dict_pathway=pathway_dict,
        add_missing=config.n_unannotated,
        fully_connected=True,
    )
    keep = np.sum(mask, axis=0) > 4
    mask = mask[:, keep]
    pathway = pathway[keep]
    order = sorted(np.argsort(np.sum(mask, axis=0))[-min(config.max_gs, mask.shape[1]) :])
    mask = mask[:, order]
    pathway = pathway[order]
    np.save(project_dir / "mask.npy", mask)
    pathway_df = pd.DataFrame({"pathway": pathway, "n_genes": np.sum(mask, axis=0).astype(int)})
    pathway_df.to_csv(project_dir / "pathway.csv", index=False)
    pd.DataFrame({"gene": gene_names}).to_csv(project_dir / "genes_used.csv", index=False)
    return mask, pathway_df


def compute_auc(y_true: np.ndarray, probs: np.ndarray, classes: list[str]) -> tuple[float, float, pd.DataFrame]:
    y_true = np.asarray(y_true, dtype=int)
    if len(classes) == 2:
        macro_auc = float(roc_auc_score(y_true, probs[:, 1]))
        weighted_auc = macro_auc
        rows = []
        for class_idx, class_name in enumerate(classes):
            rows.append(
                {
                    "class": class_name,
                    "auc_ovr": float(roc_auc_score((y_true == class_idx).astype(int), probs[:, class_idx])),
                }
            )
        return macro_auc, weighted_auc, pd.DataFrame(rows)

    y_bin = label_binarize(y_true, classes=list(range(len(classes))))
    macro_auc = float(roc_auc_score(y_bin, probs, average="macro"))
    weighted_auc = float(roc_auc_score(y_bin, probs, average="weighted"))
    rows = []
    for class_idx, class_name in enumerate(classes):
        rows.append(
            {
                "class": class_name,
                "auc_ovr": float(roc_auc_score(y_bin[:, class_idx], probs[:, class_idx])),
            }
        )
    return macro_auc, weighted_auc, pd.DataFrame(rows)


def run_one_epoch(model, loader, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(train)
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        _, logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        pred = torch.argmax(logits, dim=1)
        loss_sum += float(loss.detach().cpu()) * int(y.numel())
        correct += int(torch.eq(pred, y).sum().detach().cpu())
        total += int(y.numel())
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def predict_loader(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    labels = []
    obs_names = []
    for x, y, names in loader:
        x = x.to(device, non_blocking=True)
        _, logits, _ = model(x)
        probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        labels.append(y.numpy())
        obs_names.extend([str(n) for n in names])
    return np.concatenate(labels), np.vstack(probs), np.asarray(obs_names, dtype=str)


def train_tosica_model(
    model_name: str,
    adata: sc.AnnData,
    label_col: str,
    project_dir: Path,
    config: TrainConfig,
    seed: int = RANDOM_STATE,
) -> dict[str, object]:
    project_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    labels = adata.obs[label_col].astype(str).to_numpy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    classes = [str(c) for c in encoder.classes_]
    pd.DataFrame({label_col: classes}).to_csv(project_dir / "label_dictionary.csv")

    train_idx, valid_idx = train_test_split(
        np.arange(adata.n_obs),
        test_size=config.test_size,
        random_state=seed,
        stratify=y,
    )
    mask, pathway_df = build_pathway_mask(list(map(str, adata.var_names)), project_dir, config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(
        num_classes=len(classes),
        num_genes=adata.n_vars,
        mask=mask,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        has_logits=False,
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: ((1 + math.cos(epoch * math.pi / config.epochs)) / 2) * (1 - config.lrf) + config.lrf,
    )

    pin_memory = bool(torch.cuda.is_available())
    train_ds = SparseExpressionDataset(adata.X[train_idx], y[train_idx], adata.obs_names.to_numpy()[train_idx])
    valid_ds = SparseExpressionDataset(adata.X[valid_idx], y[valid_idx], adata.obs_names.to_numpy()[valid_idx])
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    if config.balanced_sampler:
        class_counts = np.bincount(y[train_idx], minlength=len(classes)).astype(np.float64)
        sample_weights = 1.0 / np.maximum(class_counts[y[train_idx]], 1.0)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=int(train_idx.shape[0]),
            replacement=True,
            generator=loader_generator,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            sampler=sampler,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            generator=loader_generator,
        )
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False, pin_memory=pin_memory)

    metric_rows = []
    best_acc = -np.inf
    best_auc_at_best_acc = -np.inf
    best_epoch = -1
    best_path = project_dir / "model_best.pth"
    start = time.time()
    for epoch in range(config.epochs):
        train_loss, train_acc = run_one_epoch(model, train_loader, optimizer, device, train=True)
        scheduler.step()
        val_loss, val_acc = run_one_epoch(model, valid_loader, optimizer, device, train=False)
        val_y, val_probs, _ = predict_loader(model, valid_loader, device)
        macro_auc, weighted_auc, _ = compute_auc(val_y, val_probs, classes)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_error_rate": 1.0 - train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_error_rate": 1.0 - val_acc,
            "val_auc_macro_ovr": macro_auc,
            "val_auc_weighted_ovr": weighted_auc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        metric_rows.append(row)
        print(
            "[%s epoch %02d] train_acc=%.4f val_acc=%.4f val_auc_macro=%.4f"
            % (model_name, epoch, train_acc, val_acc, macro_auc),
            flush=True,
        )
        if (val_acc > best_acc) or (np.isclose(val_acc, best_acc) and macro_auc > best_auc_at_best_acc):
            best_acc = val_acc
            best_auc_at_best_acc = macro_auc
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

    final_path = project_dir / "model_final.pth"
    torch.save(model.state_dict(), final_path)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(project_dir / "training_metrics.csv", index=False)

    model.load_state_dict(torch.load(best_path, map_location=device))
    val_y, val_probs, val_names = predict_loader(model, valid_loader, device)
    pred_idx = np.argmax(val_probs, axis=1)
    pred_label = np.asarray(classes, dtype=object)[pred_idx]
    true_label = np.asarray(classes, dtype=object)[val_y]
    pred_prob = np.max(val_probs, axis=1)
    macro_auc, weighted_auc, per_class_auc = compute_auc(val_y, val_probs, classes)
    per_class_auc.to_csv(project_dir / "validation_per_class_auc.csv", index=False)
    pred_table = pd.DataFrame(
        {
            "cell_id": val_names,
            "true_label": true_label,
            "predicted_label": pred_label,
            "probability": pred_prob,
        }
    )
    for idx, class_name in enumerate(classes):
        pred_table["prob_%s" % class_name] = val_probs[:, idx]
    pred_table.to_csv(project_dir / "validation_predictions.csv", index=False)
    cm = pd.DataFrame(confusion_matrix(true_label, pred_label, labels=classes), index=classes, columns=classes)
    cm.to_csv(project_dir / "validation_confusion_matrix.csv")
    precision, recall, f1, support = precision_recall_fscore_support(
        true_label,
        pred_label,
        labels=classes,
        zero_division=0,
    )
    per_class_metrics = pd.DataFrame(
        {
            "class": classes,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )
    per_class_metrics.to_csv(project_dir / "validation_per_class_metrics.csv", index=False)
    val_acc = float(accuracy_score(true_label, pred_label))
    summary = {
        "model_name": model_name,
        "label_col": label_col,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_classes": int(len(classes)),
        "classes": classes,
        "n_pathways": int(pathway_df.shape[0]),
        "best_epoch": int(best_epoch),
        "best_val_auc_macro_ovr": float(macro_auc),
        "best_val_auc_weighted_ovr": float(weighted_auc),
        "best_val_acc": val_acc,
        "best_val_error_rate": float(1.0 - val_acc),
        "best_val_macro_recall": float(per_class_metrics["recall"].mean()),
        "best_val_min_recall": float(per_class_metrics["recall"].min()),
        "selection_metric": "validation_accuracy",
        "accuracy_threshold": ACCURACY_THRESHOLD,
        "usable": bool(val_acc > ACCURACY_THRESHOLD),
        "model_weight_path": str(best_path),
        "training_seconds": float(time.time() - start),
        "seed": int(seed),
        "config": config.__dict__,
    }
    (project_dir / "model_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def eligible_subtypes(obs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for major in SUBTYPE_MAJORS:
        sub = obs[obs[MAJOR_COL].astype(str).eq(major)].copy()
        counts = sub[SUBTYPE_COL].astype(str).value_counts()
        for subtype, count in counts.items():
            rows.append(
                {
                    "major": major,
                    "subtype": str(subtype),
                    "n_cells": int(count),
                    "eligible": bool(count > MIN_SUBTYPE_CELLS),
                    "target_for_final_model": str(subtype) in TARGET_SUBTYPES_BY_MAJOR.get(major, []),
                }
            )
    return pd.DataFrame(rows).sort_values(["major", "n_cells"], ascending=[True, False])


def write_readme(status: pd.DataFrame, subtype_table: pd.DataFrame) -> None:
    usable = status[status["usable"].astype(bool)].shape[0]
    total = status.shape[0]
    text = [
        "# Hierarchical TOSICA Migration Models",
        "",
        "Reference: GSE178341 full single-cell h5ad, restricted to the common 16,496 protein-coding genes used by the existing reference.",
        "",
        "Workflow:",
        "1. Train one major model on `clTopLevel`.",
        "2. Train subtype models for TNKILC, Myeloid, Strom, and Plasma.",
        "3. Subtype models keep the subtype features used by the final scRNA model and merge all other cells in the same major class as `Other`.",
        "4. Training uses balanced sampling to reduce majority-class domination.",
        "5. A model is marked usable only if internal validation accuracy is greater than 0.95.",
        "",
        "Usable models: %d/%d" % (usable, total),
        "",
        "Key files:",
        "- `tables/model_use_status.csv`",
        "- `tables/eligible_subtypes_by_major.csv`",
        "- `models/<model_name>/training_metrics.csv`",
        "- `models/<model_name>/validation_predictions.csv`",
        "- `models/<model_name>/validation_confusion_matrix.csv`",
        "- `models/<model_name>/model_best.pth`",
        "",
        "Model status:",
        status[
            [
                "model_name",
                "n_classes",
                "best_val_acc",
                "best_val_macro_recall",
                "best_val_min_recall",
                "best_val_auc_macro_ovr",
                "usable",
            ]
        ].to_string(index=False),
        "",
        "Target subtype counts:",
        subtype_table[subtype_table["target_for_final_model"]].groupby("major")["subtype"].count().to_string(),
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    genes, gene_indices = common_genes()
    obs = read_obs()
    subtype_table = eligible_subtypes(obs)
    subtype_table.to_csv(OUT_DIR / "tables" / "eligible_subtypes_by_major.csv", index=False)
    major_counts = obs[MAJOR_COL].astype(str).value_counts()
    major_labels = major_counts[major_counts > MIN_SUBTYPE_CELLS].index.astype(str).tolist()
    major_rows, major_sample_summary = sample_rows(
        obs,
        label_col=MAJOR_COL,
        allowed_labels=major_labels,
        cap_per_class=CONFIG.major_cap_per_class,
        seed=RANDOM_STATE,
    )
    major_sample_summary.to_csv(OUT_DIR / "tables" / "major_training_sampling.csv", index=False)
    major_adata = load_subset(major_rows, gene_indices, genes, MAJOR_COL)
    summaries = [
        train_tosica_model(
            "tosica_major",
            major_adata,
            MAJOR_COL,
            OUT_DIR / "models" / "tosica_major",
            CONFIG,
            seed=RANDOM_STATE,
        )
    ]
    del major_adata
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for major in SUBTYPE_MAJORS:
        targets = TARGET_SUBTYPES_BY_MAJOR[major]
        allowed = [subtype for subtype in targets if subtype in set(subtype_table["subtype"].astype(str))]
        allowed.append("Other")
        model_obs = obs[obs[MAJOR_COL].astype(str).eq(major)].copy()
        model_obs[TRAIN_LABEL_COL] = np.where(
            model_obs[SUBTYPE_COL].astype(str).isin(targets),
            model_obs[SUBTYPE_COL].astype(str),
            "Other",
        )
        rows, sample_summary = sample_rows(
            model_obs,
            label_col=TRAIN_LABEL_COL,
            allowed_labels=allowed,
            cap_per_class=CONFIG.subtype_cap_per_class,
            seed=RANDOM_STATE + len(summaries),
        )
        model_dir = OUT_DIR / "models" / ("tosica_subtype_%s" % major)
        sample_summary.to_csv(model_dir.parent / ("%s_training_sampling.csv" % model_dir.name), index=False)
        subtype_adata = load_subset(rows, gene_indices, genes, SUBTYPE_COL)
        subtype_adata.obs[TRAIN_LABEL_COL] = model_obs.loc[subtype_adata.obs_names, TRAIN_LABEL_COL].astype(str).to_numpy()
        summaries.append(
            train_tosica_model(
                model_dir.name,
                subtype_adata,
                TRAIN_LABEL_COL,
                model_dir,
                CONFIG,
                seed=RANDOM_STATE,
            )
        )
        del subtype_adata
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    status = pd.DataFrame(summaries)
    status.to_csv(OUT_DIR / "tables" / "model_use_status.csv", index=False)
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reference_h5ad": str(FULL_GSE178341),
        "common_gene_reference": str(COMMON_GENE_REFERENCE),
        "output_dir": str(OUT_DIR),
        "accuracy_threshold": ACCURACY_THRESHOLD,
        "selection_metric": "validation_accuracy",
        "subtype_label_strategy": "final_model_targets_plus_other",
        "target_subtypes_by_major": TARGET_SUBTYPES_BY_MAJOR,
        "min_subtype_cells": MIN_SUBTYPE_CELLS,
        "config": CONFIG.__dict__,
        "models": summaries,
    }
    (OUT_DIR / "migrate_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(status, subtype_table)
    print("\nModel use status:")
    print(
        status[
            [
                "model_name",
                "n_classes",
                "best_val_acc",
                "best_val_macro_recall",
                "best_val_min_recall",
                "best_val_auc_macro_ovr",
                "usable",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
