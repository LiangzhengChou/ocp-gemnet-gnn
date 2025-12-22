"""Utilities for exporting hardness predictions."""

from __future__ import annotations

import csv
import os
from typing import Iterable, List

import numpy as np
import torch
from tqdm import tqdm


def gather_ids(batch_data) -> List[str]:
    if hasattr(batch_data, "sample_id"):
        sample_id = batch_data.sample_id
        if isinstance(sample_id, str):
            return [sample_id]
        if isinstance(sample_id, list):
            return [str(item) for item in sample_id]
        if isinstance(sample_id, np.ndarray):
            return [str(item) for item in sample_id.tolist()]
        if torch.is_tensor(sample_id):
            return [str(item) for item in sample_id.tolist()]
        return [str(item) for item in sample_id]
    if hasattr(batch_data, "sid"):
        return [str(item) for item in batch_data.sid.tolist()]
    return [str(idx) for idx in range(batch_data.num_nodes)]


def gather_targets(batch_data) -> torch.Tensor:
    if hasattr(batch_data, "y_relaxed"):
        return batch_data.y_relaxed
    if hasattr(batch_data, "y"):
        return batch_data.y
    raise AttributeError("Batch does not contain target values.")


def write_predictions_csv(
    trainer,
    loader,
    out_path: str,
    disable_tqdm: bool = False,
) -> None:
    trainer.model.eval()
    predictions = []

    if trainer.ema:
        trainer.ema.store()
        trainer.ema.copy_to()

    if trainer.normalizers is not None and "target" in trainer.normalizers:
        trainer.normalizers["target"].to(trainer.device)

    for batch in tqdm(loader, desc=f"Predicting {out_path}", disable=disable_tqdm):
        batch_data = batch[0]
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=trainer.scaler is not None):
                out = trainer._forward(batch)

        energy = out["energy"]
        if trainer.normalizers is not None and "target" in trainer.normalizers:
            energy = trainer.normalizers["target"].denorm(energy)

        targets = gather_targets(batch_data).to(energy.device)
        ids = gather_ids(batch_data)

        for sample_id, target, pred in zip(
            ids, targets.detach().cpu().tolist(), energy.detach().cpu().tolist()
        ):
            predictions.append(
                {
                    "sample_id": sample_id,
                    "target": float(target),
                    "prediction": float(pred),
                }
            )

    if trainer.ema:
        trainer.ema.restore()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "target", "prediction"]
        )
        writer.writeheader()
        writer.writerows(predictions)
