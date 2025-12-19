"""Generate CSV predictions for train/val/test splits.

Outputs CSV files with columns: id, target, prediction.
"""

import argparse
import csv
import logging
import os
from typing import Iterable, List

import numpy as np
import torch
from tqdm import tqdm

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import load_config, setup_imports, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate hardness prediction CSVs for each split."
    )
    parser.add_argument(
        "--config-yml", required=True, help="Path to hardness config YAML"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--out-dir",
        default="hardness_predictions",
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to export (train val test)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution",
    )
    return parser.parse_args()


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


def write_predictions(trainer, loader, out_path: str) -> None:
    trainer.model.eval()
    predictions = []

    if trainer.ema:
        trainer.ema.store()
        trainer.ema.copy_to()

    if trainer.normalizers is not None and "target" in trainer.normalizers:
        trainer.normalizers["target"].to(trainer.device)

    for batch in tqdm(loader, desc=f"Predicting {out_path}"):
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
                {"id": sample_id, "target": float(target), "prediction": float(pred)}
            )

    if trainer.ema:
        trainer.ema.restore()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "target", "prediction"])
        writer.writeheader()
        writer.writerows(predictions)


def main() -> None:
    args = parse_args()
    setup_logging()
    setup_imports()

    config, _, _ = load_config(args.config_yml)
    trainer = registry.get_trainer_class(config.get("trainer", "energy"))(
        task=config["task"],
        model=config["model"],
        dataset=config["dataset"],
        optimizer=config["optim"],
        identifier="hardness-predict",
        run_dir=os.getcwd(),
        print_every=10,
        seed=0,
        logger=config.get("logger", "tensorboard"),
        local_rank=0,
        amp=False,
        cpu=args.cpu,
        slurm={},
    )

    trainer.load_checkpoint(args.checkpoint)

    split_map = {
        "train": trainer.train_loader,
        "val": trainer.val_loader,
        "test": trainer.test_loader,
    }

    for split in args.splits:
        loader = split_map.get(split)
        if loader is None:
            logging.warning("Split '%s' is not available in config", split)
            continue
        out_path = os.path.join(args.out_dir, f"{split}.csv")
        write_predictions(trainer, loader, out_path)
        logging.info("Wrote %s predictions to %s", split, out_path)


if __name__ == "__main__":
    main()
