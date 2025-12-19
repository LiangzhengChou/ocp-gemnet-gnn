"""Preprocess crystal hardness datasets into LMDB files.

Expected CSV columns:
- cif_path: path to a CIF file (relative to --cif-root if provided)
- hardness: target value (float)
"""

import argparse
import csv
import json
import os
import pickle
from typing import Dict, List, Tuple

import ase.io
import lmdb
import numpy as np
import torch
from tqdm import tqdm

from ocpmodels.preprocessing import AtomsToGraphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CIF + hardness CSV into an LMDB dataset."
    )
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument(
        "--out-path",
        required=True,
        help="Output directory for data.lmdb and metadata.npz",
    )
    parser.add_argument(
        "--cif-root",
        default=None,
        help="Optional root directory prepended to cif paths",
    )
    parser.add_argument(
        "--cif-column",
        default="cif_path",
        help="CSV column containing CIF paths",
    )
    parser.add_argument(
        "--target-column",
        default="hardness",
        help="CSV column containing hardness values",
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help="Optional CSV column to store as sample_id",
    )
    parser.add_argument(
        "--max-neigh",
        type=int,
        default=50,
        help="Maximum neighbors per atom",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=6.0,
        help="Neighbor cutoff radius (Angstrom)",
    )
    parser.add_argument(
        "--get-edges",
        action="store_true",
        help="Store edge indices in LMDB instead of OTF graph building",
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Skip rows that fail to parse",
    )
    parser.add_argument(
        "--map-size-gb",
        type=int,
        default=64,
        help="LMDB map size in GB",
    )
    return parser.parse_args()


def build_cif_path(cif_root: str, cif_path: str) -> str:
    if cif_root:
        return os.path.join(cif_root, cif_path)
    return cif_path


def load_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def convert_row(
    a2g: AtomsToGraphs,
    row: Dict[str, str],
    idx: int,
    cif_root: str,
    cif_column: str,
    target_column: str,
    id_column: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cif_path = build_cif_path(cif_root, row[cif_column])
    atoms = ase.io.read(cif_path)
    data_object = a2g.convert(atoms)
    data_object.y = torch.tensor(
        [float(row[target_column])], dtype=torch.float
    )
    data_object.sid = idx
    if id_column:
        data_object.sample_id = str(row[id_column])
    data_object.cif_path = os.path.abspath(cif_path)

    natoms = torch.tensor([data_object.natoms])
    if hasattr(data_object, "edge_index"):
        neighbors = torch.tensor([data_object.edge_index.shape[1]])
    else:
        neighbors = torch.tensor([0])

    return data_object, natoms, neighbors


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    a2g = AtomsToGraphs(
        max_neigh=args.max_neigh,
        radius=args.radius,
        r_energy=False,
        r_forces=False,
        r_distances=False,
        r_edges=args.get_edges,
        r_fixed=True,
    )

    db_path = os.path.join(args.out_path, "data.lmdb")
    db = lmdb.open(
        db_path,
        map_size=args.map_size_gb * 1024**3,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    rows = load_rows(args.csv)
    natoms_list = []
    neighbors_list = []
    targets = []

    idx = 0
    with db.begin(write=True) as txn:
        for row in tqdm(rows, desc="Converting CIFs to LMDB"):
            try:
                data_object, natoms, neighbors = convert_row(
                    a2g,
                    row,
                    idx,
                    args.cif_root,
                    args.cif_column,
                    args.target_column,
                    args.id_column,
                )
            except Exception as exc:
                if args.skip_failed:
                    print(
                        f"Skipping row {idx} due to error: {exc}",
                        flush=True,
                    )
                    continue
                raise

            txn.put(
                f"{idx}".encode("ascii"),
                pickle.dumps(data_object, protocol=-1),
            )
            natoms_list.append(natoms)
            neighbors_list.append(neighbors)
            targets.append(float(row[args.target_column]))
            idx += 1

        txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))

    db.sync()
    db.close()

    natoms_arr = torch.cat(natoms_list).numpy() if natoms_list else np.array([])
    neighbors_arr = (
        torch.cat(neighbors_list).numpy() if neighbors_list else np.array([])
    )
    np.savez(
        os.path.join(args.out_path, "metadata.npz"),
        natoms=natoms_arr,
        neighbors=neighbors_arr,
    )

    if targets:
        target_array = np.array(targets, dtype=np.float32)
        stats = {
            "target_mean": float(target_array.mean()),
            "target_std": float(target_array.std()),
            "num_samples": int(target_array.shape[0]),
        }
    else:
        stats = {"target_mean": 0.0, "target_std": 1.0, "num_samples": 0}

    stats_path = os.path.join(args.out_path, "target_stats.json")
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(
        "Finished writing LMDB with {count} samples. Stats saved to {stats}.".format(
            count=idx, stats=stats_path
        )
    )


if __name__ == "__main__":
    main()
