# Crystal hardness training & prediction (all model examples)

This guide shows how to use the existing OCP training stack (GemNet + `EnergyTrainer`) to
train and predict crystal hardness from CIF structures.

## 1) Prepare a CSV file

Create a CSV with at least the following columns:

```csv
cif_path,hardness
structures/Fe2O3.cif,12.3
structures/TiO2.cif,14.8
```

Optional columns:
- `sample_id` (or any column specified via `--id-column`) is stored as `sample_id` in the LMDB.

## 2) Build LMDB datasets

Use the new preprocessing script to convert CIF files into LMDBs. Run it separately
for train/val/test splits:

```bash
python scripts/preprocess_hardness.py \
  --csv data/hardness/train.csv \
  --out-path data/hardness/train \
  --cif-root data/hardness/cifs \
  --get-edges

python scripts/preprocess_hardness.py \
  --csv data/hardness/val.csv \
  --out-path data/hardness/val \
  --cif-root data/hardness/cifs \
  --get-edges
```

This command writes:
- `data.lmdb` with graph data
- `metadata.npz` with atom/neighbor counts (for load balancing)
- `target_stats.json` with `target_mean` and `target_std`

Copy the `target_mean` and `target_std` values into your config file (see below).

## 3) Configure model for hardness regression

This repo ships example configs for all included models under `configs/hardness/`:

- `cgcnn/cgcnn.yml`
- `dimenet/dimenet.yml`
- `dimenet_plus_plus/dpp.yml`
- `forcenet/forcenet.yml`
- `gemnet/gemnet.yml`
- `schnet/schnet.yml`
- `spinconv/spinconv.yml`

Pick one config and update the dataset statistics:

```yaml
dataset:
  - src: data/hardness/train/data.lmdb
    normalize_labels: True
    target_mean: <from target_stats.json>
    target_std: <from target_stats.json>
  - src: data/hardness/val/data.lmdb
  - src: data/hardness/test/data.lmdb
```

Notes:
- Hardness is an intensive property, so `extensive: False` is set in the GemNet config.
- `scale_file` in the GemNet config reuses the existing GemNet scaling factors.

## 4) Train and predict

Train (replace the config with any of the models above):

```bash
python main.py --mode train --config-yml configs/hardness/gemnet/gemnet.yml
```

Predict:

```bash
python main.py --mode predict --config-yml configs/hardness/gemnet/gemnet.yml \
  --checkpoint checkpoints/<run-id>/checkpoint.pt
```

Predictions are saved to `results/<run-id>/predictions.npz`.
