"""Thin, well-behaved launcher for robomimic's BC / BC-RNN trainer.

Why a wrapper instead of calling robomimic directly
--------------------------------------------------
robomimic's own entry point is

    python -m robomimic.scripts.train --config <json>

and that works, but it means the config file must hardcode absolute paths for
the dataset and the output directory, which is exactly what makes a repo
unportable. This wrapper keeps the committed config free of machine-specific
paths by substituting two tokens:

    configs/bc_rnn_task0.json
        "data":       "__DATA__"
        "output_dir": "__OUTPUT_DIR__"

resolved here from `--data` / `--output-dir` (defaulting to $REPO/data and
$REPO/outputs). Everything else in the config is passed through untouched, so
the committed file remains the single source of truth for hyperparameters.

The experiment name is derived from the config's `experiment.name`, and
robomimic appends its own timestamped run directory, e.g.
    outputs/bc_rnn_aug_200ep/20260927165848/models/model_epoch_200.pth

Usage
-----
    python scripts/train_bc_rnn.py --config configs/bc_rnn_task0.json
    python scripts/train_bc_rnn.py --config configs/bc_rnn_task0.json \
        --set train.num_epochs=50 --set algo.rnn.enabled=true

Notes on the specific configuration in configs/bc_rnn_task0.json
-----------------------------------------------------------------
  validate          false  -- robomimic's validation step wants to build a live
                             mujoco_py env, which this venv does not have. Pure
                             offline training does not need it. Consequence: this
                             run has NO held-out validation split, so training
                             log-likelihood is not evidence of generalisation.
  obs_randomizer    CropRandomizer 76x76 out of 128x128 -- random image crops, a
                             stand-in for "the bowl has shifted left/right". The
                             ablation that isolates its effect has not been run
                             (see README §Ablations).
  gmm               enabled, 5 modes -- robomimic's default BC-RNN head.
  rnn.horizon       10, matching train.seq_length = 10.
  lr                constant; `epoch_schedule` is empty, so there is no decay.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def resolve_paths(cfg: dict, data: Path, outdir: Path) -> dict:
    """Replace the __DATA__ / __OUTPUT_DIR__ tokens, recursively.

    Fails loudly on an unresolved token rather than letting robomimic try to
    open a file literally named "__DATA__".
    """
    def walk(node):
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, str) and node in ("__DATA__", "__OUTPUT_DIR__"):
            return str(data if node == "__DATA__" else outdir)
        return node

    out = walk(copy.deepcopy(cfg))
    blob = json.dumps(out)
    for token in ("__DATA__", "__OUTPUT_DIR__"):
        if token in blob:
            raise ValueError(f"unresolved token {token} in config")
    return out


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply `--set a.b.c=value` overrides (values parsed as JSON, else str)."""
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node:
                raise SystemExit(f"no such config key: {key}")
            node = node[p]
        if parts[-1] not in node:
            raise SystemExit(f"no such config key: {key}")
        node[parts[-1]] = val
        print(f"  override {key} = {val!r}")
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=REPO / "configs" / "bc_rnn_task0.json")
    p.add_argument("--data", type=Path, default=REPO / "data" / "libero_spatial",
                   help="directory or file passed to robomimic's train.data")
    p.add_argument("--output-dir", type=Path, default=REPO / "outputs")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="dotted config override, repeatable")
    p.add_argument("--dry-run", action="store_true",
                   help="print the resolved config and exit without training")
    a = p.parse_args()

    cfg = json.loads(a.config.read_text())
    cfg = resolve_paths(cfg, a.data.resolve(), a.output_dir.resolve())
    cfg = apply_overrides(cfg, a.overrides)

    # robomimic reads MUJOCO_GL for its offscreen renderer; harmless and required
    # if anyone re-enables rollout/validation.
    os.environ.setdefault("MUJOCO_GL", "egl")

    print(f"config      : {a.config}")
    print(f"data        : {cfg['train']['data']}")
    print(f"output_dir  : {cfg['train']['output_dir']}")
    print(f"experiment  : {cfg['experiment']['name']}")
    print(f"epochs      : {cfg['train']['num_epochs']}  seed: {cfg['train']['seed']}")

    if a.dry_run:
        print("\n--dry-run: resolved config")
        print(json.dumps(cfg, indent=2))
        return

    import robomimic.utils.file_utils as FileUtils
    from robomimic.config import Config
    from robomimic.scripts import train as rm_train

    # robomimic's train() reads the config from a file path, so materialise the
    # resolved config to a temp file rather than reimplementing its main().
    resolved = a.output_dir.resolve() / f".resolved_{a.config.stem}.json"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(cfg, indent=2))
    print(f"\nresolved config written to {resolved}")
    print("invoking robomimic.scripts.train ...")
    rm_train.main(["--config", str(resolved)])


if __name__ == "__main__":
    sys.exit(main())