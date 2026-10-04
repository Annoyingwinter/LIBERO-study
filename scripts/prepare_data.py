"""Add the `next_obs` group that robomimic's dataset loader requires but that the
released LIBERO demo files do not contain.

Background
----------
robomimic's `SequenceDataset` always reads `next_obs`, because BC-RNN and the
GMM action head are trained on (obs_t, action_t, next_obs_t) triples. The public
LIBERO demos only ship `obs`. Without the patch, `robomimic/scripts/train.py`
dies at dataset construction.

This script is a *reconstruction*: the original ad-hoc patch was not saved, so it
was rebuilt from the shape of the file that patch produced and then verified
byte-for-byte against that file (see `verify()` below and §Provenance in the
README). The rule it implements was confirmed empirically:

    next_obs[t] = obs[t + 1]   for t < T - 1
    next_obs[T - 1] = obs[T - 1]        (final step duplicates itself)

`next_obs` mirrors every one of the 7 keys under `obs`, with identical shapes
and dtypes. Demo-level attrs (`init_state`, `model_file`, `num_samples`) are
copied verbatim -- `model_file` in particular must survive, because
robomimic/robosuite reads the MJCF out of it to rebuild the env.

Usage
-----
    python scripts/prepare_data.py \
        --in  data/libero_spatial/<task>_demo.hdf5 \
        --out data/libero_spatial/<task>_nextobs.hdf5

    # check the result instead of building it
    python scripts/prepare_data.py --in ... --out ... --verify
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np


def add_next_obs(src: Path, dst: Path) -> int:
    """Copy `src` to `dst`, adding a next_obs group to every demo."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copyfile(src, dst)

    n_demos = 0
    with h5py.File(dst, "r+") as f:
        data = f["data"]
        for demo_key in sorted(data.keys()):
            demo = data[demo_key]
            if "next_obs" in demo:              # idempotent
                print(f"  {demo_key}: already has next_obs, skipped")
                n_demos += 1
                continue
            if "obs" not in demo:
                raise KeyError(f"{demo_key} has no 'obs' group; not a LIBERO demo")

            t_max = demo["actions"].shape[0]
            nxt = demo.create_group("next_obs")
            for obs_key in demo["obs"].keys():
                obs_ds = demo["obs"][obs_key]
                shape, dtype = obs_ds.shape, obs_ds.dtype
                nxt_ds = nxt.create_dataset(obs_key, shape=shape, dtype=dtype)

                # read in chunks: a single (T,128,128,3) RGB block is ~4.8 MB and
                # the full dataset has 50 demos, so we never hold a demo's images
                # and its successor in RAM at full size at once.
                if obs_ds.ndim == 4 and obs_ds.shape[1:] == (128, 128, 3):
                    for t in range(t_max - 1):
                        nxt_ds[t] = obs_ds[t + 1]
                else:
                    arr = obs_ds[:]
                    nxt_ds[: t_max - 1] = arr[1:t_max]
                nxt_ds[t_max - 1] = obs_ds[t_max - 1]   # final step self-copies
            n_demos += 1
            print(f"  {demo_key}: T={t_max}, added next_obs for "
                  f"{len(demo['obs'])} keys")
    return n_demos


def verify(path: Path) -> bool:
    """Assert the next_obs rule holds on every demo and every key.

    Exits non-zero on the first violation so it is usable as a CI check.
    """
    bad = 0
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys())
        print(f"verifying {len(demos)} demos in {path.name}")
        for dk in demos:
            demo = f["data"][dk]
            if "next_obs" not in demo:
                print(f"  FAIL {dk}: no next_obs group")
                bad += 1
                continue
            t_max = demo["actions"].shape[0]
            keys_obs = set(demo["obs"].keys())
            keys_nxt = set(demo["next_obs"].keys())
            if keys_obs != keys_nxt:
                print(f"  FAIL {dk}: key mismatch {keys_obs ^ keys_nxt}")
                bad += 1
                continue
            for k in sorted(keys_obs):
                o, n = demo["obs"][k], demo["next_obs"][k]
                if o.shape != n.shape or o.dtype != n.dtype:
                    print(f"  FAIL {dk}/{k}: shape/dtype {o.shape}{o.dtype} "
                          f"vs {n.shape}{n.dtype}")
                    bad += 1
                    continue
                if not np.array_equal(n[: t_max - 1], o[1:t_max]):
                    print(f"  FAIL {dk}/{k}: next_obs[t] != obs[t+1]")
                    bad += 1
                if not np.array_equal(n[t_max - 1], o[t_max - 1]):
                    print(f"  FAIL {dk}/{k}: last step does not self-copy")
                    bad += 1
            print(f"  ok {dk}: T={t_max}, {len(keys_obs)} keys")
        missing_attrs = [d for d in demos
                         if not {"init_state", "model_file", "num_samples"}
                         <= set(f["data"][d].attrs.keys())]
        if missing_attrs:
            print(f"  FAIL demos missing attrs: {missing_attrs}")
            bad += 1
    if bad:
        print(f"\nFAILED: {bad} problem(s)")
        return False
    print("\nOK: next_obs[t]==obs[t+1] (last step self-copies) on all demos")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", type=Path, required=True)
    p.add_argument("--out", dest="dst", type=Path, required=True)
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()

    if a.verify_only:
        raise SystemExit(0 if verify(a.dst) else 1)

    if not a.src.exists():
        raise SystemExit(f"input not found: {a.src}")
    print(f"{a.src}  ->  {a.dst}")
    n = add_next_obs(a.src, a.dst)
    print(f"\npatched {n} demos")
    print(f"verifying...")
    raise SystemExit(0 if verify(a.dst) else 1)


if __name__ == "__main__":
    main()