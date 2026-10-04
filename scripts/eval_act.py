"""Evaluate ACT (action chunking) on LIBERO-Spatial, in the same protocol as
scripts/eval.py.

PROVENANCE
----------
Derived from the working script `day_act_eval.py`. That script was written but
never run to completion, so **no ACT success rate is reported anywhere in this
repo yet**. What changed here:

  CHANGED  loops over all benchmark init-states and writes CSV, matching
           scripts/eval.py so the two methods are directly comparable.
  CHANGED  suite/task/ckpt/episodes/seed/exec-horizon are flags.
  UNCHANGED the ACT-specific inference path, which is where all the alignment
           risk lives (see below).

Why the inference path is delicate
----------------------------------
ACT was trained on the LIBERO demos with a specific normalisation and layout.
Getting any of these wrong does not raise -- it silently produces bad actions,
so a success rate measured through a misaligned pipeline is a number about
nothing. All four must hold:

  1. images  two cameras stacked on dim 0, (2,3,128,128), float in [0,1].
             Order is agentview then eye_in_hand, matching
             `ActConfig.camera_names`.
  2. qpos    `robot_states` is 9-D: gripper qpos(2) + eef pos(3) +
             eef quat xyzw(4). Concatenate in that order and normalise with the
             mean/std stored in the checkpoint. The dataset stores this as a
             ready-made `robot_states` key; the env does not, so it must be
             assembled by hand -- see docs/notes/day2note.md §2.
  3. output  ACT returns a normalised chunk of shape (T,7). Denormalise with
             action_std / action_mean from the checkpoint.
  4. execute the chunk open-loop, then re-observe and re-predict. `--exec-steps`
             controls the temporal-ensembling horizon. `chunk_size=50` predicts
             50 actions but only the first 10 are executed before replanning,
             so the effective open-loop horizon is 10 steps at 20 Hz = 0.5 s.
             This choice is itself an experimental variable and has not been
             ablated (README §Ablations).

Usage
-----
    # smoke test
    python scripts/eval_act.py --ckpt outputs/act/policy_latest_full.pth \
        --init-states 0 --episodes 1

    # full protocol, matching scripts/eval.py
    python scripts/eval_act.py --ckpt outputs/act/policy_latest_full.pth \
        --episodes-per-init 10 --seed 0 --out-csv results/act_full_eval.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

REPO = Path(__file__).resolve().parent.parent
ACT_DIR = REPO / "act"

SUITE = "libero_spatial"
MAX_STEPS = 400
DEFAULT_EXEC = 10      # temporal-ensembling horizon; matches day_act_eval.py


def _import_act_policy():
    """Import the vendored ACTPolicy.

    Two paths are required. Upstream `detr/models/backbone.py` and
    `position_encoding.py` use an absolute `from util.misc import ...`, but
    `util` is not a top-level package in the upstream layout either -- the
    original repo fails the same way when imported. Adding `act/` and
    `act/act_src/detr/` to sys.path makes both `policy` and `util.misc`
    resolvable. See act/act_src/PROVENANCE.md §布局说明.
    """
    for p in (ACT_DIR, ACT_DIR / "act_src", ACT_DIR / "act_src" / "detr"):
        sys.path.insert(0, str(p))
    from policy import ACTPolicy
    return ACTPolicy


def build_policy(ckpt_path: Path, device: str = "cuda:0"):
    ACTPolicy = _import_act_policy()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    policy = ACTPolicy(ckpt["config"])
    policy.load_state_dict(ckpt["model"])
    policy.to(device).eval()
    return policy, ckpt["norm_stats"]


def env_obs_to_act(o: dict, stats: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ACT's two inputs from one env observation.

    Returns
        image : (2,3,128,128) float32 in [0,1], dim0 = (agentview, eye_in_hand)
        qpos  : (9,)    float32, normalised
    """
    imgs = np.stack([o["agentview_image"], o["robot0_eye_in_hand_image"]], axis=0)
    image = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0

    qpos = np.concatenate([
        np.asarray(o["robot0_gripper_qpos"], dtype=np.float32),   # 2
        np.asarray(o["robot0_eef_pos"], dtype=np.float32),       # 3
        np.asarray(o["robot0_eef_quat"], dtype=np.float32),      # 4  (xyzw)
    ])                                                            # -> 9
    qpos = (torch.from_numpy(qpos).float() - stats["qpos_mean"]) / stats["qpos_std"]
    return image, qpos


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--episodes-per-init", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-states", type=int, nargs="*", default=None)
    p.add_argument("--exec-steps", type=int, default=DEFAULT_EXEC,
                   help="actions executed per predicted chunk before replanning")
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--experiment-id", default=None)
    a = p.parse_args()

    exp_id = a.experiment_id or a.ckpt.stem
    policy, stats = build_policy(a.ckpt)
    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(a.task)
    states = suite.get_task_init_states(a.task)
    sids = a.init_states if a.init_states else list(range(len(states)))
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)

    per_state = []
    try:
        for sid in sids:
            init = states[sid]
            ok = 0
            for ep in range(a.episodes_per_init):
                env.seed(a.seed)
                env.reset()
                obs = env.set_init_state(init)
                steps, done = 0, False
                while steps < MAX_STEPS and not done:
                    image, qpos = env_obs_to_act(obs, stats)
                    with torch.no_grad():
                        a_hat = policy(qpos[None].cuda(), image[None].cuda())
                    chunk = a_hat[0].cpu().numpy()                       # (T,7)
                    chunk = chunk * stats["action_std"] + stats["action_mean"]
                    for t in range(a.exec_steps):                        # open loop
                        if steps >= MAX_STEPS:
                            break
                        obs, _r, done, _i = env.step(chunk[t])
                        steps += 1
                        if done:
                            break
                ok += int(done)
            per_state.append({"init_state": sid, "success": ok,
                              "episodes": a.episodes_per_init,
                              "rate": ok / a.episodes_per_init})
            print(f"init_state {sid:2d}: {ok:2d}/{a.episodes_per_init} "
                  f"({ok / a.episodes_per_init:.0%})")
    finally:
        env.close()

    tot_s = sum(r["success"] for r in per_state)
    tot_e = sum(r["episodes"] for r in per_state)
    print("=" * 56)
    print(f"ckpt        : {a.ckpt}")
    print(f"exec_steps  : {a.exec_steps}   env seed: {a.seed}")
    print(f"episodes    : {tot_e}")
    print(f"SUCCESS     : {tot_s} / {tot_e} = {tot_s / tot_e:.1%}"
          f"      <-- FIRST TIME THIS NUMBER EXISTS; record it honestly")
    print("=" * 56)

    if a.out_csv:
        a.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with a.out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_state[0].keys()))
            w.writeheader(); w.writerows(per_state)
        a.out_csv.with_suffix(".meta.json").write_text(json.dumps(
            {"ckpt": str(a.ckpt), "task": a.task, "task_language": task.language,
             "exec_steps": a.exec_steps, "episodes_per_init": a.episodes_per_init,
             "env_seed": a.seed, "max_steps": MAX_STEPS,
             "total_success": tot_s, "total_episodes": tot_e,
             "success_rate": tot_s / tot_e, "recorded_at": time.strftime("%Y-%m-%d")},
            indent=2))
        print(f"wrote {a.out_csv}")


if __name__ == "__main__":
    main()