"""Classify why rollouts fail, by measuring the gripper-to-bowl distance every step.

PROVENANCE
----------
Derived from the working script `day_failure_census.py`. What changed:

  CHANGED  results are written to CSV (per-episode and per-init-state), not only
           printed. The original printed to stdout only, so the numbers quoted in
           the lab notes could not be re-checked from an artifact.
  CHANGED  suite/task/checkpoint/init-state are argparse flags.
  CHANGED  the distance-threshold constant is a flag, and its value is recorded
           in the output CSV header so a number can never be quoted without the
           threshold that produced it.
  UNCHANGED the measurement itself: 2-D (x,y) distance in millimetres between
           `robot0_eef_pos` and the bowl's centre, and the 4-way classifier.

Why measure distance at all
---------------------------
Static object-position tables were tried first and were not sufficient: three
geometric hypotheses about the hard init-states were each refuted by a matched
control init-state (see docs/notes/day7note.md §3). Position tables produce
hypotheses; per-step kinematics produce the cause.

Labels
------
  success      env reported done (BDDL goal `On akita_black_bowl_1 plate_1`)
  locate_fail  never came within --near-mm of the bowl in the x-y plane
  drop         bowl rose >30mm then fell back >50mm from its peak
  align_fail   reached the bowl but it never rose -> closed on nothing
  other        bowl moved but the episode still failed

Usage
-----
    python scripts/failure_census.py --ckpt <path.pth> --init-state 13 --episodes 10
    python scripts/failure_census.py --ckpt <path.pth> --init-states 1 2 10 13 --episodes 10
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

import robomimic.utils.file_utils as FileUtils
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

SUITE = "libero_spatial"
MAX_STEPS = 400
BOWL_KEY = "akita_black_bowl_1_pos"   # object the task goal refers to
LIFT_M = 0.03                          # bowl considered "picked up" above +30mm
DROP_M = 0.05                          # ...and "dropped" if it falls 50mm below peak


def env_obs_to_policy(o: dict) -> dict:
    """Same translation as scripts/eval.py; kept local so this script has no
    import-time dependency on the eval module."""
    return {
        "agentview_rgb": (o["agentview_image"].astype(np.float32) / 255.0).transpose(2, 0, 1),
        "eye_in_hand_rgb": (o["robot0_eye_in_hand_image"].astype(np.float32) / 255.0).transpose(2, 0, 1),
        "gripper_states": o["robot0_gripper_qpos"].astype(np.float32),
        "joint_states": o["robot0_joint_pos"].astype(np.float32),
    }


def finger_center(o: dict) -> np.ndarray:
    """World x-y-z of the gripper centre.

    Uses `robot0_eef_pos`, i.e. the end-effector site. It is an approximation of
    the midpoint between the two fingertips: the real finger centre would need
    the two finger joint sites, which the observation dict does not expose
    directly. The approximation is constant across all episodes, so comparisons
    between episodes remain valid, but the absolute millimetre values carry that
    systematic offset. Stated here so the threshold is not over-interpreted.
    """
    return np.asarray(o["robot0_eef_pos"], dtype=float)


def classify(done: bool, min_dist_mm: float, lifted: bool, dropped: bool,
             near_mm: float) -> str:
    if done:
        return "success"
    if min_dist_mm > near_mm:
        return "locate_fail"
    if dropped:
        return "drop"
    if not lifted:
        return "align_fail"
    return "other"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--init-states", type=int, nargs="+", default=[13, 1, 2, 10],
                   help="default: the four hardest init-states from the 500-ep run")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--near-mm", type=float, default=60.0,
                   help="x-y distance below which the gripper counts as having "
                        "reached the bowl (locate vs align split)")
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--experiment-id", default=None)
    a = p.parse_args()

    exp_id = a.experiment_id or Path(a.ckpt).stem
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=a.ckpt, device="cuda:0")
    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(a.task)
    states = suite.get_task_init_states(a.task)
    bddl = os.path.join(get_libero_path("bddl_files"),
                        task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    ep_rows, state_rows = [], []

    try:
        for sid in a.init_states:
            init = states[sid]
            census = {"success": 0, "locate_fail": 0, "align_fail": 0,
                      "drop": 0, "other": 0}
            for ep in range(a.episodes):
                env.seed(a.seed)
                env.reset()
                obs = env.set_init_state(init)
                policy.start_episode()

                bowl0 = np.asarray(obs[BOWL_KEY], dtype=float)
                min_dist, steps, done = 1e9, 0, False
                lifted = dropped = False
                peak = bowl0[2]
                gripper_open_at_min = None

                while steps < MAX_STEPS:
                    obs, _r, done, _i = env.step(policy(env_obs_to_policy(obs)))
                    steps += 1
                    f = finger_center(obs)
                    b = np.asarray(obs[BOWL_KEY], dtype=float)
                    d = float(np.linalg.norm((f - b)[:2]) * 1000.0)
                    if d < min_dist:
                        min_dist = d
                        gripper_open_at_min = float(np.asarray(
                            obs["robot0_gripper_qpos"], dtype=float).mean())
                    if b[2] > bowl0[2] + LIFT_M:
                        lifted = True
                    peak = max(peak, b[2])
                    if lifted and b[2] < peak - DROP_M:
                        dropped = True
                    if done:
                        break

                label = classify(done, min_dist, lifted, dropped, a.near_mm)
                census[label] += 1
                ep_rows.append({
                    "init_state": sid, "episode": ep, "steps": steps,
                    "min_dist_mm": round(min_dist, 2),
                    "peak_lift_mm": round((peak - bowl0[2]) * 1000.0, 1),
                    "label": label,
                })
                print(f"init {sid:2d} ep {ep:2d}: steps={steps:3d} "
                      f"min_dist={min_dist:6.1f}mm "
                      f"peak_lift={(peak - bowl0[2]) * 1000.0:5.0f}mm -> {label}")

            n = a.episodes
            state_rows.append({"init_state": sid, "episodes": n,
                               **{k: census[k] for k in census},
                               "success_rate": census["success"] / n})
            print(f"\ninit {sid:2d} census: {census}\n")
    finally:
        env.close()

    ep_csv = a.out_dir / f"census_{exp_id}_episodes.csv"
    st_csv = a.out_dir / f"census_{exp_id}_by_init_state.csv"
    with ep_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ep_rows[0].keys()))
        w.writeheader(); w.writerows(ep_rows)
    with st_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(state_rows[0].keys()))
        w.writeheader(); w.writerows(state_rows)

    meta = {"ckpt": a.ckpt, "task": a.task, "task_language": task.language,
            "init_states": a.init_states, "episodes_per_init": a.episodes,
            "env_seed": a.seed, "near_mm": a.near_mm,
            "lift_threshold_m": LIFT_M, "drop_threshold_m": DROP_M,
            "bowl_key": BOWL_KEY}
    (a.out_dir / f"census_{exp_id}_meta.json").write_text(json.dumps(meta, indent=2))

    tot = sum(r["success"] for r in state_rows)
    tot_ep = sum(r["episodes"] for r in state_rows)
    print("=" * 56)
    for r in state_rows:
        print(f"init {r['init_state']:2d}: success={r['success']}/{r['episodes']} "
              f"locate_fail={r['locate_fail']} align_fail={r['align_fail']} "
              f"drop={r['drop']} other={r['other']}")
    print(f"TOTAL success {tot}/{tot_ep}")
    print(f"\nwrote {ep_csv}\nwrote {st_csv}")


if __name__ == "__main__":
    main()