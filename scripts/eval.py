"""Evaluate a policy on LIBERO-Spatial and report success rate per init-state.

PROVENANCE
----------
Derived from the working script `day_full_eval.py`, which produced the 500-episode
number reported in the README (`results/bc_rnn_200ep_full_eval.csv`,
439/500 = 87.8%). What changed relative to that file, and why:

  CHANGED  paths / suite / task / episodes / seed are argparse flags instead of
           module constants, so the same script serves every row of the results
           table without editing source.
  CHANGED  `--video-dir` is keyed on experiment name + checkpoint stem. The
           original derived it from `basename(ckpt)` alone, which collided:
           `bc_rnn_libero_task0/model_epoch_50.pth` and
           `bc_rnn_aug_200ep/model_epoch_50.pth` both produced
           `eval_videos/model_epoch_50/` and overwrote each other by episode
           index. See README §Limitations.
  CHANGED  per-episode rows are written to CSV, not only per-init-state totals,
           so an individual failure can be traced back to a step count.
  UNCHANGED the rollout loop itself, the observation translation, the env
           construction (OffScreenRenderEnv, 128x128), MAX_STEPS=400, and
           `env.seed(0)` — all of these affect the reported numbers and were
           deliberately left alone so this script still reproduces 439/500.

Success criterion
-----------------
Success is the env's own BDDL goal predicate, surfaced as `done` from
`env.step()`. For LIBERO-Spatial task 0 the goal is
`(And (On akita_black_bowl_1 plate_1))`. This is NOT the same as robosuite's
internal `self.done` (which fires at the timestep horizon) nor the `dones` array
stored in the demos (which marks where the demonstrator stopped). See
docs/notes/day3note.md §3 for the three-way disambiguation with file:line refs.

Usage
-----
    python scripts/eval.py --ckpt <path.pth> --episodes-per-init 10 --seed 0
    python scripts/eval.py --ckpt <path.pth> --episodes-per-init 10 \
        --video-dir media/rollouts/bc_rnn_200ep --video-successes 1 --video-failures 1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

import robomimic.utils.file_utils as FileUtils
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# ---------------------------------------------------------------------------
# Fixed by the benchmark, not by the experiment: these must not be flags.
# ---------------------------------------------------------------------------
SUITE = "libero_spatial"
MAX_STEPS = 400          # longest demo is 168 steps; 400 leaves headroom
CAM_HW = 128             # must match the 128x128 images stored in the demos


def build_policy(ckpt: str, device: str = "cuda:0"):
    """Rebuild a robomimic policy straight from its checkpoint.

    robomimic stores the full training config inside the .pth, so
    policy_from_checkpoint recovers the network structure, obs keys and shapes
    without needing the original config file.
    """
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt, device=device)
    return policy


def env_obs_to_policy(o: dict) -> dict:
    """Translate one env observation dict into robomimic's expected schema.

    Three translations, each of which silently corrupts results if skipped:
      key name   agentview_image          -> agentview_rgb
                 robot0_eye_in_hand_image -> eye_in_hand_rgb
      layout     (128,128,3) uint8       -> (3,128,128) float32 in [0,1]
      rename     robot0_gripper_qpos     -> gripper_states
                 robot0_joint_pos        -> joint_states
    """
    return {
        "agentview_rgb": (o["agentview_image"].astype(np.float32) / 255.0).transpose(2, 0, 1),
        "eye_in_hand_rgb": (o["robot0_eye_in_hand_image"].astype(np.float32) / 255.0).transpose(2, 0, 1),
        "gripper_states": o["robot0_gripper_qpos"].astype(np.float32),
        "joint_states": o["robot0_joint_pos"].astype(np.float32),
    }


def make_env(task):
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    return OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=CAM_HW, camera_widths=CAM_HW)


def run_episode(env, policy, init_state, seed: int, recorder=None):
    """One closed-loop episode. Returns (success, n_steps).

    Order matters and is not arbitrary:
      env.seed(s)      -> controls any stochastic env reset
      env.reset()      -> must precede set_init_state
      env.set_init_state(init)  -> overwrites the sim with the benchmark state
      policy.start_episode()    -> zeroes the RNN hidden state; skipping it
                                   leaks the previous episode's history into
                                   this one, which silently inflates scores.
    """
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(init_state)
    policy.start_episode()

    steps, done = 0, False
    while steps < MAX_STEPS:
        action = policy(env_obs_to_policy(obs))
        obs, _reward, done, _info = env.step(action)
        steps += 1
        if recorder is not None:
            recorder(obs["agentview_image"])
        if done:
            break
    return bool(done), steps


def _iter_modules(obj, depth: int = 0, max_depth: int = 8):
    """Yield every nn.Module reachable from `obj`, descending through robomimic's
    plain-object wrappers.

    `FileUtils.policy_from_checkpoint` returns a `RolloutPolicy`, which is not an
    nn.Module. Its `.policy` is a robomimic `Module`, also not an nn.Module, whose
    `.nets` is a robomimic `ModuleDict` wrapping a torch `nn.ModuleDict` (which is
    *not* itself an nn.Module and so has no `.modules()`). torch's ModuleDict /
    ModuleList / Sequential all implement `named_modules()`, which is the reliable
    way in. The real chain is

        RolloutPolicy -> .policy -> .nets["policy"] -> .nets.encoder.nets.obs
                                      -> .obs_randomizers["agentview_rgb"]

    so this walks `.net` / `.nets` / `.policy` by name and uses `named_modules()`
    wherever it exists.
    """
    if depth > max_depth:
        return
    if hasattr(obj, "named_modules"):
        yield from obj.named_modules()
        return
    for attr in ("policy", "nets", "net", "obs_encoder", "encoder", "obs"):
        child = getattr(obj, attr, None)
        if child is None or child is obj:
            continue
        if isinstance(child, dict):
            for v in child.values():
                yield from _iter_modules(v, depth + 1, max_depth)
        elif isinstance(child, (list, tuple)):
            for v in child:
                yield from _iter_modules(v, depth + 1, max_depth)
        else:
            yield from _iter_modules(child, depth + 1, max_depth)


def _center_crop(x, out_h: int, out_w: int):
    """Deterministic centre crop of a (B, C, H, W) tensor.

    An identity bypass does NOT work here. robomimic sizes the downstream encoder
    from `randomizer.output_shape_in()` at construction time
    (`models/obs_nets.py:166`), so the ResNet expects 76x76 and raises
    `assert tuple(inputs.shape)[-ndim:] == tuple(self.input_shape)` on a 128x128
    input. The consequence is worth stating plainly: **a policy trained with
    CropRandomizer cannot see the full frame at all**, by construction. The only
    question a centre crop can answer is whether the *randomness* of the crop
    helps or hurts at inference, not whether cropping itself is costly.
    """
    import torch
    _, _, h, w = x.shape[-4:]
    top = (h - out_h) // 2
    left = (w - out_w) // 2
    return x[..., top:top + out_h, left:left + out_w]


def patch_test_randomizers(policy, mode: str = "center", verbose: bool = True) -> int:
    """Replace each observation randomizer's `forward_in` with a deterministic one.

    modes
    -----
    center   centre crop of the same size as the randomizer's crop (default).
             Removes the randomness, keeps the shape contract.
    identity feed the observation through untouched. Works only if the encoder
             was not shaped by the randomizer; otherwise it asserts.

    `forward_out` is left alone: it averages crop features, which is a no-op at
    num_crops == 1 (the configuration used here).

    Returns the number of randomizers patched.
    """
    targets = [(n, m) for n, m in _iter_modules(policy)
               if type(m).__name__.endswith("Randomizer")]
    for _name, m in targets:
        if mode == "identity":
            m.forward_in = lambda x: x
        else:
            ch, cw = getattr(m, "crop_height", None), getattr(m, "crop_width", None)
            if ch is None or cw is None:
                raise AttributeError(
                    f"{type(m).__name__} has no crop_height/crop_width; "
                    f"--test-randomizer={mode} does not know how to replace it")
            m.forward_in = (lambda c: lambda x: _center_crop(x, c[0], c[1]))((ch, cw))
    if verbose:
        for name, m in targets:
            print(f"  patched {name}: {mode} crop "
                  f"{getattr(m, 'crop_height', '?')}x{getattr(m, 'crop_width', '?')}")
    return len(targets)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="path to a robomimic .pth checkpoint")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--suite", default=SUITE, help="read-only: only libero_spatial is wired up")
    p.add_argument("--episodes-per-init", type=int, default=10,
                   help="repetitions per benchmark init-state")
    p.add_argument("--seed", type=int, default=0,
                   help="env seed. NOTE: this is the only seed; repetition comes "
                        "from --episodes-per-init, so results carry no seed variance")
    p.add_argument("--init-states", type=int, nargs="*", default=None,
                   help="optional subset of init-state ids (e.g. the 4 hard ones)")
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--video-dir", type=Path, default=None,
                   help="record rollout clips here; name is suffixed with the "
                        "experiment id so two runs cannot collide")
    p.add_argument("--video-successes", type=int, default=0)
    p.add_argument("--video-failures", type=int, default=0)
    p.add_argument("--experiment-id", default=None,
                   help="label for the output; defaults to the checkpoint stem")
    p.add_argument("--test-randomizer", choices=["as-trained", "center"],
                   default="as-trained",
                   help="'as-trained' (default) leaves the observation randomizer "
                        "active, which is how these numbers were originally "
                        "produced. 'center' replaces the random crop with a "
                        "deterministic centre crop of the same size, isolating the "
                        "effect of the crop's randomness. 'identity' is not offered: "
                        "robomimic sizes the encoder from the randomizer's output "
                        "shape (obs_nets.py:166), so feeding the full 128x128 frame "
                        "raises a shape assertion. See README section 7.3.")
    a = p.parse_args()

    if a.suite != SUITE:
        raise SystemExit(f"only {SUITE} is wired up (got {a.suite})")

    exp_id = a.experiment_id or Path(a.ckpt).stem
    suite = benchmark.get_benchmark_dict()[a.suite]()
    task = suite.get_task(a.task)
    states = suite.get_task_init_states(a.task)
    sids = a.init_states if a.init_states else list(range(len(states)))

    writer = None
    if a.video_dir:
        a.video_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(task)
    policy = build_policy(a.ckpt)
    if a.test_randomizer != "as-trained":
        n = patch_test_randomizers(policy, mode=a.test_randomizer)
        print(f"patched {n} observation randomizer(s) -> '{a.test_randomizer}'")
        if n == 0:
            print("NOTE: --test-randomizer was passed but no randomizer was found. "
                  "If this checkpoint has obs_randomizer_class=null then the flag "
                  "is a no-op and this run duplicates a no-aug baseline.")
    per_state, per_ep = [], []
    n_rec_s = n_rec_f = 0

    try:
        for sid in sids:
            init = states[sid]
            ok = 0
            for ep in range(a.episodes_per_init):
                # Record only the first N of each outcome class, so a 500-episode
                # run does not write 500 clips.
                want_success = ok > 0 and n_rec_s < a.video_successes
                want_failure = ok == 0 and n_rec_f < a.video_failures
                frames: list = []
                rec = frames.append if (a.video_dir and (want_success or want_failure)) else None

                success, steps = run_episode(env, policy, init, a.seed, rec)

                if rec is not None:
                    kind = "success" if success else "failure"
                    save_clip(frames, a.video_dir / f"{exp_id}_init{sid}_{kind}.mp4")
                    if success:
                        n_rec_s += 1
                    else:
                        n_rec_f += 1

                if success:
                    ok += 1
                per_ep.append({"init_state": sid, "episode": ep,
                               "steps": steps, "success": int(success)})
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
    print(f"ckpt      : {a.ckpt}")
    print(f"task      : {a.task} ({task.language})")
    print(f"episodes  : {tot_e} = {len(sids)} init-states x {a.episodes_per_init}")
    print(f"env seed  : {a.seed}  (single seed -- no seed variance reported)")
    print(f"SUCCESS   : {tot_s} / {tot_e} = {tot_s / tot_e:.1%}")
    print("=" * 56)

    if a.out_csv:
        a.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with a.out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_state[0].keys()))
            w.writeheader()
            w.writerows(per_state)
        with a.out_csv.with_suffix(".episodes.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_ep[0].keys()))
            w.writeheader()
            w.writerows(per_ep)
        print(f"wrote {a.out_csv}")
        print(f"wrote {a.out_csv.with_suffix('.episodes.csv')}")

    if a.out_csv:
        meta = {"ckpt": a.ckpt, "task": a.task, "task_language": task.language,
                "suite": a.suite, "episodes_per_init": a.episodes_per_init,
                "env_seed": a.seed, "max_steps": MAX_STEPS,
                "test_randomizer": a.test_randomizer,
                "total_success": tot_s, "total_episodes": tot_e,
                "success_rate": tot_s / tot_e, "recorded_at": time.strftime("%Y-%m-%d")}
        a.out_csv.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
        print(f"wrote {a.out_csv.with_suffix('.meta.json')}")


def save_clip(frames: list, path: Path, fps: int = 10) -> None:
    """Write an agentview clip.

    The recorded frames are flipped vertically and converted RGB->BGR to match
    the display convention used throughout this project (agentview in LIBERO
    renders upside down relative to the on-screen window).
    """
    import cv2
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames:
        vw.write(cv2.cvtColor(np.asarray(fr)[::-1], cv2.COLOR_RGB2BGR))
    vw.release()


if __name__ == "__main__":
    main()