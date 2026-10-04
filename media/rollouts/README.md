# Rollout clips — currently empty, on purpose

There are 89 recorded rollout clips in the working directory
(`eval_videos/model_epoch_{50,100,150,200}/`), produced by the ad-hoc eval
script `day_policy_eval.py`. **None of them are published here yet**, for two
reasons that both have to be cleared before anything goes in this directory.

## Reason 1 — the directory is contaminated, so no aggregate number from it is usable

`day_policy_eval.py` derived its output directory from the checkpoint filename
alone:

```python
VIDEO_DIR = os.path.join("<REPO>/eval_videos",
                         os.path.basename(CKPT).replace(".pth", ""))
```

Two different experiments both own a checkpoint named `model_epoch_50.pth`:

```
robomimic_out/bc_rnn_libero_task0/20260927153018/models/model_epoch_50.pth   # no aug
robomimic_out/bc_rnn_aug_200ep/20260927165848/models/model_epoch_50.pth      # crop-aug
```

Both wrote into `eval_videos/model_epoch_50/` and overwrote each other by
episode index. `model_epoch_200/` shows the signature: **29 files for a
20-episode run** — all 20 `_SUCCESS` clips plus 9 `_fail` clips, i.e. leftovers
from a second run.

Consequences, stated plainly:
- The success rate derivable from those filenames is meaningless.
- The `5% (no augmentation) -> 87.8% (crop augmentation)` comparison in
  `docs/notes/day6note.md` rests on a run whose artifact directory is
  contaminated. **The 5% figure in that note is not currently supported by any
  surviving artifact.** The 87.8% figure is unaffected: it comes from
  `results/bc_rnn_200ep_full_eval.csv`, produced by a script that recorded no
  video at all.

`scripts/eval.py` fixes the naming (keyed on experiment id + init-state id) and
is the script to use from now on.

## Reason 2 — 128×128 is not legible enough to verify the failure mode

Checked on a **freshly recorded, uncontaminated** clip (`scripts/eval.py`,
init-state 13, 400 frames, 0/2 success — consistent with the 1/10 in the 500-episode
run). Sampling 5 frames across it:

- **Legible**: the cabinet, the plate (white ring), the ramekin (grey cup), the
  cookie box, the arm.
- **Not legible**: the black bowl starts at frame 0 to the right of the ramekin,
  but whether it is subsequently nudged or lifted **cannot be determined**.

So the more fundamental problem is not one mislabelled clip — it is that at this
resolution **neither I nor a reviewer can verify the failure analysis**. The
claim in `docs/notes/day7note.md` that the gripper reached the bowl in 7/7
failure clips but closed on it in 0/7 currently has no publishable visual
evidence behind it.

The older `ep04_SUCCESS.mp4` additionally appears to show the bowl unmoved
throughout, which is inconsistent with its label — but at this resolution that
is not conclusive either.

## What to do instead

Record fresh clips at a resolution that actually reads, from a clean run, using
the fixed script (its `--video-dir` branch is smoke-tested and works):

```bash
python scripts/eval.py \
    --ckpt outputs/bc_rnn_aug_200ep/<timestamp>/models/model_epoch_200.pth \
    --episodes-per-init 10 --seed 0 \
    --out-csv results/bc_rnn_200ep_rerun.csv \
    --video-dir media/rollouts --video-successes 1 --video-failures 1 \
    --experiment-id bc_rnn_200ep
```

One change needed first, because the current script inherits the limitation from
the original:

- **Resolution.** `scripts/eval.py` builds the env at 128×128 to match the
  training data, and that is correct for *inference*. For *recording*, add a
  `--record-camera-size` flag that builds a second env at 512×512 — or record
  the wrist camera (`eye_in_hand`) as well, which is usually the more legible
  view of a grasp. **Note that the observation pipeline and the recording
  pipeline must be kept separate**: raising the camera resolution for recording
  must not change what the policy is fed, or the recorded rollout is no longer
  the rollout that was scored.

Also point `--init-states` at the failing configurations, so the failure clip is
actually a failure:

```bash
--init-states 1 2 10 13
```