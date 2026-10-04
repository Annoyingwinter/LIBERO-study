# results/

Every file was written by a script. Nothing is transcribed from a lab note. The
"missing" section lists files that do not exist yet, so the gaps stay visible
instead of being quietly absent.

## Evaluation results — all five policies, one protocol

Protocol for every row: **50 benchmark init-states × 10 episodes = 500 episodes**,
max 400 steps, `env.seed(0)`, success = the env's own BDDL goal predicate.

| File | Policy | Inference view | Rate | Collapsed init-states¹ |
|---|---|---|---|---|
| `bcrnn_noaug50_full_eval.csv` | BC-RNN 50ep, no crop aug | full frame | 65/500 = 13.0% | 23 |
| `bcrnn_aug50_full_eval.csv` | BC-RNN 50ep, crop aug | random crop | 203/500 = 40.6% | 8 |
| `bcrnn_aug50_center.csv` | BC-RNN 50ep, crop aug | **centre crop** | **79/500 = 15.8%** | **27** |
| `bcrnn_noaug200_full_eval.csv` | BC-RNN 200ep, no crop aug | full frame | 260/500 = 52.0% | 10 |
| `bcrnn_aug200_full_eval.csv` | BC-RNN 200ep, crop aug | random crop | 439/500 = 87.8% | 0 |
| `bcrnn_aug200_center.csv` | BC-RNN 200ep, crop aug | **centre crop** | **433/500 = 86.6%** | 1 |
| `act_500ep_full_eval.csv` | ACT 500ep | random crop | 480/500 = 96.0% | 2 |

¹ all 10 episodes fail at that init-state.

All seven rows were re-summarised independently from the CSVs; the rates above are
what `scripts/plot_ablation.py` prints on every run.

### The two `_center` files are the same weights, re-evaluated

`bcrnn_aug{50,200}_center.csv` come from the same checkpoints as their
non-`_center` counterparts, run with `scripts/eval.py --test-randomizer center`.
The flag replaces `CropRandomizer.forward_in` with a deterministic centre crop of
the same 76×76 size.

**An identity bypass is not possible here** — robomimic sizes the encoder from the
randomizer's output shape at construction (`obs_nets.py:166`), so feeding the full
128×128 frame raises a shape assertion. A crop-augmented policy therefore cannot
see the full frame at all, by construction.

Effect of the change, and it is not uniform:

| arm | random | centre | Δ |
|---|---|---|---|
| aug 50ep | 40.6% | 15.8% | **−24.8 pts** |
| aug 200ep | 87.8% | 86.6% | −1.2 pts |

The 200-epoch number is robust; **the 50-epoch number was substantially inflated by
test-time input variability**, and README §2.1 quotes the centre-crop column for
that reason. Single seed only — see the missing-files table below.

## Failure census

| File | Model | init-state | success | locate_fail | align_fail | drop | other |
|---|---|---|---|---|---|---|---|
| `census_act_state13.csv` | ACT | 13 | 10 | 0 | 0 | 0 | 0 |
| `census_act_state4.csv` | ACT | 4 | 0 | 0 | 0 | 10 | 0 |
| `census_act_state46.csv` | ACT | 46 | 0 | 0 | 0 | 10 | 0 |

**These are the only failure-mode artifacts in the repo.** There is no BC-RNN census
output, so README §4.2 ("complementary failure modes") is currently evidenced for
ACT only.

## Perturbation — raw data, not a conclusion

| File | offsets | BC-RNN | ACT |
|---|---|---|---|
| `perturbation/day_perturbation_bcrnn_int.csv` | 0–50 mm | 80/50/0/0/0/0% | — |
| `perturbation/day_perturbation_act_int.csv` | 0–50 mm | — | 100/100/0/0/0/0% |
| `perturbation/day_perturbation_bcrnn.csv` | 12.5/15/17.5 mm | 60/**0**/60% | — |
| `perturbation/day_perturbation_act.csv` | 12.5/15/17.5 mm | — | 100/100/100% |
| `perturbation/day_perturbation_bcrnn_fine.csv` | 12.5/15/17.5 mm | 80/**0**/80% | — |
| `perturbation/day_perturbation_act_fine.csv` | 12.5/15/17.5 mm | — | 100/100/100% |

Three unresolved problems, detailed in README §3:

1. **BC-RNN collapses to exactly 0/10 at exactly 15.0 mm in both independent runs**,
   while 12.5 mm and 17.5 mm score 60–80%. Non-monotone; unexplained.
2. **`_fine` is a repeat run, not a finer sweep** — `day_perturbation.py:42`'s `TAG`
   is only an output-file suffix. Both files use the same three offsets. It does
   show run-to-run variance (60% vs 80%), which is worth keeping, but the name
   misleads.
3. **`day_plot_curve.py` silently drops one repeat**: it loads `int`, then `fine`,
   then the untagged file into one dict keyed by offset, so the last one wins and
   the `_fine` run never reaches the figure. It also plots missing offsets as 0%
   via `dict.get(x, 0)`.

## Earlier measurements

| File | What it is |
|---|---|
| `wk1_harness_placeholder_policy.csv` | Week-1 harness check: 20 episodes of a hard-coded placeholder policy, all fail at 100 steps. Validates the harness, **not** any policy. |
| `day1_state_transition.csv` | Single-axis action sweep (motion-model material) |
| `day1_zitai.csv` | End-effector orientation sweep (axis-angle) |

## Files that should exist and do not

| Missing | Blocks | How to produce |
|---|---|---|
| `census_bcrnn_*.csv` | README §4.2 — the "complementary failure modes" claim | `python scripts/failure_census.py --ckpt <bc_rnn_aug200.pth> --init-states 1 2 10 13 --episodes 10` |
| ACT under centre crop | §2.2 covered BC-RNN only; ACT's 96.0% is still random-view | `scripts/eval.py` has no ACT path — use `--test-randomizer center` added to `eval_act.py` |
| a second seed for `bcrnn_aug50_center.csv` | the 24.8-pt gap is §2.2's headline and rests on one seed | `scripts/eval.py --seed 1 --test-randomizer center ...` |
| a re-run perturbation sweep with more repeats at 14–16 mm | README §3 — the 15 mm hole | `python day_perturbation.py bcrnn 14,14.5,15,15.5,16` |
| multi-seed CSVs (`--seed 1/2/3`) | README §7.2 — no std is reported | `scripts/eval.py --seed 1 --out-csv ...` |
| eval CSVs for init-states other than task 0 | README §7.1 | `scripts/eval.py --task 1 ...` |
| rollout clips | README §2.5 | see `media/rollouts/README.md` |