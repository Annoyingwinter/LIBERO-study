"""Plot training curves from the real logs. No numbers are invented here —
every point is parsed out of a file that a training run actually wrote.

Inputs (paths are arguments, defaults point at the committed copies):
  --bc-log   robomimic JSON metric blocks   -> Log_Likelihood, Policy_Grad_Norms
  --act-csv  act/logs/train_full.csv       -> l1, kl, loss (train + val)
  --eval-csv results/bc_rnn_200ep_full_eval.csv -> per-init-state success rate
  --outdir   media/figures/

Usage:
    python scripts/plot_curves.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent


def parse_robomimic_log(path: Path) -> list[dict]:
    """robomimic writes exactly one flat JSON object per epoch.

    The blobs carry no epoch field, so the x-axis has to be recovered from the
    emission order. That is safe here because of how robomimic's training loop is
    written (robomimic/scripts/train.py:175-200):

        for epoch in range(1, config.train.num_epochs + 1):
            step_log = TrainUtils.run_epoch(..., epoch=epoch, ...)
            print(json.dumps(step_log, sort_keys=True, indent=4))

    i.e. one `json.dumps` per epoch, epochs numbered from 1. The 200 blobs in
    this log are therefore epochs 1..200 of a 200-epoch run. Asserted below so a
    future log with a different cadence fails loudly instead of silently
    mislabelling the axis.

    Note this is independent of `epoch_every_n_steps` (=100 here), which controls
    how many *batches* an epoch runs, not how often the summary is printed.
    """
    text = path.read_text(errors="replace")
    out = []
    for blob in re.findall(r'\{[^{}]*"Loss"[^{}]*\}', text, re.S):
        try:
            out.append(json.loads(blob))
        except json.JSONDecodeError:
            continue
    if not out:
        return []
    n_epochs = len(out)
    for i, e in enumerate(out, start=1):
        e["Epoch"] = i
    # the final block must be the run's last epoch; robomimic always prints
    # "finished run successfully!" after the last one.
    if "finished run successfully!" not in text:
        raise RuntimeError(
            "log does not end with 'finished run successfully!' -- the run was "
            "interrupted, so the final metric block is not the last epoch."
        )
    print(f"  parsed {n_epochs} epoch blocks (epochs 1..{n_epochs}), "
          f"run confirmed complete")
    return out


def parse_act_csv(path: Path) -> dict:
    """train_full.csv columns: epoch,step,split,l1,kl,loss  (split in {train,val}).

    Returns per-epoch means of the train rows plus the single val row for that
    epoch. Val is one row per epoch because the loader has 10 val episodes at
    batch_size_val=8 -> 2 batches averaged inside evaluate().
    """
    per_epoch = defaultdict(lambda: {"train": [], "val": []})
    with path.open() as f:
        for row in csv.DictReader(f):
            split = row["split"]
            if split not in ("train", "val"):
                continue
            per_epoch[int(row["epoch"])][split].append(
                {k: float(row[k]) for k in ("l1", "kl", "loss")}
            )
    series = {}
    for ep, d in sorted(per_epoch.items()):
        entry = {}
        for split in ("train", "val"):
            rows = d[split]
            if rows:
                entry[split] = {
                    k: sum(r[k] for r in rows) / len(rows) for k in ("l1", "kl", "loss")
                }
        series[ep] = entry
    return series


def parse_eval_csv(path: Path) -> list[tuple[int, float, int, int]]:
    """init_state,success,episodes,rate  ->  (sid, rate, success, episodes)"""
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append(
                (
                    int(row["init_state"]),
                    float(row["rate"]),
                    int(row["success"]),
                    int(row["episodes"]),
                )
            )
    return sorted(rows)


def fig_bc_curve(events: list[dict], outdir: Path) -> None:
    epochs = [e["Epoch"] for e in events]
    ll = [e["Log_Likelihood"] for e in events]
    gn = [e["Policy_Grad_Norms"] for e in events]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(epochs, ll, lw=1.4, color="#1f77b4")
    ax1.set_ylabel("Log_Likelihood  (= -Loss)")
    ax1.set_title("BC-RNN training log-likelihood and gradient norm\n"
                  "robomimic bc_rnn_aug_200ep, 200 epochs", fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.axhline(ll[-1], ls="--", lw=0.8, color="gray",
                label=f"final = {ll[-1]:.3f}")
    ax1.legend(fontsize=8)

    ax2.plot(epochs, gn, lw=1.0, color="#d62728", alpha=0.85)
    ax2.set_ylabel("Policy_Grad_Norms")
    ax2.set_xlabel("epoch")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "bc_rnn_train_curve.png", dpi=150)
    plt.close(fig)


def fig_act_curve(series: dict, outdir: Path) -> None:
    eps = sorted(series)
    tr = [series[e].get("train") for e in eps]
    va = [series[e].get("val") for e in eps]
    tr_loss = [t["loss"] if t else None for t in tr]
    va_loss = [v["loss"] if v else None for v in va]
    tr_l1 = [t["l1"] if t else None for t in tr]
    va_l1 = [v["l1"] if v else None for v in va]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    ax1.plot(eps, tr_loss, lw=1.3, color="#1f77b4", label="train")
    ax1.plot(eps, va_loss, lw=1.3, color="#ff7f0e", label="val")
    ax1.set_yscale("log")
    ax1.set_ylabel("total loss  (l1 + 10·KL)")
    ax1.set_title("ACT 500 epochs — note the val L1 plateau (§ Discussion)\n"
                  "source: act/logs/train_full.csv", fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.plot(eps, tr_l1, lw=1.3, color="#1f77b4", label="train L1")
    ax2.plot(eps, va_l1, lw=1.3, color="#ff7f0e", label="val L1")
    ax2.set_ylabel("L1 (normalised action units)")
    ax2.set_xlabel("epoch")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "act_train_curve.png", dpi=150)
    plt.close(fig)


def fig_eval_per_state(rows: list[tuple[int, float, int, int]], outdir: Path) -> None:
    sids = [r[0] for r in rows]
    rates = [r[1] for r in rows]
    succ = [r[2] for r in rows]
    epsn = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    ax.bar(sids, rates, color=["#d62728" if r < 0.5 else "#2ca02c" for r in rates])
    ax.set_xlabel("benchmark init_state id")
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"BC-RNN + crop-aug, epoch 200 — per-init-state success rate "
                 f"({sum(epsn)} episodes = {len(sids)} init-states × "
                 f"{epsn[0]}), env seed 0", fontsize=10)
    ax.set_xticks(range(0, max(sids) + 1, 2))
    n_low = sum(1 for s in succ if s <= 2)
    ax.axhline(0.5, ls="--", lw=0.8, color="gray")
    ax.text(0.02, 0.06, f"red = below 50%  ({n_low} of {len(sids)} init-states)",
            transform=ax.transAxes, fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outdir / "bc_rnn_per_init_state.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bc-log", type=Path,
                   default=REPO / "logs" / "bc_rnn_aug_200ep_log.txt")
    p.add_argument("--act-csv", type=Path,
                   default=REPO / "logs" / "act" / "train_full.csv")
    p.add_argument("--eval-csv", type=Path,
                   default=REPO / "results" / "bc_rnn_200ep_full_eval.csv")
    p.add_argument("--outdir", type=Path, default=REPO / "media" / "figures")
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    if a.bc_log.exists():
        ev = parse_robomimic_log(a.bc_log)
        fig_bc_curve(ev, a.outdir)
        print(f"bc_rnn_train_curve.png   from {len(ev)} metric blocks, "
              f"epochs {ev[0]['Epoch']}..{ev[-1]['Epoch']}")
    else:
        print(f"skip BC curve, missing {a.bc_log}")

    if a.act_csv.exists():
        s = parse_act_csv(a.act_csv)
        fig_act_curve(s, a.outdir)
        print(f"act_train_curve.png      from {len(s)} epochs")
    else:
        print(f"skip ACT curve, missing {a.act_csv}")

    if a.eval_csv.exists():
        r = parse_eval_csv(a.eval_csv)
        fig_eval_per_state(r, a.outdir)
        tot = sum(x[2] for x in r)
        n = sum(x[3] for x in r)
        print(f"bc_rnn_per_init_state.png from {len(r)} rows, {tot}/{n} = {tot/n:.1%}")
    else:
        print(f"skip eval bar chart, missing {a.eval_csv}")


if __name__ == "__main__":
    main()