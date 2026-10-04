"""ACT 训练入口（改编自官方 act/train.py：去 wandb/matplotlib，CSV 日志）。

数据路径默认相对仓库根目录解析（act/act_config.py），也可用 ACT_DATA 环境变量覆盖。

用法（在 libero venv 中）：
  cd act
  python train_act.py --smoke                      # 2 epoch 冒烟
  python train_act.py --epochs 500                 # 正式训练
  python train_act.py --verify-ckpt checkpoints/policy_latest_smoke.pth
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ACT_DIR = Path(__file__).resolve().parent
for p in (ACT_DIR, ACT_DIR / "act_src", ACT_DIR / "act_src" / "detr"):
    sys.path.insert(0, str(p))

from policy import ACTPolicy            # noqa: E402  vendored
from act_config import CFG, make_args_override   # noqa: E402
from act_dataset import build_loaders   # noqa: E402


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def forward_pass(policy, batch):
    image, qpos, action, is_pad = batch
    return policy(qpos.cuda(), image.cuda(), action.cuda(), is_pad.cuda())


def evaluate(policy, loader) -> dict:
    policy.eval()
    totals, n = {}, 0
    with torch.no_grad():
        for batch in loader:
            for k, v in forward_pass(policy, batch).items():
                totals[k] = totals.get(k, 0.0) + v.item()
            n += 1
    return {k: v / n for k, v in totals.items()}


def save_checkpoint(policy, stats, cfg, epoch: int, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": policy.state_dict(), "norm_stats": stats,
                "config": make_args_override(cfg), "epoch": epoch}, path)


def train(cfg, num_epochs: int, tag: str):
    set_seed(cfg.seed)
    assert torch.cuda.is_available(), "ACTPolicy 构建内部 model.cuda()，需要 GPU"
    train_loader, val_loader, stats, split = build_loaders(cfg.source_hdf5, cfg)
    print(f"train/val episodes: {len(split['train_ids'])}/{len(split['val_ids'])}")
    policy = ACTPolicy(make_args_override(cfg))
    optimizer = policy.configure_optimizers()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.log_dir / f"train_{tag}.csv"
    with open(log_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["epoch", "step", "split", "l1", "kl", "loss"])
        for epoch in range(num_epochs):
            policy.train()
            for step, batch in enumerate(train_loader):
                t0 = time.time()
                loss_dict = forward_pass(policy, batch)
                optimizer.zero_grad()
                loss_dict["loss"].backward()
                optimizer.step()
                writer.writerow([epoch, step, "train",
                                 f"{loss_dict['l1'].item():.5f}",
                                 f"{loss_dict['kl'].item():.4f}",
                                 f"{loss_dict['loss'].item():.5f}"])
                fcsv.flush()
                if step % 10 == 0:
                    print(f"epoch {epoch} step {step} loss {loss_dict['loss'].item():.4f} "
                          f"({time.time()-t0:.2f}s)")
            val_means = evaluate(policy, val_loader)
            writer.writerow([epoch, -1, "val", f"{val_means['l1']:.5f}",
                             f"{val_means['kl']:.4f}", f"{val_means['loss']:.5f}"])
            fcsv.flush()
            print(f"== epoch {epoch}: val loss {val_means['loss']:.4f} ==")
            save_checkpoint(policy, stats, cfg, epoch, cfg.checkpoint_dir / f"policy_latest_{tag}.pth")
    vram = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"VRAM peak: {vram:.2f} GiB | log: {log_path}")
    assert vram < 7.5, "显存超 7.5GB：按 spec 第 10 节降 batch 8→4 或 chunk 50→25"
    return policy, stats


def verify_checkpoint(cfg, ckpt_path: Path):
    """V4：加载 checkpoint，val batch 推理，断言 a_hat (B, chunk, 7) 且有限。"""
    ckpt = torch.load(ckpt_path, map_location="cuda")
    _, val_loader, _, _ = build_loaders(cfg.source_hdf5, cfg, stats=ckpt["norm_stats"])
    policy = ACTPolicy(ckpt["config"])
    policy.load_state_dict(ckpt["model"])
    image, qpos, _, _ = next(iter(val_loader))
    with torch.no_grad():
        a_hat = policy(qpos.cuda(), image.cuda())
    assert a_hat.shape == (qpos.shape[0], cfg.chunk_size, 7), f"a_hat shape {tuple(a_hat.shape)}"
    assert torch.isfinite(a_hat).all()
    print(f"V4 VERIFY-CKPT PASS: a_hat {tuple(a_hat.shape)} finite, from {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="2 epoch 冒烟训练（V2）")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--verify-ckpt", type=str, default=None)
    args = parser.parse_args()
    if args.verify_ckpt:
        verify_checkpoint(CFG, Path(args.verify_ckpt))
        return
    if args.smoke:
        train(CFG, 2, tag="smoke")
        return
    train(CFG, args.epochs or CFG.num_epochs, tag="full")


if __name__ == "__main__":
    main()
