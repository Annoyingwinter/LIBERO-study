"""V0：ACTPolicy 在锁定 venv + torch 2.2.2 下构建/前向/反向。
运行: /home/flexing/libero/venv/bin/python tests/test_policy_build.py
首次运行会下载 resnet18 ImageNet 权重（~45MB，已缓存则秒过）。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "act_src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "act_src" / "detr"))
from policy import ACTPolicy          # noqa: E402
from act_config import CFG, make_args_override  # noqa: E402


def main():
    assert torch.cuda.is_available(), "ACTPolicy 构建内部 model.cuda()，需要 GPU"
    policy = ACTPolicy(make_args_override(CFG))
    print("ACTPolicy built:", sum(x.numel() for x in policy.model.parameters()), "params")
    B = 2
    img = torch.rand(B, len(CFG.camera_names), 3, 128, 128).cuda()
    qpos = torch.randn(B, 9).cuda()
    act = torch.randn(B, CFG.chunk_size, 7).cuda()
    pad = torch.zeros(B, CFG.chunk_size, dtype=torch.bool).cuda()
    loss_dict = policy(qpos, img, act, pad)
    loss_dict["loss"].backward()
    assert torch.isfinite(loss_dict["loss"]), f"loss not finite: {loss_dict}"
    with torch.no_grad():
        a_hat = policy(qpos, img)
    assert a_hat.shape == (B, CFG.chunk_size, 7), f"a_hat shape {a_hat.shape}"
    assert torch.isfinite(a_hat).all()
    print("TEST-POLICY-BUILD PASS")


if __name__ == "__main__":
    main()
