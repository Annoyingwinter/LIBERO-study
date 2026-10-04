"""act_dataset 测试（TDD；合成 mini LIBERO hdf5，不碰真实数据）。
运行: /home/flexing/libero/venv/bin/python tests/test_dataset.py
"""
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from act_dataset import (LiberoEpisodicDataset, get_norm_stats, load_or_compute_stats,
                         load_episode_lengths, split_episodes)


def make_mini_libero(path, lengths=(4, 6, 5)):
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for i, T in enumerate(lengths):
            g = data.create_group(f"demo_{i}")
            g.create_dataset("actions", data=np.random.randn(T, 7).astype(np.float64))
            obs = g.create_group("obs")
            obs.create_dataset("agentview_rgb",
                               data=np.random.randint(0, 256, (T, 128, 128, 3), dtype=np.uint8))
            obs.create_dataset("eye_in_hand_rgb",
                               data=np.random.randint(0, 256, (T, 128, 128, 3), dtype=np.uint8))
            obs.create_dataset("joint_states", data=np.random.randn(T, 7).astype(np.float64))
            obs.create_dataset("gripper_states", data=np.random.randn(T, 2).astype(np.float64))
            g.create_dataset("robot_states", data=np.random.randn(T, 9).astype(np.float64))


def main():
    tmp = Path(tempfile.mkdtemp())
    mini = tmp / "mini.hdf5"
    make_mini_libero(mini)

    # 1) 长度表
    assert load_episode_lengths(str(mini)) == {0: 4, 1: 6, 2: 5}

    # 2) 划分：确定性、不重叠、比例 0.8
    tr, va = split_episodes(50, seed=42, train_ratio=0.8)
    tr2, va2 = split_episodes(50, seed=42, train_ratio=0.8)
    assert tr.tolist() == tr2.tolist() and va.tolist() == va2.tolist()
    assert sorted(tr.tolist() + va.tolist()) == list(range(50))
    assert len(tr) == 40 and len(va) == 10

    # 3) 归一化统计形状
    stats = get_norm_stats(str(mini), [0, 1, 2], "robot_states")
    assert stats["qpos_mean"].shape == (9,) and stats["action_mean"].shape == (7,)
    assert (stats["qpos_std"] > 0).all() and (stats["action_std"] > 0).all()

    # 3b) 两种 qpos 表征都是 9 维
    s2 = get_norm_stats(str(mini), [0], "joint_gripper")
    assert s2["qpos_mean"].shape == (9,)

    # 4) 样本形状 + padding 边界（chunk=3, demo_0 T=4, ts=3 → 只剩 1 步）
    ds = LiberoEpisodicDataset([0, 1, 2], str(mini), ("agentview", "eye_in_hand"),
                               stats, "robot_states", chunk_size=3)
    img, qpos, act, pad = ds.sample_at(0, 3)
    assert img.shape == (2, 3, 128, 128) and img.dtype == torch.float32 and img.max() <= 1.0
    assert qpos.shape == (9,) and qpos.dtype == torch.float32
    assert act.shape == (3, 7) and act.dtype == torch.float32
    assert pad.dtype == torch.bool and pad.tolist() == [False, True, True]
    img, qpos, act, pad = ds.sample_at(1, 1)   # demo_1 T=6 → 满 chunk，无 pad
    assert pad.sum() == 0 and act.shape == (3, 7)
    img, qpos, act, pad = ds[0]                # 随机 __getitem__ 形状一致
    assert img.shape == (2, 3, 128, 128) and act.shape == (3, 7)

    # 5) 统计缓存：命中 / 指纹变化重算
    cache = tmp / "stats.pkl"
    s1 = load_or_compute_stats(str(mini), [0, 1, 2], "robot_states", cache)
    s1b = load_or_compute_stats(str(mini), [0, 1, 2], "robot_states", cache)
    assert np.allclose(s1["qpos_mean"], s1b["qpos_mean"])
    s3 = load_or_compute_stats(str(mini), [0, 1, 2], "joint_gripper", cache)
    assert not np.allclose(s1["qpos_mean"], s3["qpos_mean"])

    print("TEST-DATASET PASS")


if __name__ == "__main__":
    main()
