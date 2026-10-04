"""LIBERO hdf5 → ACT 张量的数据层（只读，原地读，零拷贝）。

接口与官方 act/utils.py 的 EpisodicDataset 对齐，但有三处 LIBERO 适配：
  1. 直接读单文件多 demo（官方为每集一个文件）；
  2. action/is_pad pad 到 chunk_size（官方 pad 到整集长度——官方每集等长所以能 collate，
     我们变长（98~155 步），pad 到 chunk_size 后 policy 内的 [:, :num_queries] 截断成为空操作，语义不变）；
  3. qpos 支持 robot_states（ee 空间 9 维，默认）与 joint_gripper（关节 9 维）两种表征。
qvel 官方读了但不返回、ACT 不使用 → 完全不读。
LIBERO 是仿真环境 → 采用官方 sim=True 语义（action 从 start_ts 起截取，无 real 机器人的 -1 hack）。
"""
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader


def _demo_key(episode_id: int) -> str:
    return f"data/demo_{episode_id}"


def load_episode_lengths(source_file) -> dict:
    with h5py.File(source_file, "r") as f:
        return {int(k.split("_")[1]): f[f"data/{k}"]["actions"].shape[0] for k in f["data"]}


def read_qpos(demo_group, qpos_source: str) -> np.ndarray:
    if qpos_source == "robot_states":
        return demo_group["robot_states"][()]
    if qpos_source == "joint_gripper":
        return np.concatenate([demo_group["obs/joint_states"][()],
                               demo_group["obs/gripper_states"][()]], axis=1)
    raise ValueError(f"unknown qpos_source: {qpos_source}")


def split_episodes(num_episodes: int, seed: int, train_ratio: float):
    rs = np.random.RandomState(seed)
    shuffled = rs.permutation(num_episodes)
    n_train = int(train_ratio * num_episodes)
    return shuffled[:n_train], shuffled[n_train:]


def get_norm_stats(source_file, episode_ids, qpos_source: str) -> dict:
    """官方 get_norm_stats 语义：std 下限 1e-2；官方对等长集 stack 后 mean/std，
    变长集用 cat 沿时间维等价实现。仅统计给定 episode_ids（spec 第 7 节：仅 train 集）。"""
    all_qpos, all_action = [], []
    with h5py.File(source_file, "r") as f:
        for eid in episode_ids:
            g = f[_demo_key(eid)]
            all_qpos.append(torch.from_numpy(read_qpos(g, qpos_source)).float())
            all_action.append(torch.from_numpy(g["actions"][()]).float())
    qpos = torch.cat(all_qpos, dim=0)
    action = torch.cat(all_action, dim=0)
    stats = {
        "qpos_mean": qpos.mean(dim=0).numpy(),
        "qpos_std": torch.clip(qpos.std(dim=0), 1e-2, np.inf).numpy(),
        "action_mean": action.mean(dim=0).numpy(),
        "action_std": torch.clip(action.std(dim=0), 1e-2, np.inf).numpy(),
    }
    return stats


def load_or_compute_stats(source_file, episode_ids, qpos_source, cache_path) -> dict:
    fingerprint = {"source": str(source_file), "qpos_source": qpos_source,
                   "episode_ids": sorted(int(i) for i in episode_ids)}
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("fingerprint") == fingerprint:
            return cached["stats"]
    stats = get_norm_stats(source_file, episode_ids, qpos_source)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"fingerprint": fingerprint, "stats": stats}, f)
    return stats


class LiberoEpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, source_file, camera_names, norm_stats, qpos_source, chunk_size):
        self.episode_ids = list(int(i) for i in episode_ids)
        self.source_file = str(source_file)
        self.camera_names = list(camera_names)
        self.norm_stats = norm_stats
        self.qpos_source = qpos_source
        self.chunk_size = chunk_size
        self._lengths = load_episode_lengths(self.source_file)
        with h5py.File(self.source_file, "r") as f:   # 校验字段齐全（只读）
            g0 = f[_demo_key(self.episode_ids[0])]
            for cam in self.camera_names:
                assert f"obs/{cam}_rgb" in g0, f"缺相机字段 obs/{cam}_rgb"
            assert "actions" in g0 and "robot_states" in g0

    def __len__(self):
        return len(self.episode_ids)

    def sample_at(self, index: int, start_ts: int):
        eid = self.episode_ids[index]
        with h5py.File(self.source_file, "r") as f:
            g = f[_demo_key(eid)]
            T = g["actions"].shape[0]
            assert 0 <= start_ts < T
            qpos = read_qpos(g, self.qpos_source)[start_ts]                     # (9,)
            imgs = np.stack([g[f"obs/{cam}_rgb"][start_ts] for cam in self.camera_names], axis=0)  # (k,H,W,C)
            action = g["actions"][start_ts:start_ts + self.chunk_size]          # (t<=chunk, 7)
        t = action.shape[0]
        padded = np.zeros((self.chunk_size, action.shape[1]), dtype=np.float32)
        padded[:t] = action
        is_pad = np.zeros(self.chunk_size, dtype=bool)
        is_pad[t:] = True
        image = torch.from_numpy(imgs)
        image = torch.einsum("k h w c -> k c h w", image).float() / 255.0       # (k,C,H,W) ∈ [0,1]
        qpos_t = (torch.from_numpy(qpos).float() - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        action_t = (torch.from_numpy(padded).float() - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        return image, qpos_t, action_t, torch.from_numpy(is_pad)

    def __getitem__(self, index):
        start_ts = np.random.choice(self._lengths[self.episode_ids[index]])
        return self.sample_at(index, start_ts)


def build_loaders(source_file, cfg, stats=None):
    lengths = load_episode_lengths(source_file)
    train_ids, val_ids = split_episodes(len(lengths), cfg.seed, cfg.train_ratio)
    if stats is None:
        stats = load_or_compute_stats(source_file, train_ids.tolist(), cfg.qpos_source, cfg.stats_path)
    common = dict(source_file=source_file, camera_names=cfg.camera_names,
                  norm_stats=stats, qpos_source=cfg.qpos_source, chunk_size=cfg.chunk_size)
    train_ds = LiberoEpisodicDataset(train_ids.tolist(), **common)
    val_ds = LiberoEpisodicDataset(val_ids.tolist(), **common)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size_train, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True,
                              prefetch_factor=1 if cfg.num_workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size_val, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True,
                            prefetch_factor=1 if cfg.num_workers > 0 else None)
    return train_loader, val_loader, stats, {"train_ids": train_ids.tolist(), "val_ids": val_ids.tolist()}


def self_check():
    """V1 自检：真实数据上的形状/取值断言。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from act_config import CFG

    lengths = load_episode_lengths(CFG.source_hdf5)
    vals = list(lengths.values())
    print(f"episodes: {len(vals)}, T min/max/mean = {min(vals)}/{max(vals)}/{np.mean(vals):.1f}")
    train_loader, val_loader, stats, split = build_loaders(CFG.source_hdf5, CFG)
    print(f"train/val episodes: {len(split['train_ids'])}/{len(split['val_ids'])}")
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        img, qpos, act, pad = next(iter(loader))
        print(f"{name} batch: img {tuple(img.shape)} {img.dtype}, qpos {tuple(qpos.shape)}, "
              f"act {tuple(act.shape)}, pad {tuple(pad.shape)} {pad.dtype}")
        assert img.dtype == torch.float32 and img.max() <= 1.0
        assert qpos.shape[1] == 9 and act.shape[1] == CFG.chunk_size and act.shape[2] == 7
        assert pad.dtype == torch.bool
    ds = val_loader.dataset
    eid = ds.episode_ids[0]
    T = lengths[eid]
    _, _, _, pad = ds.sample_at(0, T - 1)   # 集末边界 → 必有 pad
    assert not pad[0].item()
    assert pad[-1].item()
    print("V1 SELF-CHECK PASS")


if __name__ == "__main__":
    self_check()
