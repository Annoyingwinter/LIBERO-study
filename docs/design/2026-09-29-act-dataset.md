# ACT 数据集接入实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 LIBERO task0 hdf5 原地接入 vendored ACT，端到端验证到 smoke train（V1 数据自检 / V2 GPU 冒烟 / V3 主流程无恙 / V4 checkpoint 回推）。

**Architecture:** vendor ACT 官方源码（仅 3 类已实测补丁）+ 自写数据适配层原地读单文件 hdf5 + 训练入口（CSV 日志）。数据流：hdf5 → `EpisodicDataset.__getitem__` → (image, qpos, action, is_pad) 归一化 → ACTPolicy。

**Tech Stack:** Python 3.10.12（固定用 `<REPO>/venv/bin/python`），torch 2.2.2+cu121，h5py，einops。零 pip 安装。

**Spec:** `docs/superpowers/specs/2026-09-29-act-dataset-design.md`

## Global Constraints（每个任务隐含遵守）

- 解释器固定 `<REPO>/venv/bin/python`；**零 pip install**。
- 新文件仅限 `<REPO>/act/` 与 `docs/superpowers/`；现有文件只读（含 venv、LIBERO repo、hdf5）。
- hdf5 一律 `h5py.File(path, "r")`；数据源固定 `<REPO>/task0_nextobs.hdf5`。
- 本项目**不是 git 仓库**：无 commit 步骤，以 Task 5 的 V3 检查替代提交纪律。
- 日志走 CSV；不引入 wandb / matplotlib / IPython / pytest（测试为纯 assert 脚本）。
- 超参以 `act_config.py` 为唯一事实来源，值照抄 spec 第 8 节。

## 探针已验证的事实（2026-09-29，实施时不必重验）

1. **vendored 布局需要双路径**：上游 `detr/models/backbone.py:14` 与 `position_encoding.py:9` 用绝对导入 `from util.misc import ...`，仅 `act_src` 在 sys.path 时报 `No module named 'util'`；`act_src` + `act_src/detr` 双路径加入后 `import policy` 通过（实测）。
2. **3 类补丁（全部实测通过）**：
   a. 剥离 `import IPython` / `e = IPython.embed`：`policy.py`、`utils.py`、`detr/main.py`；
   b. `detr/main.py` 两处 `args = parser.parse_args()` → `parse_args([占位值])`（`set_defaults` 实测**不能**满足 `required=True`）；
   c. `detr/models/detr_vae.py` 把写死的 14 参数化为 `action_dim`（动作 7 维）/`qpos_dim`（本体 9 维），共 7 处（含 `build()` 里的 `state_dim=14`）；`build_cnnmlp` 保持原样（未使用）。
3. **端到端探针**（batch 2, 128×128, chunk 50）：loss 89.07（l1 0.83 / kl 8.82），`a_hat (2,50,7)` 有限，**VRAM 峰值 0.77 GiB**；resnet18 预训练权重从 download.pytorch.org 下载成功（已缓存 `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`）。
4. **下载通道**：github.com 直连不稳定（clone/codeload 超时），**jsDelivr CDN 可用**；上游 HEAD = `742c753c0d4a5d87076c8f69e5628c79a8cc5488`。
5. `/tmp/act_probe/` 内已有打好全部补丁并通过探针的副本（`detr/`、`policy.py`、`utils.py`）；`/tmp/act_upstream/` 有未打补丁的原始文件（含 `train.py`）。

---

### Task 1: 安全快照 + vendor 源码

**Files:**
- Create: `act/pre_freeze.txt`、`act/act_src/{policy.py,utils.py,train.py}`、`act/act_src/detr/**`、`act/act_src/PROVENANCE.md`

**Interfaces:**
- Produces: 可 import 的 `policy.ACTPolicy`（sys.path 含 `act_src` 与 `act_src/detr` 时）；后续任务依赖的 vendored 包。

- [ ] **Step 1: pip freeze 快照（V3 参照点，先于一切写操作）**

```bash
mkdir -p <REPO>/act
<REPO>/venv/bin/pip freeze > <REPO>/act/pre_freeze.txt
```

- [ ] **Step 2: 复制已打补丁的探针副本（首选）；不存在则走 jsDelivr + 补丁脚本（回退路径，见 Step 2b）**

```bash
rm -rf <REPO>/act/act_src
mkdir -p <REPO>/act/act_src
cp -r /tmp/act_probe/detr /tmp/act_probe/policy.py /tmp/act_probe/utils.py <REPO>/act/act_src/
cp /tmp/act_upstream/train.py <REPO>/act/act_src/train.py   # 仅参考：未打补丁、不可 import（缺 constants.py），PROVENANCE 注明
```

- [ ] **Step 2b（仅当 /tmp 副本不存在时）: jsDelivr 逐文件下载 + 重放补丁**

下载清单与命令（对每个文件依次尝试 cdn.jsdelivr.net / fastly.jsdelivr.net）：
`https://<host>/gh/tonyzhaozh/act@main/<path>`，path ∈ {`policy.py`,`utils.py`,`train.py`,`detr/main.py`,`detr/models/__init__.py`,`detr/models/backbone.py`,`detr/models/detr_vae.py`,`detr/models/position_encoding.py`,`detr/models/transformer.py`,`detr/util/__init__.py`,`detr/util/misc.py`,`detr/util/box_ops.py`}，用 `py_compile` 验证。补丁重放脚本 = Step 2c。

- [ ] **Step 2c: 补丁重放脚本（打在 vendored 副本上；探针副本已含，重跑应全部 skip）**

```python
#!/usr/bin/env python
"""apply_patches.py — 对 vendored ACT 源码重放 3 类补丁（幂等）。用法: python apply_patches.py <act_src_dir>"""
import pathlib, sys
root = pathlib.Path(sys.argv[1])

def patch(rel, reps):
    p = root / rel; s = p.read_text(); s2 = s
    for old, new, n in reps:
        if s2.count(new) >= n and s2.count(old) == 0:
            continue  # already applied
        assert s2.count(old) == n, f"{rel}: count {s2.count(old)} != {n} for {old[:60]!r}"
        s2 = s2.replace(old, new)
    p.write_text(s2)

ip = [("import IPython\n", "", 1), ("e = IPython.embed\n", "", 1)]
for rel in ["policy.py", "utils.py", "detr/main.py"]:
    patch(rel, ip)

pa_old = "    args = parser.parse_args()"
pa_new = ("    # vendored patch: 外部 import 调用，不消费调用方 argv；必需参数给占位值，全部被 args_override 覆盖\n"
          "    args = parser.parse_args(['--ckpt_dir', '', '--policy_class', 'ACT',\n"
          "                              '--task_name', '', '--seed', '1', '--num_epochs', '1'])")
patch("detr/main.py", [(pa_old, pa_new, 2)])

patch("detr/models/detr_vae.py", [
 ("def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names):",
  "def __init__(self, backbones, transformer, encoder, action_dim, qpos_dim, num_queries, camera_names):", 1),
 ("self.action_head = nn.Linear(hidden_dim, state_dim)",
  "self.action_head = nn.Linear(hidden_dim, action_dim)", 1),
 ("self.input_proj_robot_state = nn.Linear(14, hidden_dim)",
  "self.input_proj_robot_state = nn.Linear(qpos_dim, hidden_dim)", 2),
 ("self.encoder_action_proj = nn.Linear(14, hidden_dim) # project action to embedding",
  "self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding", 1),
 ("self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding",
  "self.encoder_joint_proj = nn.Linear(qpos_dim, hidden_dim)  # project qpos to embedding", 1),
 ("def build(args):\n    state_dim = 14 # TODO hardcode",
  "def build(args):\n    action_dim = args.action_dim # vendored patch: was state_dim = 14 hardcode\n    qpos_dim = args.qpos_dim", 1),
 ("        state_dim=state_dim,\n        num_queries=args.num_queries,",
  "        action_dim=action_dim, qpos_dim=qpos_dim,\n        num_queries=args.num_queries,", 1),
])
print("patches OK")
```

- [ ] **Step 3: 补丁后审计**

```bash
cd <REPO>/act/act_src
grep -rn "IPython" . && echo "FAIL: IPython remains" || echo "IPython: clean"
grep -rn "^import matplotlib\|^import wandb\|^import timm\|^import pytorch_lightning" . || echo "heavy deps: clean"
<REPO>/venv/bin/python -m py_compile policy.py utils.py detr/main.py detr/models/*.py detr/util/*.py && echo "compile: OK"
```
Expected: 三行 clean/OK。

- [ ] **Step 4: import 冒烟（无需 GPU、无需权重）**

```bash
cd <REPO>/act && <REPO>/venv/bin/python -c "
import sys
sys.path.insert(0, 'act_src'); sys.path.insert(0, 'act_src/detr')
import utils, policy
print('vendored import OK')"
```
Expected: `vendored import OK`。

- [ ] **Step 5: 写 PROVENANCE.md**

内容必须包含：上游 repo/commit（`tonyzhaozh/act@742c753c0d4a5d87076c8f69e5628c79a8cc5488`）、复制日期、 vendored 文件清单、3 类补丁逐条说明（文件、原行为、新行为、原因：①venv 无 IPython 且仅为调试入口 ②parse_args 会消费调用方 argv ③action/qpos 维度从 ALOHA 的 14 改为参数化 7/9 以适配 LIBERO）、`train.py` 为未打补丁的参考件不可 import、CNNMLP 路径未适配（未使用）。

---

### Task 2: act_config.py

**Files:**
- Create: `act/act_config.py`

**Interfaces:**
- Produces: `CFG`（ActConfig 实例），字段：`source_hdf5, camera_names, qpos_source, task_name, chunk_size, kl_weight, lr, lr_backbone, weight_decay, hidden_dim, dim_feedforward, enc_layers, dec_layers, nheads, dropout, backbone, position_embedding, pre_norm, batch_size_train, batch_size_val, num_epochs, seed, dt, train_ratio, num_workers, act_root, checkpoint_dir, log_dir, stats_path`；以及 `make_args_override(cfg) -> dict`（含 `action_dim: 7, qpos_dim: 9`）。

- [ ] **Step 1: 写入以下完整内容**

```python
"""ACT 训练配置 —— 单一事实来源。

所有路径、超参、相机名单都在这里改。对应 spec 第 6/8/12 节:
docs/superpowers/specs/2026-09-29-act-dataset-design.md
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ActConfig:
    # --- 数据 ---
    source_hdf5: str = "<REPO>/task0_nextobs.hdf5"
    camera_names: tuple = ("agentview", "eye_in_hand")   # 文件键为 obs/<cam>_rgb
    qpos_source: str = "robot_states"   # "robot_states"(ee 空间 9 维) | "joint_gripper"(关节 9 维)
    task_name: str = "libero_task0"
    # --- ACT 超参（除 chunk/DT 外均为官方默认） ---
    chunk_size: int = 50          # = num_queries；ALOHA 100@50Hz ≈ 50@20Hz 的 2.5 秒
    kl_weight: int = 10
    lr: float = 1e-5
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    hidden_dim: int = 512
    dim_feedforward: int = 3200
    enc_layers: int = 4
    dec_layers: int = 7
    nheads: int = 8
    dropout: float = 0.1
    backbone: str = "resnet18"
    position_embedding: str = "sine"
    pre_norm: bool = False
    # --- 训练 ---
    batch_size_train: int = 8
    batch_size_val: int = 8
    num_epochs: int = 500
    seed: int = 42
    dt: float = 0.05              # LIBERO 20Hz
    train_ratio: float = 0.8      # 官方 utils.py load_data 实测值 → 40 train / 10 val
    num_workers: int = 1
    # --- 输出 ---
    act_root: Path = Path(__file__).resolve().parent

    @property
    def checkpoint_dir(self) -> Path:
        return self.act_root / "checkpoints"

    @property
    def log_dir(self) -> Path:
        return self.act_root / "logs"

    @property
    def stats_path(self) -> Path:
        return self.act_root / "stats.pkl"


CFG = ActConfig()


def make_args_override(cfg: ActConfig) -> dict:
    """ACTPolicy 构建参数：detr/main.py 对 args_override 逐键 setattr。
    action_dim/qpos_dim 由 vendored detr_vae 补丁消费（ALOHA 写死 14，LIBERO 为 7/9）。"""
    return {
        "lr": cfg.lr, "lr_backbone": cfg.lr_backbone,
        "weight_decay": cfg.weight_decay,
        "enc_layers": cfg.enc_layers, "dec_layers": cfg.dec_layers,
        "dropout": cfg.dropout, "nheads": cfg.nheads,
        "num_queries": cfg.chunk_size, "kl_weight": cfg.kl_weight,
        "hidden_dim": cfg.hidden_dim, "dim_feedforward": cfg.dim_feedforward,
        "backbone": cfg.backbone, "position_embedding": cfg.position_embedding,
        "pre_norm": cfg.pre_norm, "masks": False, "dilation": False,
        "camera_names": list(cfg.camera_names),
        "action_dim": 7, "qpos_dim": 9,
    }
```

- [ ] **Step 2: 自检**

```bash
cd <REPO>/act && <REPO>/venv/bin/python -c "
from act_config import CFG, make_args_override
print(CFG.chunk_size, CFG.camera_names, CFG.train_ratio)
print(make_args_override(CFG)['action_dim'], make_args_override(CFG)['qpos_dim'])"
```
Expected: `50 ('agentview', 'eye_in_hand') 0.8` / `7 9`。

---

### Task 3: act_dataset.py（TDD）

**Files:**
- Create: `act/act_dataset.py`、`act/tests/test_dataset.py`

**Interfaces:**
- Consumes: `CFG`、`make_args_override`（Task 2）。
- Produces:
  - `load_episode_lengths(source_file) -> dict[int,int]`
  - `read_qpos(demo_group, qpos_source) -> np.ndarray (T,9)`
  - `split_episodes(num_episodes, seed, train_ratio) -> (train_ids, val_ids)`（np.ndarray）
  - `get_norm_stats(source_file, episode_ids, qpos_source) -> dict[str, np.ndarray]`（键 `qpos_mean/qpos_std/action_mean/action_std`，一维）
  - `load_or_compute_stats(source_file, episode_ids, qpos_source, cache_path) -> dict`（指纹不匹配自动重算）
  - `class LiberoEpisodicDataset(episode_ids, source_file, camera_names, norm_stats, qpos_source, chunk_size)`：`__len__`、`sample_at(index, start_ts) -> (img (k,C,H,W) float32∈[0,1], qpos (9,) float32, action (chunk,7) float32, is_pad (chunk,) bool)`、`__getitem__`（随机起点）
  - `build_loaders(source_file, cfg, stats=None) -> (train_loader, val_loader, stats, {"train_ids": [...], "val_ids": [...]})`
  - `self_check()`（V1 自检 main）

- [ ] **Step 1: 先写测试（此时 act_dataset.py 不存在，运行必须失败）**

```python
"""act_dataset 测试（TDD；合成 mini LIBERO hdf5，不碰真实数据）。
运行: <REPO>/venv/bin/python tests/test_dataset.py
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
```

- [ ] **Step 2: 运行确认失败**

```bash
cd <REPO>/act && <REPO>/venv/bin/python tests/test_dataset.py
```
Expected: FAIL，`ModuleNotFoundError: No module named 'act_dataset'`。

- [ ] **Step 3: 实现 act_dataset.py（完整内容如下）**

```python
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
    import sys
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
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd <REPO>/act && <REPO>/venv/bin/python tests/test_dataset.py
```
Expected: `TEST-DATASET PASS`。

---

### Task 4: train_act.py + V2 smoke

**Files:**
- Create: `act/train_act.py`、`act/tests/test_policy_build.py`、`act/logs/train_smoke.csv`（运行产物）、`act/checkpoints/policy_latest_smoke.pth`（运行产物）

**Interfaces:**
- Consumes: `ACTPolicy`（vendored）、`CFG/make_args_override`（Task 2）、`build_loaders`（Task 3）。
- Produces: CLI——`--smoke`（2 epoch，V2）、`--epochs N`、`--verify-ckpt PATH`（V4）；checkpoint 格式 `{"model": policy.state_dict(), "norm_stats": stats, "config": args_override, "epoch": int}`。

- [ ] **Step 1: 先写策略构建测试（TDD，首跑会触发 resnet18 权重下载 ~45MB，已缓存则秒过）**

```python
"""V0：ACTPolicy 在锁定 venv + torch 2.2.2 下构建/前向/反向。
运行: <REPO>/venv/bin/python tests/test_policy_build.py
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
```

- [ ] **Step 2: 运行策略构建测试（预期直接 PASS——探针已在同路径验证过；本测试不依赖 train_act.py）**

```bash
cd <REPO>/act && <REPO>/venv/bin/python tests/test_policy_build.py
```
Expected: `TEST-POLICY-BUILD PASS`。若失败，按报错修 vendored 补丁并更新 PROVENANCE，不得继续。

- [ ] **Step 3: 实现 train_act.py（完整内容如下）**

```python
"""ACT 训练入口（改编自官方 act/train.py：去 wandb/matplotlib，CSV 日志）。

用法（libero venv）：
  cd <REPO>/act
  ../venv/bin/python train_act.py --smoke                      # V2：2 epoch 冒烟
  ../venv/bin/python train_act.py --epochs 500                 # 正式训练
  ../venv/bin/python train_act.py --verify-ckpt checkpoints/policy_latest_smoke.pth   # V4
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
from act_config import CFG              # noqa: E402
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
                loss_dict = forward_pass(policy, batch)
                optimizer.zero_grad()
                loss_dict["loss"].backward()
                optimizer.step()
                writer.writerow([epoch, step, "train",
                                 f"{loss_dict['l1'].item():.5f}",
                                 f"{loss_dict['kl'].item():.4f}",
                                 f"{loss_dict['loss'].item():.5f}"])
                fcsv.flush()
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
```

- [ ] **Step 4: V2 冒烟训练**

```bash
cd <REPO>/act && ../venv/bin/python train_act.py --smoke
```
Expected: 两个 epoch 打印 val loss（有限值），`VRAM peak: < 7.5 GiB`，产出 `checkpoints/policy_latest_smoke.pth` 与 `logs/train_smoke.csv`（有 train 与 val 行）。

---

### Task 5: 全量验收（V1 / V3 / V4）

**Files:**
- Read-only 复查 + 运行产物。

- [ ] **Step 1: V1 数据自检（真实文件）**

```bash
cd <REPO>/act && ../venv/bin/python act_dataset.py
```
Expected: `V1 SELF-CHECK PASS`，打印 episodes=50、train/val=40/10、batch 形状。

- [ ] **Step 2: V4 checkpoint 回推**

```bash
cd <REPO>/act && ../venv/bin/python train_act.py --verify-ckpt checkpoints/policy_latest_smoke.pth
```
Expected: `V4 VERIFY-CKPT PASS: a_hat (8, 50, 7) finite`。

- [ ] **Step 3: V3 主流程无恙**

```bash
<REPO>/venv/bin/pip freeze > /tmp/post_freeze.txt
diff <REPO>/act/pre_freeze.txt /tmp/post_freeze.txt && echo "FREEZE OK"
find <REPO> -newer <REPO>/act/pre_freeze.txt \
     -not -path "*/act/*" -not -path "*/docs/*" -not -path "*/.zcode/*" -not -path "*/__pycache__/*"
```
Expected: `FREEZE OK`；find 输出为空（act/ 与 docs/ 之外零改动）。

- [ ] **Step 4: 收尾报告**

向用户汇报：四项验证结果、VRAM/耗时、checkpoint 与日志位置、PROVENANCE 路径，以及"下一步=正式训练/rollout 评测"的建议。
