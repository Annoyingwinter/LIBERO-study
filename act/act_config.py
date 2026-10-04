"""ACT 训练配置 —— 单一事实来源。

所有路径、超参、相机名单都在这里改。对应 spec 第 6/8/12 节:
docs/design/2026-09-29-act-dataset-design.md

路径不做绝对路径硬编码：source_hdf5 可用环境变量 ACT_DATA 覆盖，默认相对仓库
根目录解析（脚本不依赖 CWD）。
"""
import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


@dataclass
class ActConfig:
    # --- 数据 ---
    source_hdf5: str = os.environ.get(
        "ACT_DATA", str(REPO / "data" / "libero_spatial" / "task0_nextobs.hdf5")
    )
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
