# ACT 数据集接入设计（LIBERO task0 → ACT 训练管道）

- 日期：2026-09-29
- 状态：待用户评审（未批准前不动手实施）
- 范围：补齐 ACT 训练的"数据层"，端到端验证到 smoke train。不含正式训练调参、不含 rollout 评测。

## 0. 一句话

把 LIBERO task0 的 hdf5 **原地**接入 vendored ACT，全程零 pip 安装、零修改现有文件，验证到"数据加载正确 + GPU 上 smoke 训练能跑 + 主流程无恙"。

## 1. 背景与已验证事实

前期调查结论（已对齐）：本机无 lerobot、无 ACT 源码；venv 锁依赖不能装新库；选定 vendor ACT 源码 + 自写数据适配层的路线。

本轮设计前实测确认的环境与数据事实：

| 项 | 值 |
|---|---|
| venv | Python 3.10.12，torch 2.2.2+cu121（**CUDA 可用**），torchvision 0.17.2，einops 0.8.2，h5py 3.16.0，numpy 1.23.5 |
| GPU | RTX 4060 Laptop 8GB |
| torchvision 兼容性 | 实测 `resnet18(pretrained=False)` 可调用（旧式 API 仍被接受）→ ACT 原版 backbone 代码无需修改；首次训练需联网下载 ImageNet 权重 ~45MB |
| 网络/磁盘 | `git ls-remote` GitHub 可达；/home 剩余 100G |
| git | `<REPO>` **不是** git 仓库 → 本 spec 只写文件不 commit |
| 数据源一致性 | `task0_nextobs.hdf5` = libero_spatial task0（黑碗 between plate/ramekin）50 条 demo 的完整拷贝，demo_0 的 actions / agentview_rgb / joint_states 与官方文件**逐位一致** |
| 字段解码 | `robot_states (T,9)` = [夹爪指位(2), ee 位置(3), ee 四元数 xyzw(4)]（前 5 维与 gripper_states / ee_pos 逐位吻合，后 4 维单位范数）；**不等于** joint+gripper 拼接 |
| episode 长度 | 变长（demo_0=98 步，stove 任务 demo_0=155 步），动作 7 维 = ee 位姿增量(3 平移+3 旋转) + 夹爪 |

## 2. 目标与成功标准（Definition of Done）

1. **V1 数据自检**：`python act_dataset.py` 通过——50 集全部加载，逐字段形状/取值范围断言，DataLoader 能出一个合法 batch。
2. **V2 smoke 训练**：`python train_act.py --smoke` 通过——GPU 上 2 个 epoch 真实前向+反向，loss 为有限值且被记录，显存峰值 < 7.5GB，产出可 `torch.load` 的 checkpoint。
3. **V3 主流程无恙**：实施前后 `pip freeze` 完全一致；`libero/` 下除新增目录外无任何文件被改动（`find -newer` 验证）。
4. **V4 推理回路**：smoke checkpoint 加载回 ACTPolicy，在 val batch 上前向一次，输出 `a_hat` 形状 (B, chunk, 7)。
5. 全程**零 pip install**、零依赖变更。

## 3. 红线约束（不损坏主流程）

- **只增不改**：所有新文件仅出现在 `<REPO>/act/` 和 `<REPO>/docs/superpowers/specs/` 下。
- **数据只读**：所有 hdf5 以 `h5py.File(..., 'r')` 打开，任何环节不写数据文件。
- **venv 不动**：不执行任何 pip 命令。
- **回滚 = 删除新目录**：`rm -rf <REPO>/act/` 即完全还原。
- 代码风格跟随现有课程脚本：中文 docstring、常量置顶、只读纪律写在文件头。

## 4. 方案选型

| 方案 | 说明 | 结论 |
|---|---|---|
| **A. vendor + in-place adapter**（选定） | 拷贝 ACT 源码进项目，自写 Dataset 直接读现有 hdf5，零拷贝零安装 | ✅ 依赖全齐、代码全在自己仓库可读可改，符合学习目标 |
| B. 物理转换为 ACT 官方格式 | 生成 `episode_i.hdf5`，官方 loader 原样可用 | 备选：复制 ~600M 图像、形成两份真相源；默认不做（第 12 节开放决策） |
| C. 独立 venv + lerobot | 不动主 venv，但需重下 torch、数据转 parquet+mp4 | 本轮否决，留给以后生态路线 |

A 与 B/C 的取舍已在前两轮对话中与用户对齐。

## 5. 架构与文件布局

```
<REPO>/act/
├── act_src/            # vendor 自 tonyzhaozh/act（main HEAD, 742c753）
│   ├── policy.py       # ACTPolicy / CNNMLPPolicy
│   ├── utils.py        # set_seed / normalize / get_norm_stats 等（不用其 load_data）
│   ├── detr/           # 模型定义（main.py, models/, util/）—— 原样
│   ├── train.py        # 仅作参考，不进 import 路径
│   └── PROVENANCE.md   # 来源 commit + 对 vendored 代码的逐条改动记录
├── act_config.py       # 所有路径、超参、相机名单一事实来源
├── act_dataset.py      # 数据层：EpisodicDataset(LIBERO hdf5) + build_loaders() + 自检 main
├── train_act.py        # 训练循环（改编自 act/train.py）：argparse、循环、CSV 日志、checkpoint
├── pre_freeze.txt      # 实施前的 pip freeze 快照（V3 用）
├── stats.pkl           # 归一化统计缓存
├── checkpoints/
└── logs/
```

单元边界：`act_src/` 不知道 LIBERO 的存在（只认 ACT 格式张量）；`act_dataset.py` 只读数据、不做训练；`train_act.py` 只管训练循环。数据流：hdf5 → `EpisodicDataset.__getitem__` → (image dict, qpos, action, is_pad) → normalize(stats) → ACTPolicy。

## 6. 数据映射（act_dataset.py 规格）

| ACT 张量 | 源字段（data/demo_i/…） | 处理 | 形状 |
|---|---|---|---|
| qpos | `robot_states` | 原样（ee 空间本体：夹爪+位置+四元数，与 ee-delta 动作同空间） | (T, 9) |
| images["agentview"] | `obs/agentview_rgb` | uint8 原样取出，float/255 后交 policy（ImageNet 归一化在 ACT policy 内已有） | (T, 128, 128, 3) |
| images["eye_in_hand"] | `obs/eye_in_hand_rgb` | 同上 | (T, 128, 128, 3) |
| action | `actions` | 原样（ee 位姿增量 3+3 + 夹爪；ACT 只做向量回归，语义无碍） | (T, 7) |
| is_pad | 变长 episode | 随机起点 ts，chunk 越过集末的部分零填充并置 1 | (chunk,) |

- **qpos 备选**：`concat(obs/joint_states (7), obs/gripper_states (2))`，`act_config.py` 一行切换；两种表征都已验证存在且 rollout 时同样可得。
- **qvel**：ACT 的 ACTPolicy 不使用它，不提供；若 vendored Dataset 接口需要占位则写 0。
- **camera_names** = `["agentview", "eye_in_hand"]`（相机名对 ACT 只是 dict 键）。
- 实施时以 `grep -rn "IPython\|matplotlib\|wandb act_src/"` 兜底：venv 缺这三个包，vendored 代码中相应 import 一律删除（改动全部记入 PROVENANCE.md）。

## 7. 采样、划分与归一化

- **划分**：跟随官方 `load_data` 的随机划分逻辑，**实测官方源码 `train_ratio = 0.8`** → 50 集 = 40 train / 10 val，`np.random.RandomState(seed=42)` 固化，结果可复现。
- **采样**：沿用官方 EpisodicDataset 逻辑——每条样本随机取起点 ts，图像取单帧、action 取 `ts : ts+chunk`，越过集末零填充并记 is_pad。
- **归一化**：官方 `get_norm_stats` 逻辑——仅对 train 集全量计算 qpos/action 的 mean/std，缓存到 `act/stats.pkl`；val 集不参与统计。

## 8. 关键默认超参（act_config.py）

| 参数 | 默认值 | 说明 |
|---|---|---|
| DT | 0.05 | LIBERO 20Hz（ALOHA 是 50Hz） |
| chunk_size | 50 | 2.5s @ 20Hz；显存吃紧则降 25 |
| kl_weight | 10 | ACT 官方默认 |
| batch_size | 8 | 8GB 显存约束下保守起步 |
| lr / lr_backbone | 1e-5 / 1e-5 | ACT 官方默认 |
| hidden_dim / dim_feedforward | 512 / 3200 | ACT 官方默认 |
| enc_layers / dec_layers / nheads | 4 / 7 / 8 | ACT 官方默认 |
| num_queries | = chunk_size | ACT 官方约定 |
| num_epochs | smoke=2 | 正式训练轮数不在本 spec 范围 |
| seed | 42 | 划分/采样/torch 全部统一 |
| 日志 | CSV | 不引入 wandb、matplotlib |

## 9. 测试与验证方案

- **V1** `python act_dataset.py`：打印 50 集长度分布；逐字段断言形状与 dtype（qpos (T,9) float、两相机 (T,128,128,3) uint8 ∈ [0,255]、action (T,7)）；出一个 train batch 和一个 val batch 验证形状。
- **V2** `python train_act.py --smoke`：断言 CUDA 可用；2 epoch；每 step 记录 loss 到 CSV；结束时断言 `torch.cuda.max_memory_allocated() < 7.5 GiB`、`checkpoints/smoke.pth` 存在。
- **V3** 主流程守护：实施开始前 `pip freeze > act/pre_freeze.txt`（该文件的时间戳作为参照点）；实施完成后 `find <REPO> -newer act/pre_freeze.txt -not -path "*/act/*" -not -path "*/docs/*"` 应为空——即 act/ 与 docs/ 之外零改动。
- **V4** `python train_act.py --verify-ckpt checkpoints/smoke.pth`：加载 checkpoint，val batch 前向，断言 `a_hat.shape == (B, chunk, 7)` 且有限。

## 10. 风险与对策

| 风险 | 概率 | 对策 |
|---|---|---|
| torch 2.2 与 2023 年 ACT 代码的 API 差异 | 低 | V2 smoke 即暴露；vendored 代码可就地修，改动记入 PROVENANCE.md |
| resnet18 预训练权重下载失败 | 低 | 网络已实测可达；失败则 `weights=None` 从头训，管道不受影响，仅精度受损 |
| 显存超 7.5GB | 中 | 依次降 batch 8→4、chunk 50→25；AMP 留作后续 |
| 变长 episode 边界 bug | 低 | 官方 pad + is_pad 逻辑，V1 断言覆盖边界样本 |
| vendored 代码隐藏 import（IPython 等） | 低 | 第 6 节 grep 兜底；IPython 已确认不在 venv |

## 11. 明确不做（YAGNI）

LeRobotDataset（parquet+mp4）转换、LIBERO env rollout 闭环评测、wandb/matplotlib、多任务/多 suite 训练、timm/DINOv2 backbone、分布式训练、正式训练调参。

## 12. 开放决策（默认已选，评审时可改）

1. **数据源 = `task0_nextobs.hdf5`**：已验证与官方文件逐位一致；读它而不读 LIBERO repo 内文件，避免触碰 repo 目录。
2. **qpos 表征 = `robot_states`（ee 空间 9 维）**：与动作同空间、LIBERO 原生表征；备选 joint+gripper 9 维，一行切换。
3. **chunk_size = 50**。
4. **方案 B（物理转换）不做**。

## 13. 下一步

spec 批准后：writing-plans skill 产出实施计划 → 按计划实施（act_dataset 先写断言测试再实现，TDD）。
