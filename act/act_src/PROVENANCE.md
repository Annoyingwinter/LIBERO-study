# PROVENANCE — vendored ACT 源码

- **上游**: https://github.com/tonyzhaozh/act ，commit `742c753c0d4a5d87076c8f69e5628c79a8cc5488`（main HEAD，2026-09-29 取用）
- **取用方式**: github.com 直连不稳定（clone/codeload 超时），经 jsDelivr CDN 逐文件下载（`cdn.jsdelivr.net/gh/tonyzhaozh/act@main/<path>`），`py_compile` 全部通过
- **补丁重放**: `../apply_patches.py`（幂等；本目录已含补丁时全部 skip）
- **vendored 清单**: `policy.py`、`utils.py`、`detr/{main.py, models/{__init__,backbone,detr_vae,position_encoding,transformer}.py, util/{__init__,box_ops,misc}.py}`
- **未 vendored**: `train.py`（参考件，本次 CDN 下载未成功；官方 train.py 依赖 constants.py/simulate.py，本就不可 import）、`constants.py`、`simulate.py`、`imitate_episodes.py` 等（仿真/ALOHA 专用，不需要）

## 对上游代码的改动（全部由 apply_patches.py 管理）

### A. 剥离 IPython（policy.py / utils.py / detr/main.py）
删除 `import IPython` 与 `e = IPython.embed` 两行 ×3 文件。
原因：venv 锁定依赖、无 IPython；该 import 仅服务上游作者的调试断点。

### B. detr/main.py：`parse_args()` → `parse_args([占位值])`（两处 build 函数）
原行为：`parser.parse_args()` 在**每次构建模型时**重新解析 sys.argv——被外部脚本 import 调用时会把调用方的命令行参数（如 `--smoke`）误当 detr 的参数，直接崩溃。
新行为：传入占位 argv；`ckpt_dir/policy_class/task_name/seed/num_epochs` 五个 required 参数给占位值，真实值全部由 `args_override` 字典覆盖（main.py 有现成的 `for k, v in args_override.items(): setattr(args, k, v)` 循环）。
注：实测 `parser.set_defaults()` **不能**满足 `required=True`（argparse 仍报 missing），所以必须给占位 argv。

### C. detr/models/detr_vae.py：写死的 14 参数化为 action_dim / qpos_dim（7 处）
原行为：ALOHA 双臂写死 14 维——`build()` 里 `state_dim = 14 # TODO hardcode`，`__init__` 里 `input_proj_robot_state / encoder_action_proj / encoder_joint_proj` 三处 `nn.Linear(14, hidden_dim)`。
新行为：`build(args)` 读取 `args.action_dim`（动作 7 维：LIBERO ee 位姿增量 3+3+夹爪）与 `args.qpos_dim`（本体 9 维：robot_states = 夹爪 2 + ee 位置 3 + ee 四元数 xyzw 4），`DETRVAE.__init__` 签名相应参数化；两值由 `../act_config.py::make_args_override` 提供。
`build_cnnmlp` / `CNNMLP` 保持原样（未使用；如需使用需同样适配）。

## 布局说明（为什么需要两条 sys.path）

上游 `detr/models/backbone.py` 与 `position_encoding.py` 用**绝对导入** `from util.misc import ...`，而 `util` 并非上游仓库的顶层包——原版布局同样会报 `No module named 'util'`（实测）。本项目在入口（`train_act.py`、`tests/*`）把 `act_src/` 与 `act_src/detr/` 同时加入 sys.path：前者让 `import policy`、`from detr.main import ...` 可解析，后者让 `util.misc` 可解析。实测通过。

## 验证记录（2026-09-29 探针，torch 2.2.2+cu121 / RTX 4060 8GB）

- import 冒烟：`import utils, policy` OK
- ACTPolicy 构建：83,885,192 params（action_dim=7, qpos_dim=9）
- 训练前向+反向：loss 89.07（l1 0.83 / kl 8.82），有限
- 推理：`a_hat (2, 50, 7)` 有限
- VRAM 峰值：0.77 GiB @ batch 2
- resnet18 ImageNet 权重下载成功（~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth）
