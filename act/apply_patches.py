"""apply_patches.py — 对 vendored ACT 源码重放 3 类补丁（幂等）。

补丁清单（已实测，见 docs/superpowers/plans/2026-09-29-act-dataset.md 探针事实）：
  A. 剥离 IPython（policy.py / utils.py / detr/main.py）—— venv 无 IPython，且仅为上游调试入口；
  B. detr/main.py 两处 parse_args() → parse_args([占位值]) —— 该函数被外部 import 调用，
     不能消费调用方 sys.argv；set_defaults 实测无法满足 required=True；
  C. detr/models/detr_vae.py 把写死的 14 参数化为 action_dim(7)/qpos_dim(9) —— ALOHA 双臂 14 维 → LIBERO 7/9 维。

用法: python apply_patches.py <act_src_dir>
"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1])


def patch(rel, reps):
    p = root / rel
    s = p.read_text()
    s2 = s
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
