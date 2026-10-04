# Day 4 笔记（第 3 周开局：训练管线选型）

## 一、今晚目标与结果

```text
目标：跑起 BC-RNN 训练入口
结果：未跑起，但选型完成，明晚可直接接上
```

## 二、卡点（有证据，不是猜）

```text
HF（huggingface.co）今晚完全连不上：
  9 个 demo 文件 × 每文件 15 次重试，全部 connect timeout
  （curl exit 28，连接超时）
fork 官方训练入口要求一次加载全部 10 个任务的数据，
所以缺 9 个文件 = 训练无法启动。
```

## 三、已验证的备用路径（明晚主线）

绕开 fork 的"10 任务一起训"，用 robomimic 直接单任务训练：

```text
robomimic 0.2.0 已装，入口在：
  venv/lib/python3.10/site-packages/robomimic/scripts/train.py
  用法：--config <json> --dataset <hdf5>

BC 模板已生成：
  venv/lib/python3.10/site-packages/robomimic/exps/templates/bc.json
  其中 rnn.enabled 默认 false（第 129 行），改成 true 就是 BC-RNN

数据：我们已有 task 0 的 hdf5（485MB，50 demo），单任务训练足够
```

## 四、明晚产出（写死）

```text
1. 写最小 bc_rnn 配置（基于 bc.json 改）：
   - dataset = task 0 hdf5 绝对路径
   - rnn.enabled = true，horizon/seq_length = 10
   - obs：rgb = [agentview_rgb, eye_in_hand_rgb]
          low_dim = [gripper_states, joint_states]
   - num_epochs = 5（先冒烟），batch 调小防 8GB 爆显存
2. 跑通 5 个 epoch，日志出现 loss 数字
3. 对照昨晚三个预测：
   # demonstrations = 500（10×50）
   loss 起始数量级（我猜过 1000）
   loss 往下走 = 模仿误差变小
```

## 五、已验证但今晚没用的知识（别浪费）

```text
fork 训练入口（等 HF 恢复后仍可用）：
  cd /home/flexing/libero/LIBERO
  MUJOCO_GL=egl python libero/lifelong/main.py \
    policy=bc_rnn_policy train.n_epochs=50 seed=10000
  入口硬性要求 libero_spatial 全部 10 个 demo 文件
  （libero/lifelong/main.py 的 for i in range(n_manip_tasks) 循环）

fork 数据配置（configs/data/default.yaml）：
  img 128×128，obs = agentview_rgb + eye_in_hand_rgb
                    + gripper_states + joint_states，seq_len 10
  ← 明晚 robomimic 配置照抄这套 keys
```

## 六、网络备忘

```text
HF 今晚超时，GitHub 前几天也抖。
下载一律用 curl -L -C -（断点续传）+ --retry。
能本地搞定的（如任务清单来自 libero_task_map）不依赖远端 API。
9 个缺失文件在 HF 恢复后重试：
  libero_spatial/<task>_demo.hdf5 共 9 个，每个约 485MB
```
