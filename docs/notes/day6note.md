# Day 6 笔记（第 3 周收官 + 第 4 周开跑）

## 一、第一个"模型成功率"数字

```text
评测：libero_spatial task 0，benchmark init_state 0，20 回合
结果：成功率 = 1 / 20（第 14 回合，90 步成功）
复测 40 回合：2 / 40 = 5%（ep08 164步 / ep25 161步）
结论：5% 可重复，不是单点运气
```

## 二、失败分析（我自己看回放得出的第一句）

```text
"没靠近碗，走过了再夹，夹不着"
→ 定位不准（overshoot）+ 夹爪时机不对
```

## 三、三个数字的历史意义（项目地基）

```text
第 1 周 占位策略   0/20  → 验证"数数"的 harness
第 2 周 师傅录像   1/1   → 验证成功判定
第 3 周 我的模型   5%    → 验证"模仿真的学到了一点"
```

## 四、GPU 蝴蝶效应（重要发现）

```text
20 个回合起点完全相同（同 init_state、同 seed、评测模式），
结果却 19 败 1 成。
原因：GPU 上 cuDNN LSTM 不保证逐位确定性，
微小抖动在某一步放大成完全不同的轨迹。
教训：评测必须多回合取成功率，"成功一次"不说明"会了"。
```

## 五、第 4 周改进（今晚启动）

```text
训练配方改动（bc_rnn_task0.json）：
  1. 数据增强：CropRandomizer（128×128 随机裁 76×76）
     = 模拟"碗的位置左右平移"的扰动，逼模型学会定位
  2. 轮数：50 → 200（loss 到 21.9 还在涨，没训够）
  3. 输出名：bc_rnn_aug_200ep（避免目录冲突）
预期：200 轮 × 30 秒 ≈ 100 分钟，今晚后台跑
```

## 六、明晚产出

```text
1. 检查 200 轮训练完成 + loss 曲线
2. 改 day_policy_eval.py 的 CKPT 指向新权重
3. 重跑 40 回合 → 新的成功率数字
4. 和 5% 对比 → 数据增强值多少钱
```

## 七、命令存档

```bash
# 训练（增强 + 200 轮）
cd /home/flexing/libero
MUJOCO_GL=egl /home/flexing/libero/venv/bin/python \
  /home/flexing/libero/venv/lib/python3.10/site-packages/robomimic/scripts/train.py \
  --config /home/flexing/libero/bc_rnn_task0.json

# 评测（带录像，N 回合）
cd /home/flexing/libero/LIBERO
MUJOCO_GL=egl /home/flexing/libero/venv/bin/python \
  /home/flexing/libero/day_policy_eval.py 40
```
