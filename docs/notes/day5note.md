# Day 5 笔记（BC-RNN 训练管线打通）

## 一、今晚成果

```text
✅ BC-RNN 训练真实跑通（robomimic 路径，绕开 fork 的 10 任务要求）
   5 个 epoch，每个约 30 秒，GPU 显存约 1.8GB
   证据：logs/log.txt，末尾 "finished run successfully!"
```

## 二、loss 数字（对照预测）

```text
我预测起始 loss 量级：1000
实际：Log_Likelihood 起始 -1.30（打印的 Loss = +1.30）

5 个 epoch 序列：
  epoch 1  Loss =  1.299
  epoch 2  Loss = -1.568
  epoch 3  Loss = -2.364
  epoch 4  Loss = -3.394
  epoch 5  Loss = -4.852

注意：这里的 Loss = -Log_Likelihood（负的对数似然）。
Log_Likelihood 从 -1.30 涨到 +4.85 = 模型越来越能解释师傅的动作。
打印的 Loss 数字往下走 = 变好（和"loss 下降"说法一致）。
我猜的 1000 量级错得离谱，实际是"个位数"。
```

## 三、今晚连闯的三关（都是真 bug，都修了）

```text
1. mujoco_py 缺失：robomimic 训练时想建在线验证环境
   → 关掉 validate 和 rollout，纯离线训练不需要环境
2. 输出目录已存在：robomimic 交互式问"覆盖?"
   → 后台没有 stdin → 删掉旧目录重跑
3. LIBERO 数据没有 next_obs：robomimic 写死要下一步观测
   → 复制 hdf5，给 50 个 demo 各补 next_obs 组
     （next_obs[t] = obs[t+1]，末步复制自己）
   新文件：/home/flexing/libero/task0_nextobs.hdf5
```

## 四、网络问题：本机代理救场

```text
HF 直连全超时 → 本机 xray 代理 127.0.0.1:10808 可用（0.7s 返回 200）
curl -x http://127.0.0.1:10808 -L -C - --retry ... 走代理续传
9 个缺失 demo 正在后台补，已完成 5 个
```

## 五、两个重要事实（明晚用）

```text
1. 5 个 epoch 没有保存权重！
   robomimic 配置 save.every_n_epochs=50，
   今晚只是管线冒烟，权重要 num_epochs=50 跑完才有。

2. 观察项：Policy_Grad_Norms 从 2.3 万涨到 4.6 万（梯度范数在变大）
   50 epoch 真训时盯这个数，若继续暴涨可能训练不稳（进阶选修）。
```

## 六、明晚产出（写死）

```text
1. 把 bc_rnn_task0.json 的 num_epochs 改成 50，真训约 25 分钟
2. 检查 robomimic_out 里出现模型权重文件
3. 写加载脚本：从 obs 建输入 → policy 输出 action → 喂 env
   （把 day_replay_demo.py 的 for action in actions
     换成 action = policy(obs)）
4. 用 day_eval_probe 的 harness 跑 20 回合 → 真实成功率数字
   （第 3 周收官 + 第 4 周正式评测开始）
```

## 七、今晚的关键命令存档

```bash
# 训练（离线，无验证）
cd /home/flexing/libero
MUJOCO_GL=egl /home/flexing/libero/venv/bin/python \
  /home/flexing/libero/venv/lib/python3.10/site-packages/robomimic/scripts/train.py \
  --config /home/flexing/libero/bc_rnn_task0.json

# 配置：/home/flexing/libero/bc_rnn_task0.json
# 数据：/home/flexing/libero/task0_nextobs.hdf5（带 next_obs 的副本）
# 输出：/home/flexing/libero/robomimic_out/bc_rnn_libero_task0/
```
