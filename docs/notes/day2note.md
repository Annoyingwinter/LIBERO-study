## 数据集真实结构（对照预测）
- 我预测的键：action / policy / observation
- 实际顶层：data
- 实际 demo 键：actions(98,7) dones(98,) rewards(98,) states(98,92)
  robot_states(98,9) obs(子组)
- obs 实际 7 个键：agentview_rgb(128×128×3) eye_in_hand_rgb(128×128×3)
  ee_pos(3) ee_ori(3,轴角) ee_states(6) gripper_states(2) joint_states(7)
- action 维度：7 ✓（我预测对了）
- 步数：75~168，每条不同 ✓（我预测对了）
- attrs：init_state / model_file / num_samples
- 我错在哪：预测了"policy"键，实际没有；obs 键名和我想的完全不一样

---

## 今晚补充（导师整理）

### 1. dones / rewards 的真实含义

```text
rewards = 全 0（示范数据没有奖励）
dones   = 全 0，只有最后一步是 1（create_dataset.py:228-230）
含义：dones 标记"这条示范在这一步结束"，不是直接标记"成功"
示范数据里人都是成功后停，所以两者重合
评测时 done 来自任务成功判定，是另一回事
```

### 2. robot_states(9) 的正确拆法（更正导师之前的错误）

```text
robot_states(9) = 夹爪 qpos(2) + 末端位置(3) + 末端四元数(4)
源码：bddl_base_domain.py:826-829 get_robot_state_vector
注意：7 个关节在 obs/joint_states(7) 里，不在 robot_states 里
```

### 3. states(92) 是"全集"，已用实验证明

```text
实验：只把 states[0]（92 维）灌回环境
结果：夹爪恢复到 [0.034,-0.034]，关节恢复到文件记录值附近
结论：92 维里包含手指和关节（还有物体的全部自由度）
与 robot_states(9) 的关系：全集 vs 精选，不是互斥
小偏差 0.002：记录时刻与观测时刻差一个子步（进阶选修）
```

### 4. ee_ori(3) 是轴角，env 里是四元数(4)

```text
同一个"姿态"，两个记法：
env 观测 robot0_eef_quat  = 4 维四元数 [x,y,z,w]
数据集 ee_ori             = 3 维轴角（T.quat2axisangle 转换）
ee_states(6) = ee_pos(3) + ee_ori(3)
```

### 5. 格式错的两种后果（接策略时的最大坑）

```text
a. 代码显式检查维度 → 立刻 ValueError / IndexError（好事，当场知道）
b. 代码不检查 → 静默算错，不报错，成功率数字全是假的（最危险）
```

### 6. 三问判卷结果

```text
Q1 "states 里有没有手指"：有（实验证明），但我第一轮的理由错了
Q2 ee_states/robot_states 拆账：第二轮答对
Q3 维度不匹配的后果：没答细，导师给出（见第 5 条）
```

### 7. 今晚文件资产

```text
/home/flexing/libero/day_dataset_inspect.py   只读检查脚本（29 行输出）
demo 文件：LIBERO/libero/datasets/libero_spatial/..._demo.hdf5（485 MB）
```

