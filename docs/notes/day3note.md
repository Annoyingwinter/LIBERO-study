# Day 3 笔记（第 1、2 周收官 + done 链条）

## 一、已完成战绩

```text
第 1 周：评测 harness ✓
    day_eval_probe.py：20 回合占位策略
    成功率 = 0 / 20（骨架验证，非策略表现）
    产出 CSV：day_eval_harness.csv（表头 + 20 行）

第 2 周：demo 回放 ✓
    day_replay_demo.py：回放 demo_0 的 98 步人类示范
    第 80 步 done=True，成功率 = 1 / 1
    "数据集 → 动作 → 环境 → 成功判定 → 成功率"全线打通
```

## 二、harness 与 replay 的区别

```text
harness（day_eval_probe.py）= 评测考试
    起点：benchmark 官方 init_state（50 张考场照片的第 0 张）
    终点：成功率 = 成功回合数 / 总回合数（0/20）
    用途：公平地给每个策略打分

replay（day_replay_demo.py）= 重演录像
    起点：demo 自己的 model_file + states[0]（录像第一帧）
    终点：成功率 = 1/1（验证成功判定工作正常）
    用途：确认"数据集 → 环境"回放通路
```

## 三、done 的三股链条（全部核实过 file:line）

### 1. env.step 返回的 done（成功判定，我们能拿到）

```text
你的代码：obs, reward, done, info = env.step(action)
  ↓
ControlEnv.step        env_wrapper.py:87-88          只是转发
  ↓
BDDLBaseDomain.step    bddl_base_domain.py:800-809
    obs, reward, done, info = super().step(action)    # 先推进仿真
    done = self._check_success()                      # 覆写成成功判定
  ↓
Libero_Tabletop_Manipulation._check_success
    problems/libero_tabletop_manipulation.py:135-145
    goal_state = self.parsed_problem["goal_state"]    # 从 BDDL 读目标
    for state in goal_state:
        result = self._eval_predicate(state) and result
    # 例：on(bowl, plate) 为真 → 成功
```

下游：`if done: break`、`success = done`、`success_total += 1`

### 2. 内部 self.done（回合寿命，拿不到，只会变成报错）

```text
base._post_action   robosuite/environments/base.py:428-429
    self.done = (self.timestep >= self.horizon) and not self.ignore_done
    # 步数到 1000 才 True，和任务成功无关
  ↓
base.step 开头      base.py:375-376
    if self.done:
        raise ValueError("executing action in terminated episode")
    # 第 5 循环崩溃那次就是它
```

### 3. 数据集 dones（录制时的结束标记）

```text
scripts/create_dataset.py:228-230
    dones = np.zeros(len(actions))
    dones[-1] = 1     # 最后一步标 1
```

### 对照表

| 名字 | 谁算的 | 什么时候 True | 能拿到吗 |
|---|---|---|---|
| env 返回 done | `_check_success()`（BDDL 谓词） | 碗真在盘子上（第 80 步） | 能，就是 success |
| 内部 self.done | `timestep >= horizon`（1000） | 步数耗尽 | 拿不到，变成报错 |
| 数据集 dones | 录制脚本拍的 | 录制结束那一步（第 98 步） | 只在 hdf5 里 |

## 四、第 80 步 vs 第 98 步

```text
录制者第 80 步就把碗放好了，手还多动了 18 步才停：
数据集 dones 标在"手停"（98）
环境 done 亮在"碗到位"（80）
一个记"人什么时候停"，一个算"任务什么时候成"
```

## 五、回放踩的坑（已修，留档）

```text
demo 的 XML 写死了原作者机器的绝对路径，共 93 处：
  /Users/yifengz/workspace/robosuite-master/...  64 处 → 官方 postprocess 修
  /Users/yifengz/workspace/libero-dev/chiliocosm/... 29 处 → 手动 replace 修
修复在 day_replay_demo.py 第 42-52 行，有注释
```

## 六、明晚开场题（欠账）

```text
1. harness 和 replay 的起点、终点各是什么？（已写在第二节，明晚用自己的话说）
2. 三股 done 各自的判定者是谁？（第三节表格，明晚默写一遍）
```

## 七、下一步（第 3 周）

```text
接真策略：把 replay 里的 for action in actions
换成 action = policy(obs)
选型：下载小 checkpoint 或训练 BC-RNN 小切片
```
