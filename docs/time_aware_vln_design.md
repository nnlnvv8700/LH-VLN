# Time-Aware VLN 设计梳理

这份文档用来整理当前 time-aware VLN 改造的主要 idea、已经实现的内容、后续可能的 baseline，以及还需要确认的问题。

## 1. 任务动机

原始 LH-VLN 关注长指令、多阶段任务，但通常还是按照数据里的子任务顺序来完成。我们现在想研究一个更 time-aware 的问题：

```text
给定一条长指令、多个目标、有限时间或有限步数，
agent 应该怎样安排目标顺序，才能在时间耗尽前完成尽可能多、尽可能有价值的目标？
```

也就是说，任务重点从“是否按完整顺序全部完成”扩展为：

- 时间不够时先做什么？
- 哪些目标应该优先完成？
- 剩余时间变化后策略是否会调整？
- 通用 VLN 模型能否通过 prompt 理解时间限制并灵活行动？

## 2. 当前任务定义

当前设定是：

- 输入仍然是一条 LH-VLN 风格长指令。
- 一条指令中包含多个目标。
- 评测时目标可以无序完成。
- 每个 episode 给定一个 step budget。
- agent 在预算内尽量完成更多目标。
- 当前所有目标 value 都先设为 1，即每个目标同等重要。

例子：

```text
Instruction:
Take the box in the bedroom to the dining area table and then retrieve the lamp from there.

Targets:
box, table, lamp

Budget:
60 steps
```

在 time-aware setting 下，agent 不一定必须按照 `box -> table -> lamp` 的原始顺序完成，而是可以根据距离、剩余时间、目标价值等因素决定顺序。

## 3. 时间预算设定

当前预算来自原始 LH-VLN 的 ordered oracle gt steps。也就是原数据中按原始顺序完成任务需要的专家步数。

目前使用四档预算：

```text
0.5  * ordered oracle gt steps
0.75 * ordered oracle gt steps
1.0  * ordered oracle gt steps
1.25 * ordered oracle gt steps
```

这样可以观察不同时间压力下的表现：

- `0.5`：明显时间不足。
- `0.75`：较紧张。
- `1.0`：接近原专家完整任务预算。
- `1.25`：相对宽松。

## 4. 核心评价方式

师兄建议的核心形式是：

```text
横坐标：时间限制 / step budget
纵坐标：成功率 / 任务完成度
```

所以最终实验最好画 budget curve，而不是只给单个成功率。

当前已有指标：

- `Success@Budget`：预算内是否完成所有目标。
- `Completion rate`：预算内完成目标比例。
- `Reward rate`：预算内完成目标价值比例。当前 value 全部为 1，因此基本等价于 completion rate。
- `Avg. time used`：平均实际消耗 step。
- `Failed stops`：错误 stop 次数。
- `Abandoned targets`：不可达或 follower 失败后放弃的目标数。

这些不是直接照搬 LH-VLN 原始 ISR/CSR/CGT，而是更适合 time-aware / anytime evaluation 的指标。原始的 success、oracle success、navigation error、gt step 等底层字段仍然可以保留。

## 5. 当前已实现内容

已经完成：

- 从 `data/episode_task/*.json.gz` 生成 `data/time_aware/episodes.jsonl`。
- 每条样本保留 instruction、scene、robot、split、targets、start state、ordered oracle steps 和 time budgets。
- 加入 `start_position` 和 `start_yaw`。
- target position 优先使用成功轨迹终点，也就是 expert trajectory 到达该目标附近的位置。
- 如果没有轨迹终点，则 fallback 到 Habitat semantic object position。
- 新增 no-render simulator 路径，避免服务器 EGL/OpenGL 渲染问题阻塞实验。
- 实现 nearest-target greedy baseline。
- 支持多 budget ratio sweep。
- 输出每档预算的 JSON 结果和 summary CSV。
- 处理 `GreedyFollowerError`，避免单个不可达目标中断整批实验。

当前数据规模：

```text
train: 608
val: 41
test: 403
total: 1052
```

## 6. 当前 Greedy Baseline

当前 baseline 是 nearest-target greedy。

它不是大模型，也不是 learned policy，而是 oracle-style rule baseline。

流程：

```text
1. agent 知道所有目标的大概位置。
2. 每一步计算当前点到所有未完成目标的 geodesic distance。
3. 选择最近的未完成目标。
4. 使用 Habitat-Sim 的 GreedyGeodesicFollower 生成低层动作。
5. 执行 move_forward / turn_left / turn_right / stop。
6. 如果 stop 时到达某个目标附近，则标记该目标完成。
7. 重复直到目标完成或预算耗尽。
```

其中具体导航动作不是由大模型产生，而是由 Habitat 的 shortest-path follower 根据 navmesh 自动生成。

这个 baseline 的意义：

- 给一个简单、稳定、可解释的规则 baseline。
- 测试任务在不同 budget 下的基础难度。
- 作为后续模型方法的对照。

它的局限：

- 只看当前最近目标，不考虑全局顺序。
- 不考虑目标价值。
- 知道目标位置，因此是 oracle baseline。
- 不能代表真实 VLN agent 的视觉语言推理能力。

## 7. 更强的理论上限 Baseline

师兄提到可以参考 A* 或 Dijkstra。这里需要区分两层：

### 7.1 点到点最短路径

如果问题是：

```text
当前位置到目标 A 怎么走最短？
```

这个可以用 A* / Dijkstra / Habitat pathfinder 解决。

在 Habitat 里，`pathfinder` 已经能计算两个点之间的 geodesic shortest path。当前 greedy baseline 就用了这类距离来选择最近目标。

### 7.2 多目标有限预算最优顺序

我们的更核心问题是：

```text
给定起点 S、多个目标 A/B/C/D、一个 step budget，
选择什么访问顺序可以在预算内完成最多目标？
```

这个更像：

- Orienteering Problem
- Budgeted TSP
- Prize-Collecting TSP

由于当前每条任务目标数很少，一般 2 到 4 个，所以可以不引入复杂优化器，直接枚举所有目标顺序。

例如目标是 A、B、C：

```text
S -> A -> B -> C
S -> A -> C -> B
S -> B -> A -> C
S -> B -> C -> A
S -> C -> A -> B
S -> C -> B -> A
```

对每个顺序估计实际 step cost，然后在每个 budget 下选完成目标数最多的顺序。

这个可以作为：

```text
Oracle optimal ordering baseline
```

它比 nearest-target greedy 更强，因为它全局考虑目标顺序；但它仍然是 oracle，因为它知道目标位置和路径代价。

可选实现方式：

- 简单枚举 permutation：适合 2 到 4 个目标。
- 动态规划：状态为 `(当前目标, 已完成目标集合, 已用步数)`。
- Dijkstra/A* over state space：如果以后目标数变多，可以考虑。

当前更推荐先做 permutation oracle，因为简单、透明、足够覆盖当前数据。

## 8. 通用 VLN 模型方向

师兄提到的真正创新点可能在这里：

找一个通用 VLN 模型，让它接受比较自由的 prompt，并把时间限制写进 prompt。

示例 prompt：

```text
You have 80 steps left.
Complete as many targets as possible before the budget runs out.
Instruction: Take the box in the bedroom to the dining area table and then retrieve the lamp from there.
Completed targets: none.
Remaining targets: box, table, lamp.
What is the next action?
```

模型输出动作：

```text
move_forward
turn_left
turn_right
stop
```

当模型输出 `stop` 时，系统检查是否到达某个目标附近：

- 如果到了，标记该目标完成。
- 如果没到，记录 failed stop。
- 模型不一定需要自己判断是否成功，成功判定可以由系统负责。

这个实验可以检验：

- 模型是否能理解剩余时间。
- 模型是否能主动调整目标顺序。
- 模型是否能在预算不足时优先完成一部分目标。
- 模型是否具有灵活的 prompt adaptation 能力。

这部分比 oracle greedy 更接近真实 time-aware VLN，也更可能成为创新点。

## 9. 当前结果

当前已经在 `val / batch_6` 跑通 41 条任务。

| Budget ratio | Episodes | Success@Budget | Completion rate | Reward rate | Avg. time used |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 41 | 0.4634 | 0.5772 | 0.5772 | 62.17 |
| 0.75 | 41 | 0.8049 | 0.8537 | 0.8537 | 72.63 |
| 1.00 | 41 | 0.9268 | 0.9634 | 0.9634 | 75.98 |
| 1.25 | 41 | 0.9512 | 0.9715 | 0.9715 | 77.02 |

这个结果目前主要用于确认链路跑通，不应直接当作最终实验结论。后续需要跑完整 test split。

## 10. 建议的下一步实验顺序

1. 保留 nearest-target greedy baseline。
2. 实现 oracle optimal ordering baseline，作为理论上限。
3. 跑完整 test split 的 budget curve。
4. 确认最终指标，重点是 `budget -> success/completion` 曲线。
5. 找一个可用的通用 VLN 模型或 LH-VLN 模型变体。
6. 把 remaining budget 写入 prompt，让模型直接输出动作或下一个目标。
7. 比较：
   - nearest-target greedy
   - oracle optimal ordering
   - VLN model with time prompt

## 11. 目前还不确定的问题

需要进一步确认：

1. 通用 VLN 模型具体用哪个？
   - 用 LH-VLN repo 里的模型？
   - 用 NaviLLM / NavGPT / 其他现成 VLN 模型？
   - 还是先做一个 LLM planner + Habitat follower 的中间版本？

2. 模型输入到底是什么？
   - RGB/depth 视觉输入？
   - semantic/map 信息？
   - 当前目标列表和位置信息？
   - 如果使用视觉输入，需要先解决当前服务器 EGL/OpenGL 渲染问题。

3. 时间限制怎么写进模型？
   - 只在 episode 开始给一次？
   - 每一步都更新 remaining steps？
   - 是否给 completed targets / remaining targets？

4. `stop` 的语义如何最终定义？
   - 当前设定是 stop 表示尝试完成附近目标，不代表整个 episode 结束。
   - 是否需要额外动作表示 episode stop？

5. 理论上限 baseline 的 cost 用什么？
   - geodesic distance？
   - Habitat follower 实际 step 数？
   - 更推荐实际 step 数，因为 budget 本身就是 step。

6. target position 是否可以作为 oracle 信息使用？
   - Greedy 和 optimal ordering 都依赖目标位置。
   - 需要在论文或报告中明确它们是 oracle baseline。

7. 所有目标 value 是否都设为 1？
   - 当前师兄建议先设成一样的重要性。
   - 后续是否需要区分 pickup / place / key object 等不同价值？

8. 最终数据划分和报告范围是什么？
   - 是否只报告 test split？
   - validation 是否只用于开发？
   - 是否需要重新划分或过滤 target position 缺失严重的样本？

9. 是否保留原 LH-VLN 有序任务作为对照？
   - 可以比较 ordered setting 和 unordered time-aware setting 的差异。

10. 如果模型无法判断是否成功，系统判定是否足够合理？
    - 当前设计是系统根据距离判定目标完成。
    - 需要确认这是否符合最终 benchmark 设定。
