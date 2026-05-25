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
- 新增 no-render simulator 路径，避免渲染环境问题阻塞 oracle baseline。
- 当前服务器的 Habitat RGB/depth 渲染问题已定位为 GLVND 库混用；运行 `--render` 时需要 preload 系统 `libGLdispatch.so.0`，具体命令见 `docs/lhvln_environment.md`。
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

对每个顺序用 Habitat follower 实际执行，然后在每个 budget 下选完成目标数最多、reward 最高、用时更少的顺序。

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

当前已实现 `tools/run_time_aware_oracle_ordering.py`：

```bash
HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet EGL_PLATFORM=surfaceless \
  /file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python \
  tools/run_time_aware_oracle_ordering.py \
  --split val \
  --limit 2 \
  --budget-ratios 0.5,1.0 \
  --output-dir output/time_aware/oracle_smoke \
  --summary-csv output/time_aware/oracle_smoke/summary.csv \
  --quiet
```

它和 greedy 的区别是：

- greedy 每一步只选当前最近目标。
- oracle ordering 会枚举完整目标顺序，再选择预算内最优顺序。
- 两者都仍然知道目标位置，所以都属于 oracle-style baseline。

## 8. 通用 VLN 模型方向

师兄提到的真正创新点可能在这里：

找一个通用 VLN 模型，让它接受比较自由的 prompt，并把时间限制或时间紧迫程度写进 prompt。

目前可以中和成两个 prompt setting：

### 8.1 显式 step budget prompt

每一步给模型的 prompt 都包含本 episode 的总时间限制，即 total step budget。这样模型在整个交互过程中都能持续看到任务的总体时间约束。也可以做 ablation，比较是否加入当前已用步数或 remaining steps。

示例 prompt：

```text
The total time budget for this episode is 80 steps.
Complete as many targets as possible before the budget runs out.
Instruction: Take the box in the bedroom to the dining area table and then retrieve the lamp from there.
Completed targets: none.
Remaining targets: box, table, lamp.
What is the next action?
```

### 8.2 模糊时间压力 prompt

另一种设置是不直接给精确 step 数，而是所有 prompt 都使用自然语言的轻重缓急描述，例如：

```text
Time condition: The time is sufficient.
Time condition: The time is tight.
Time condition: The time is relatively insufficient.
```

或者更自然地写成：

```text
You are in a hurry. Complete as much as possible.
The time is relatively tight. Prioritize useful progress.
You have enough time. Try to complete the whole instruction.
```

这个设置不强行把 `in a hurry` 人为映射成某个固定 ratio，而是测试模型是否会因为自然语言里的时间压力不同而改变策略。它更适合检验模型的 prompt adaptation 和 time-awareness。

### 8.3 对比原则

两个方向都合理：

- 显式 step budget 更可控、适合标准 benchmark 和 budget curve。
- 模糊时间压力更贴近自然语言 prompt，适合体现模型灵活性和创新点。

实际实验可以两种都跑，观察哪种更有区分度。如果显式 step budget 的曲线更稳定，就作为主结果；如果模糊时间压力能明显改变模型策略，可以作为 time-aware prompt adaptation 的重点实验。

为了保证可复现，模型推理时需要固定解码参数：

```text
temperature = 0
top_p = 1.0 或固定值
top_k = 固定值或关闭
seed = 固定
```

并在实验记录中保存完整 prompt、模型版本和解码配置。

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

- 模型是否能理解显式 step budget 或模糊时间压力。
- 模型是否能主动调整目标顺序。
- 模型是否能在预算不足时优先完成一部分目标。
- 模型是否具有灵活的 prompt adaptation 能力。

这部分比 oracle greedy 更接近真实 time-aware VLN，也更可能成为创新点。

### 8.4 当前先实现的 NavGPT-style Planner 版本

为了先把模型接入链路做稳，当前先采用一个中间版本：

```text
LLM / Planner 只负责高层目标选择：
给定 instruction、剩余目标、已完成目标、时间信息，
输出下一步应该去哪个 target index。

Habitat follower 负责低层导航动作：
根据 planner 选择的目标，自动生成 move_forward / turn_left / turn_right / stop。
```

这样做的好处是：

- 先验证模型是否真的会因为时间限制改变目标选择。
- 避免一开始就被动作级视觉导航、碰撞、渲染、历史图像输入等问题卡住。
- 可以直接和 nearest-target greedy、oracle ordering 对齐比较，因为三者都在“目标顺序选择”这个层面竞争。

当前入口是：

```bash
tools/run_time_aware_llm_planner.py
```

它支持三类 planner：

- `nearest`：复用 planner 框架，但规则选择最近目标，用于 smoke test。
- `first` / `random`：简单 sanity check。
- `llm`：把 prompt 通过 stdin 交给外部命令，外部命令在 stdout 输出目标 index。

当前已经加入 DeepSeek API selector：

```bash
tools/deepseek_target_selector.py
```

使用方式：

```bash
export DEEPSEEK_API_KEY="你的 DeepSeek API key"

tools/run_time_aware_llm_planner.py \
  --planner llm \
  --llm-command "python tools/deepseek_target_selector.py --model deepseek-chat"
```

DeepSeek selector 的解码参数默认固定为：

```text
temperature = 0
top_p = 1
max_tokens = 32
thinking = auto
```

这样可以尽量保证同一 prompt 下的结果可复现。API key 只从环境变量读取，不写入代码、不提交到 GitHub。

当前 prompt 模式：

- `explicit`：每一步给 total step budget、used steps、remaining steps。
- `fuzzy`：给模糊时间压力，例如 `sufficient`、`tight`、`insufficient`。
- `none`：不提供时间信息，用作 ablation。

后续真实 LLM 实验可以先不做动作级输出，而是先看：

```text
同一个任务、同一个 budget 下，
LLM 选出来的目标顺序是否比 nearest greedy 更合理，
是否接近 oracle ordering。
```

如果高层目标选择已经有区分度，再继续扩展到动作级 VLN 输出。

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
2. 使用已实现的 oracle optimal ordering baseline，作为理论上限参考。
3. 跑完整 test split 的 budget curve。
4. 确认最终指标，重点是 `budget -> success/completion` 曲线。
5. 用 `tools/run_time_aware_llm_planner.py` 接入一个真实 LLM，让它先输出下一个目标 index。
6. 对比显式 step budget prompt 和模糊时间压力 prompt。
7. 如果目标选择结果有意义，再扩展到动作级 VLN 输出。
8. 比较：
   - nearest-target greedy
   - oracle optimal ordering
   - LLM planner with time prompt
   - 后续动作级 VLN model with time prompt

## 11. V2: Scene-Level Time-Aware VLN

根据老师意见，下一阶段不再只用原始 LH-VLN 单条 episode，而是基于同一个 scene 拼接多个目标点，构造更长、更接近搜索的 scene-level time-aware benchmark。

### 11.1 核心变化

V1 当前设定：

```text
一条 LH-VLN episode
通常 2-4 个 targets
目标位置已知
研究目标顺序选择
```

V2 目标设定：

```text
一个 HM3D scene
拼接 4-8 个目标点 / 事件
形成一个更长的 scene-level episode
agent 在有限时间内完成尽可能多任务/目标
```

这样可以让任务更长，目标更多，也更容易体现 time-aware search 和全局取舍能力。

### 11.2 Test Split 初版拼接结果

当前已新增脚本：

```bash
tools/build_scene_level_time_aware_data.py
```

生成 test 版 scene-level 数据：

```bash
/file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python \
  tools/build_scene_level_time_aware_data.py \
  --split test \
  --min-targets 4 \
  --max-targets 8 \
  --coverage 0.8 \
  --budget-ratios 0.5,1.0,1.5 \
  --output data/time_aware_scene/test_episodes.jsonl
```

当前 test split 统计（已在同一 scene 内对重复目标点去重）：

```text
source scenes: 124
source tasks: 403
source targets: 1087
补全轨迹终点后 source targets with positions: 1066
scene-level 数据中 missing target positions: 0
去重后 unique target points: 790
去重删除重复 target points: 276
可拼接 scenes（去重后至少 4 个目标点）: 94
生成 scene-level episodes: 94
每条 scene-level episode 包含目标点数: 4-8，平均 6.0
每条 scene-level episode 涉及原始任务数: 2-5，平均 2.7
对所有 test targets 的覆盖率: 563/1087 = 51.79%
对可拼接 scenes 内 unique targets 的覆盖率: 563/708 = 79.52%
```

去重规则是保守合并：同一真实 scene 内，只有当目标的 `name`、`region_name` 和四舍五入后的 3D `target_position` 都一致时，才认为是同一个目标点。由于 y 坐标也进入 key，因此只会合并同一楼层的重复目标。保留的 target 会记录 `duplicate_count` 和 `source_occurrences`，方便追溯它来自哪些原始 LH-VLN 任务。仍然缺少 `target_position` 的目标默认不进入 scene-level benchmark。

需要注意：原始 val split 中每个 scene 的目标点较少，因此可能不适合直接构造稳定的 4-8 target scene-level validation。后续可能需要从 train/test 的 scene 中重新划分一个 scene-level val。

### 11.3 时间预算

老师建议横坐标改为：

```text
0.5, 1.0, 1.5 × 每个 scene-level episode 的枚举最优时间
```

当前 `build_scene_level_time_aware_data.py` 先使用：

```text
oracle_time_proxy_ordered_sum = 被选中目标点对应的原始 ordered oracle steps 之和
```

作为临时 proxy，并生成：

```text
0.5 × proxy time
1.0 × proxy time
1.5 × proxy time
```

这只是数据构造 smoke test。后续需要替换为真正的 scene-level oracle time：

```text
对拼接目标点做枚举或动态规划，
用 Habitat follower 实际执行，
得到完成全部可完成任务/目标的最优时间。
```

### 11.4 Prompt 设计

老师建议后续 prompt 不直接告诉 LLM 精确 step 数，而是实时发送模糊时间压力描述。

也就是说，系统内部仍然使用精确 step budget 评测，但模型看到的是：

```text
time is very limited
time is tight
time is moderate
time is sufficient
```

而不是：

```text
you have 80 steps remaining
```

这样更接近自然语言 time-awareness，也能避免模型只机械利用数字。

### 11.5 方法划分

后续主体可以分成三类：

```text
LLM: 文本 planner，例如 DeepSeek/GPT，输入任务列表、历史、模糊时间压力。
VLM: 图像 + 文本模型，输入当前 observation、历史和模糊时间压力。
SOTA: NavGPT-2 / NaVid / Uni-NaVid 等现有 VLN 方法。
```

如果完整 VLN 效果不好，可以退一步把问题定义为 time-aware search strategy，比对不同搜索策略在预算下的 completion curve。

## 12. 目前还不确定的问题

需要进一步确认：

1. 通用 VLN 模型具体用哪个？
   - 用 LH-VLN repo 里的模型？
   - 用 NaviLLM / NavGPT / 其他现成 VLN 模型？
   - 还是先做一个 LLM planner + Habitat follower 的中间版本？

2. 模型输入到底是什么？
   - RGB/depth 视觉输入？
   - semantic/map 信息？
   - 当前目标列表和位置信息？
   - 如果使用视觉输入，需要在运行命令中加入 `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0` 等渲染环境变量。

3. 最终采用哪种 time prompt 作为主实验？
   - 显式 total step budget？
   - 每一步更新 remaining steps？
   - 模糊时间压力描述，例如时间宽裕、时间紧张、时间相对不足？
   - 两种都跑，按区分度决定主结果和补充结果？

4. 除了 total step budget，prompt 里是否还要加入动态时间状态？
   - 是否给当前已用步数？
   - 是否给 remaining steps？
   - 是否给 completed targets / remaining targets？

5. `stop` 的语义如何最终定义？
   - 当前设定是 stop 表示尝试完成附近目标，不代表整个 episode 结束。
   - 是否需要额外动作表示 episode stop？

6. 理论上限 baseline 的 cost 用什么？
   - geodesic distance？
   - Habitat follower 实际 step 数？
   - 更推荐实际 step 数，因为 budget 本身就是 step。

7. target position 是否可以作为 oracle 信息使用？
   - Greedy 和 optimal ordering 都依赖目标位置。
   - 需要在论文或报告中明确它们是 oracle baseline。

8. 所有目标 value 是否都设为 1？
   - 当前师兄建议先设成一样的重要性。
   - 后续是否需要区分 pickup / place / key object 等不同价值？

9. 最终数据划分和报告范围是什么？
   - 是否只报告 test split？
   - validation 是否只用于开发？
   - 是否需要重新划分或过滤 target position 缺失严重的样本？

10. 是否保留原 LH-VLN 有序任务作为对照？
   - 可以比较 ordered setting 和 unordered time-aware setting 的差异。

11. 如果模型无法判断是否成功，系统判定是否足够合理？
    - 当前设计是系统根据距离判定目标完成。
    - 需要确认这是否符合最终 benchmark 设定。
