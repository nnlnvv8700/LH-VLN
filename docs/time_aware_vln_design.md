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

### 8.4 NavGPT 接入版本

当前已将路线切到直接做 NavGPT-style agent，不再把 DeepSeek target selector 作为主实验。需要注意：官方 NavGPT 代码是为 R2R/Matterport 离散 viewpoint graph 写的，输入是预生成的文字 observation、candidate viewpoint ID 和 R2R annotation；我们的 benchmark 是 HM3D/Habitat 连续动作环境。因此不能原封不动运行官方 `NavGPT.py`，需要一个 Habitat adapter。

当前已 clone 官方仓库用于参考：

```text
/file_system/vepfs/algorithm/intern03/mhw/NavGPT
https://github.com/GengzeZhou/NavGPT
```

当前新增 adapter：

```text
tools/run_time_aware_navgpt.py
tools/deepseek_navgpt_action_selector.py
```

它保留 NavGPT 的核心交互方式：

```text
History + Observation + Valid actions
Thought: ...
Action: action_maker
Action Input: "move_forward x 3"
```

但把 action space 换成 Habitat 低层动作：

```text
move_forward
turn_left
turn_right
stop
```

为了适配连续 Habitat 控制，adapter 支持 short macro-action：

```text
move_forward x N
turn_left x N
turn_right x N
```

其中 `N` 会被 `--max-action-repeat` 限制，默认最多 4。这样避免每 0.25m 小步都调用一次 API。

当模型输出 `Final Answer: Finished!` 或 `stop` 时，系统尝试完成附近目标；如果附近没有目标，则记为 failed stop。

时间 prompt 采用实时模糊描述，不告诉模型具体 step 数：

```text
Overall condition: Time is very limited / moderate / sufficient.
Current urgency: Time is tight / almost exhausted / relatively sufficient.
```

其中 overall condition 来自 `0.5/1.0/1.5 × oracle_optimal_time`，current urgency 来自当前剩余预算比例。

当前 smoke test 已通过：

```bash
HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet EGL_PLATFORM=surfaceless \
/file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python \
  tools/run_time_aware_navgpt.py \
  --episodes /file_system/nas/algorithm/Intern03/data/time_aware_scene/all_episodes_spatial_start_floor_oracle_time.jsonl \
  --split all \
  --limit 1 \
  --budget-ratios 0.5 \
  --max-decisions 3 \
  --max-action-repeat 3 \
  --navgpt-command "/file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python tools/deepseek_navgpt_action_selector.py --mock-action move_forward --mock-repeat 3" \
  --output-dir output/time_aware_scene/navgpt_smoke \
  --summary-csv output/time_aware_scene/navgpt_smoke/summary.csv \
  --quiet
```

真实 DeepSeek/NavGPT-style 运行示例：

```bash
export DEEPSEEK_API_KEY="你的 key"

HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet EGL_PLATFORM=surfaceless \
/file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python \
  tools/run_time_aware_navgpt.py \
  --episodes /file_system/nas/algorithm/Intern03/data/time_aware_scene/all_episodes_spatial_start_floor_oracle_time.jsonl \
  --split all \
  --limit 2 \
  --budget-ratios 0.5 \
  --max-action-repeat 4 \
  --navgpt-command "/file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python tools/deepseek_navgpt_action_selector.py --model deepseek-chat" \
  --observation-mode semantic_text \
  --output-dir output/time_aware_scene/navgpt_deepseek_smoke \
  --summary-csv output/time_aware_scene/navgpt_deepseek_smoke/summary.csv \
  --quiet
```

`observation-mode` 当前有三种：

```text
semantic_text: 使用 Habitat semantic sensor，把当前 front camera 里可见物体类别转成文字 observation，更接近原生 NavGPT 的视觉文本输入。
oracle_text: prompt 中给目标名称、区域和当前到目标的 shortest-path distance，用于调试链路，不作为最终主实验。
blind: 不给视觉/距离/坐标信息，只给 instruction/history/progress，作为无视觉输入下限。
```

当前 `semantic_text` smoke test 已通过，示例 observation：

```text
Current visual observation from the front camera:
- visible object/category: book
- visible object/category: cabinet
- visible object/category: chair
Current progress: completed 0/8 targets.
Remaining target descriptions:
- target 0: bag in Office
- target 1: laptop in Office
...
```

运行 `semantic_text` 需要开启 Habitat render / semantic sensor，并使用服务器 EGL 修复环境变量：

```bash
EGL_PLATFORM=surfaceless \
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0 \
HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet HABITAT_GPU_DEVICE_ID=0 \
/file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python \
  tools/run_time_aware_navgpt.py \
  --episodes /file_system/nas/algorithm/Intern03/data/time_aware_scene/all_episodes_spatial_start_floor_oracle_time.jsonl \
  --split all \
  --limit 2 \
  --budget-ratios 0.5 \
  --max-action-repeat 4 \
  --observation-mode semantic_text \
  --render \
  --navgpt-command "/file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python tools/deepseek_navgpt_action_selector.py --model deepseek-chat" \
  --output-dir output/time_aware_scene/navgpt_deepseek_semantic_smoke \
  --summary-csv output/time_aware_scene/navgpt_deepseek_semantic_smoke/summary.csv \
  --quiet
```

更进一步的 VLM 版本可以把 front/left/right RGB 图像交给多模态模型生成 caption，再写入同一个 NavGPT prompt。

DeepSeek action selector 的解码参数默认固定为：

```text
temperature = 0
top_p = 1
max_tokens = 128
```

API key 只从环境变量读取，不写入代码、不提交到 GitHub。

最新视觉动作级 smoke 进展：

```text
output/time_aware_scene/navgpt_visual_action_smoke_v3
```

配置：

```text
episodes: 2
budget ratio: 0.5
observation_mode: semantic_text
time_prompt_mode: dynamic_fuzzy
max_decisions: 40
max_action_repeat: 4
LLM: deepseek-chat
```

结果：

| Episodes | Success@Budget | Completion rate | Avg. time used | Avg. failed stops |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.0000 | 0.0000 | 69.0 | 0.0 |

这说明完整视觉动作级闭环已经跑通：

```text
Habitat render / semantic observation
-> NavGPT-style prompt
-> DeepSeek action output
-> Habitat low-level action execution
-> stop 时系统判定目标完成
```

但当前 `semantic_text + text-only LLM action policy` 还不能有效完成目标。主要现象：

- 修复前模型会因为看到 cabinet / newspaper 等类别反复错误 stop。
- 已修复动作解析 bug：不再从 Thought 文本里误抓 `stop` / `move_forward` 等动作词。
- 已修复 turn feedback：转向不会再被错误反馈为“被障碍物挡住”。
- 已弱化可见物体提示：可见类别只作为 visual cue，不再直接提示 stop。
- 修复后 failed stop 降到 0，但模型主要在场景中探索，仍未靠近并完成目标。

阶段性判断：

```text
视觉动作级 VLN 工程链路已通；
当前瓶颈是策略能力，而不是环境或 API。
仅用 Habitat semantic category 文本作为视觉 observation，对 DeepSeek-chat 来说不足以稳定导航到目标。
```

下一步有两个方向：

```text
1. 接入真正 VLM：把 front/left/right RGB 图像 caption 成更丰富的场景描述，再交给 NavGPT-style prompt。
2. 接入现成 VLN/VLM SOTA：例如 NaVid / Uni-NaVid / NavGPT-2，让模型自身承担视觉导航能力。
```

### 8.5 Habitat Waypoint Adapter

原生 NavGPT 并不是直接输出连续低层动作，而是在 R2R / Matterport3D 的离散 viewpoint graph 中选择候选 viewpoint。因此，当前进一步实现了更接近原 NavGPT 形式的 Habitat waypoint adapter：

```text
tools/run_time_aware_navgpt_waypoint.py
tools/deepseek_waypoint_selector.py
```

该版本每一步流程是：

```text
1. 从当前 agent pose 出发，在周围多个方向采样候选 waypoint。
2. 用 Habitat pathfinder 过滤不可达 waypoint。
3. 对每个候选方向渲染 semantic observation，得到可见物体类别。
4. Prompt 中列出 candidate index / relative direction / 粗略路径长度 / 可见物体 / 目标相关 cue。
5. LLM 输出一个 candidate index 或 STOP。
6. Runner 用 Habitat GreedyGeodesicFollower 执行到该 waypoint。
7. 如果 LLM 输出 STOP，则系统判断是否完成附近目标。
```

这个版本的关键区别：

```text
不是让 LLM 裸控 move_forward / turn_left / turn_right；
而是让 LLM 做更接近原 NavGPT 的 waypoint-level decision。
```

候选 waypoint 不暴露目标坐标，也不暴露具体 step budget。LLM 看到的是：

```text
模糊时间压力；
任务文本；
已完成/剩余目标；
历史 waypoint 选择；
附近目标提示；
候选 waypoint 的相对方向；
候选方向可见的 semantic object categories。
```

当前 mock smoke 已通过：

```text
output/time_aware_scene/navgpt_waypoint_mock_v3
```

真实 DeepSeek waypoint smoke 已跑通：

```text
output/time_aware_scene/navgpt_waypoint_deepseek_smoke_v2
```

配置：

```text
episodes: 2
budget ratio: 0.5
observation_mode: waypoint_semantic_text
time_prompt_mode: dynamic_fuzzy
max_decisions: 25
max_candidates: 6
LLM: deepseek-chat
```

结果：

| Episodes | Success@Budget | Completion rate | Avg. time used | Avg. failed stops |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.0000 | 0.1458 | 71.0 | 0.0 |

两个 episode 各完成了一个目标：

```text
all/00009-vLpv2VX547B/stitched_0: completed target 1 laptop
all/00016-qk9eeNeR4vw/stitched_0: completed target 2 towel
```

这说明 waypoint-level NavGPT-Habitat adapter 比低层动作版更可行：

```text
低层动作版 semantic_text smoke: completion 0.0
waypoint 版 semantic_text smoke: completion 0.1458
```

当前复跑脚本：

```bash
export DEEPSEEK_API_KEY="你的 key"

LIMIT=2 BUDGET_RATIOS=0.5 \
OUTPUT_DIR=output/time_aware_scene/navgpt_waypoint_deepseek_smoke_v2 \
SUMMARY_CSV=output/time_aware_scene/navgpt_waypoint_deepseek_smoke_v2/summary.csv \
scripts/run_deepseek_waypoint_smoke.sh
```

下一步建议以 `run_time_aware_navgpt_waypoint.py` 作为完整视觉 VLN 主线，而不是继续使用裸低层动作 runner。

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

1. 以 scene-level `*_oracle_time.jsonl` 作为唯一实验数据入口。
2. 使用枚举 oracle optimal ordering 作为理论上限；不再把 nearest greedy 作为上限。
3. 直接跑 NavGPT-Habitat adapter 的小样本真实 API smoke test。
4. 如果小样本行为正常，再跑 103 个 scene-level episodes 的 budget curve。
5. 对比不同 observation setting：
   - `semantic_text`：主实验入口，当前视野语义物体文本描述，更接近 NavGPT。
   - `oracle_text`：调试/诊断用，含目标距离信息，不作为最终主结果。
   - `blind`：无距离/坐标信息，作为无视觉输入下限。
   - 后续 `vision_caption`：front/left/right RGB 图像经 VLM caption 后输入。
6. 最终核心图仍是：
   - 横坐标：0.5 / 1.0 / 1.5 × scene-level oracle optimal time。
   - 纵坐标：Success@Budget / Completion rate / Reward rate。

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

### 11.2 Test Split 当前拼接结果

当前已新增脚本：

```bash
tools/build_scene_level_time_aware_data.py
```

当前按 start-floor setting 生成 test 版 scene-level 数据：

```bash
/file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python \
  tools/build_scene_level_time_aware_data.py \
  --split test \
  --min-targets 4 \
  --max-targets 8 \
  --coverage 0.8 \
  --budget-ratios 0.5,1.0,1.5 \
  --min-start-floor-targets 1 \
  --scene-root data/hm3d \
  --benchmark-setting start_floor \
  --output /file_system/nas/algorithm/Intern03/data/time_aware_scene/test_episodes_start_floor.jsonl \
  --summary-csv /file_system/nas/algorithm/Intern03/data/time_aware_scene/test_summary_start_floor.csv
```

当前 test split 统计（已补全 target position、按同楼层去重，并过滤掉起点楼层没有目标的 episode）：

```text
source scenes: 124
source tasks: 403
source targets: 1087
补全轨迹终点后 source targets with positions: 1066
scene-level 数据中 missing target positions: 0
去重后 unique target points: 790
去重删除重复 target points: 276
可拼接 scenes（去重后至少 4 个目标点）: 94
筛掉起点楼层 0 个目标的 scenes: 6
最终生成 scene-level episodes: 88
每条 scene-level episode 包含目标点数: 4-8，平均 6.0
每条 scene-level episode 涉及原始任务数: 2-5，平均 2.7
对所有 test targets 的覆盖率: 526/1087 = 48.39%
对可拼接 scenes 内 unique targets 的覆盖率: 526/708 = 74.29%
```

当前输出文件：

```text
/file_system/nas/algorithm/Intern03/data/time_aware_scene/test_episodes_start_floor.jsonl
/file_system/nas/algorithm/Intern03/data/time_aware_scene/test_summary_start_floor.csv
```

`test_episodes_start_floor.jsonl` 是后续跑实验的主数据，每行一个 scene-level episode。关键字段包括：

```text
episode_id
scene / scene_path / navmesh_path
robot
start_position / start_yaw
targets
floor_heights / start_floor_id
targets_on_start_floor / targets_off_start_floor
time_budgets / budget_ratios
success_distance
benchmark_setting
oracle_time_source / oracle_time_proxy_ordered_sum
```

`test_summary_start_floor.csv` 是快速检查用的统计表，包含每条 episode 的目标数、起点楼层目标数、off-floor 目标数、proxy oracle time 和三档 budget。

去重规则是保守合并：同一真实 scene 内，只有当目标的 `name`、`region_name` 和四舍五入后的 3D `target_position` 都一致时，才认为是同一个目标点。由于 y 坐标也进入 key，因此只会合并同一楼层的重复目标。保留的 target 会记录 `duplicate_count` 和 `source_occurrences`，方便追溯它来自哪些原始 LH-VLN 任务。仍然缺少 `target_position` 的目标默认不进入 scene-level benchmark。

当前 target position 补全策略：

```text
1. 优先使用原始 st_task 中对应目标的成功轨迹 endpoint。
2. 如果 st_task 没有覆盖该目标，则按原始目标顺序读取完整 task.json 中 trial_i 的最后一个位置。
3. 仍然找不到位置的目标不进入 scene-level benchmark。
```

补全后，V1 全量 `data/time_aware/episodes.jsonl` 的 missing target position 从 `488/2794 = 17.47%` 降到 `30/2794 = 1.07%`；当前 start-floor scene-level test 输出中 missing target position 为 0。

当前 benchmark 暂时采用 `start_floor` setting：

```text
只要求每条 scene-level episode 的起点楼层至少有 1 个 target。
所有目标仍保存在 targets 中，并标注 floor_id。
targets_on_start_floor / targets_off_start_floor 用于区分哪些目标和起点同楼层。
```

需要注意：原始 val split 中每个 scene 的目标点较少，因此可能不适合直接构造稳定的 4-8 target scene-level validation。后续可能需要从 train/test 的 scene 中重新划分一个 scene-level val。

### 11.2.1 当前可视化输出

当前已有三个可视化模式：

```text
--real-map：单层 navmesh 俯视图，使用起点高度切片。
--multi-floor：按 y 高度自动分楼层，一张图中显示多个楼层。
--start-floor-only：只显示起点所在楼层，off-floor target 只在右侧文本中标注 hidden-off-floor。
```

主要输出目录：

```text
output/time_aware_scene/visualizations/test_navmesh_all_dedup_v2/
output/time_aware_scene/visualizations/test_navmesh_multifloor_all/
output/time_aware_scene/visualizations/test_navmesh_start_floor_all/
```

目前最建议人工检查使用：

```text
output/time_aware_scene/visualizations/test_navmesh_start_floor_all/
```

这个目录中有 94 张图，对应未筛掉起点楼层无目标 episode 前的 scene-level 数据。若要和 NAS 中 88 条 start-floor benchmark 完全一致，需要用 `test_episodes_start_floor.jsonl` 重新生成一版。

### 11.2.2 最新 all split 空间点合并版本

由于只用 test split 时，同一楼层可用目标点不够多；并且部分点在俯视图中重合严重，目前新增一版更适合作为后续搜索 benchmark 的数据：

```text
先按同一真实 HM3D scene 分组。
选择有起点的楼层中 target point 最多的一层。
只保留这一层的目标点。
把 XZ 平面距离 0.5m 内、且属于同一楼层的目标点合并为一个 spatial waypoint。
每个 scene-level episode 选 4-8 个 spatial waypoints。
因为当前不训练模型，所以 source split 使用 all，即 train / val / test 都作为 benchmark 候选池。
```

生成命令：

```bash
/file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python \
  tools/build_scene_level_time_aware_data.py \
  --split all \
  --min-targets 4 \
  --max-targets 8 \
  --coverage 0.8 \
  --dedup-mode spatial \
  --spatial-merge-distance 0.5 \
  --floor-selection best_with_start \
  --min-start-floor-targets 4 \
  --budget-ratios 0.5,1.0,1.5 \
  --scene-root data/hm3d \
  --benchmark-setting spatial_start_floor_all \
  --output /file_system/nas/algorithm/Intern03/data/time_aware_scene/all_episodes_spatial_start_floor.jsonl \
  --summary-csv /file_system/nas/algorithm/Intern03/data/time_aware_scene/all_summary_spatial_start_floor.csv
```

当前统计：

```text
source scenes: 153
source tasks: 1052
source targets: 2794
source targets with positions: 2764
spatial waypoints after merge: 1019
duplicate / nearby target points merged: 1745
stitched episodes: 103
skipped scenes (<4 spatial waypoints): 50
targets per stitched episode: 4-8，平均 5.8
source tasks per stitched episode: 2-15，平均 6.9
off-floor targets: 0
selected target count distribution: {4: 34, 5: 17, 6: 15, 7: 10, 8: 27}
```

这一步生成过的 proxy-time 版本已经不作为实验入口，旧文件已移动到：

```text
/file_system/nas/algorithm/Intern03/data/time_aware_scene/archive_old_time/
```

对应俯视图输出：

```text
output/time_aware_scene/visualizations/all_spatial_start_floor/
```

这个版本里，每个 `target` 是一个合并后的 spatial waypoint；原始目标/事件保存在 `source_occurrences` 里，`duplicate_count` 表示这个点聚合了多少个原始目标点。导航时应以合并后的 `target_position` 作为要到达的位置，评测时可以把到达该 waypoint 视为完成这个点对应的事件集合。

### 11.3 时间预算

老师建议横坐标改为：

```text
0.5, 1.0, 1.5 × 每个 scene-level episode 的枚举最优时间
```

当前已新增脚本：

```bash
tools/compute_scene_level_oracle_time.py
```

它会对每个 scene-level episode：

```text
1. 用 Habitat GreedyGeodesicFollower 估计 start/targets 两两之间的 step cost。
2. 枚举所有 target 访问顺序。
3. 找到完成全部 spatial waypoints 的最小 step cost。
4. 写入 oracle_optimal_time / oracle_optimal_order。
5. 将 time_budgets 改为 0.5/1.0/1.5 × oracle_optimal_time。
```

当前唯一保留在主目录的正确横坐标版本：

```text
/file_system/nas/algorithm/Intern03/data/time_aware_scene/all_episodes_spatial_start_floor_oracle_time.jsonl
/file_system/nas/algorithm/Intern03/data/time_aware_scene/all_summary_spatial_start_floor_oracle_time.csv
```

统计：

```text
episodes: 103
unreachable full-order episodes: 0
oracle_optimal_time: min=25, median=100, mean=115.6, max=352
budget ratios: 0.5, 1.0, 1.5
oracle_time_source: pairwise_follower_step_enum
```

这份 `*_oracle_time.jsonl` 是后续 baseline / LLM / VLM 实验应该使用的主文件。JSON 中已经清掉旧时间字段，例如 `oracle_time_proxy_ordered_sum`、`source_ordered_gt_step` 和 pairwise cost 矩阵，只保留 `oracle_optimal_time`、`oracle_optimal_order` 和基于它生成的 `time_budgets`。

早期 `build_scene_level_time_aware_data.py` 生成的文件曾临时使用：

```text
oracle_time_proxy_ordered_sum = 被选中目标点对应的原始 ordered oracle steps 之和
```

作为临时 proxy，并生成：

```text
0.5 × proxy time
1.0 × proxy time
1.5 × proxy time
```

这只是数据构造 smoke test，不应作为最终横坐标，相关旧文件已移入 `archive_old_time/`，避免后续误用。

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

### 11.6 第一阶段实验：LLM 目标调度 + Oracle Follower

第一阶段不直接复现完整 LLM-VLN 系统，也不让 LLM 输出 `move_forward`、`turn_left`、`waypoint` 等底层控制。当前先把问题抽象成：

```text
在同一室内场景中，机器人面对若干个无序目标；
在不同时间压力下，LLM 能否动态选择更合适的下一个目标，
从而在预算内完成更多任务？
```

建议阶段名：

```text
Budget-Conditioned Unordered Long-Horizon Navigation with an Oracle Follower
基于模糊时间压力的无序长时程导航规划
```

当前职责划分：

```text
LLM：只负责高层目标调度，即从 remaining targets 中选下一个 target index。
Habitat follower：负责底层最短路执行，生成 move_forward / turn_left / turn_right / stop。
系统评测器：负责 stop 后的目标完成判定，以及 budget curve 指标统计。
```

这样可以把“时间感知任务规划能力”与以下因素解耦：

- 视觉识别；
- 局部避障；
- 底层动作控制；
- waypoint 生成；
- 端到端 VLN 模型训练。

Habitat-Sim 已提供 `GreedyGeodesicFollower`，可以根据目标坐标沿 geodesic shortest path 生成底层动作序列。因此它很适合作为第一阶段统一的 oracle low-level navigator。

当前主实验入口：

```bash
tools/run_time_aware_llm_planner.py
```

默认设置已经切到 scene-level time-aware 数据：

```text
episodes: /file_system/nas/algorithm/Intern03/data/time_aware_scene/all_episodes_spatial_start_floor_oracle_time.jsonl
split: all
budget ratios: 0.5, 1.0, 1.5
time prompt: fuzzy
fuzzy time: auto
planner observation: text_only
```

其中 `text_only` 表示 LLM 不看目标坐标、不看 geodesic distance、不看具体 step 数，只看：

```text
模糊时间压力；
原始任务文本；
已完成目标；
剩余目标 index / name / region。
```

后续 SOTA / VLM 接入，例如 NaVid、Uni-NaVid、NavGPT-2，先作为第二阶段，不影响第一阶段实验推进。

当前 DeepSeek smoke test 已跑通：

```bash
HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet EGL_PLATFORM=surfaceless \
  /file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python \
  tools/run_time_aware_llm_planner.py \
  --episodes /file_system/nas/algorithm/Intern03/data/time_aware_scene/all_episodes_spatial_start_floor_oracle_time.jsonl \
  --split all \
  --limit 5 \
  --budget-ratios 0.5,1.0,1.5 \
  --planner llm \
  --time-prompt fuzzy \
  --fuzzy-time auto \
  --planner-observation text_only \
  --llm-command "/file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python tools/deepseek_target_selector.py --model deepseek-chat" \
  --invalid-llm-choice end_episode \
  --output-dir output/time_aware_scene/navgpt_oracle_follower_deepseek_smoke \
  --summary-csv output/time_aware_scene/navgpt_oracle_follower_deepseek_smoke/summary.csv \
  --quiet \
  --save-prompts
```

5 个 scene-level episode 的 smoke result：

| Budget ratio | Episodes | Success@Budget | Completion rate | Reward rate | Avg. time used |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 5 | 0.0000 | 0.3155 | 0.3155 | 69.0 |
| 1.0 | 5 | 0.0000 | 0.5607 | 0.5607 | 138.4 |
| 1.5 | 5 | 0.2000 | 0.7679 | 0.7679 | 202.0 |

本次 smoke test 中 LLM 输出均为有效 target index，没有触发 fallback。这个结果只说明链路已跑通、指标随 budget 增大有基本趋势，不作为最终结论。

后续全量运行命令只需要把 `--limit 5` 改为 `--limit 0`，输出目录换成正式目录。

当前 DeepSeek 全量结果已完成，输出目录：

```text
output/time_aware_scene/navgpt_oracle_follower_deepseek_full
```

完整 103 个 scene-level episode 的结果：

| Budget ratio | Episodes | Success@Budget | Completion rate | Reward rate | Avg. time used |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 103 | 0.0000 | 0.2275 | 0.2275 | 57.76 |
| 1.0 | 103 | 0.0194 | 0.4665 | 0.4665 | 111.20 |
| 1.5 | 103 | 0.3592 | 0.7696 | 0.7696 | 161.39 |

结果文件：

```text
output/time_aware_scene/navgpt_oracle_follower_deepseek_full/planner_llm_fuzzy_all_budget_0p5.json
output/time_aware_scene/navgpt_oracle_follower_deepseek_full/planner_llm_fuzzy_all_budget_1p0.json
output/time_aware_scene/navgpt_oracle_follower_deepseek_full/planner_llm_fuzzy_all_budget_1p5.json
output/time_aware_scene/navgpt_oracle_follower_deepseek_full/summary.csv
```

### 11.7 全量对照实验

为了判断当前 DeepSeek high-level scheduler 是否真的学到了更好的目标调度策略，已经补跑了以下对照：

```text
oracle_order: 按预计算 oracle_optimal_order 执行，作为上限参考。
first: 始终按 target index 从小到大执行，作为简单固定顺序 baseline。
random: 随机选择下一个剩余目标，使用固定 seed，作为随机 baseline。
deepseek_no_time: DeepSeek LLM scheduler，但不提供任何时间压力 prompt。
deepseek_fuzzy: DeepSeek LLM scheduler，提供模糊时间压力 prompt。
```

注意：这里的 `oracle_order` 不重新做慢速全排列枚举，而是直接使用数据集中已经保存的：

```text
oracle_optimal_order
oracle_optimal_time
time_budgets = 0.5 / 1.0 / 1.5 * oracle_optimal_time
```

所有方法仍然共享同一个低层导航器：

```text
Habitat GreedyGeodesicFollower
```

也就是说，这组实验比较的是“高层目标选择/排序能力”，不是低层导航能力。

结果汇总文件：

```text
output/time_aware_scene/comparison_oracle_follower_summary.csv
```

完整 103 个 scene-level episode 的 completion rate：

| Method | 0.5 | 1.0 | 1.5 |
| --- | ---: | ---: | ---: |
| oracle_order | 0.4254 | 0.8525 | 1.0000 |
| first | 0.2338 | 0.5568 | 0.8201 |
| random | 0.1917 | 0.4742 | 0.7624 |
| deepseek_fuzzy | 0.2275 | 0.4665 | 0.7696 |
| deepseek_no_time | 0.2127 | 0.5066 | 0.8069 |

Success@Budget：

| Method | 0.5 | 1.0 | 1.5 |
| --- | ---: | ---: | ---: |
| oracle_order | 0.0000 | 0.3786 | 1.0000 |
| first | 0.0000 | 0.0388 | 0.4757 |
| random | 0.0000 | 0.0194 | 0.3689 |
| deepseek_fuzzy | 0.0000 | 0.0194 | 0.3592 |
| deepseek_no_time | 0.0000 | 0.0000 | 0.4660 |

当前观察：

- `oracle_order` 明显高于其他方法，说明数据集和 budget curve 有足够区分度。
- `deepseek_fuzzy` 在 0.5 档略高于 `deepseek_no_time` 和 `random`，但低于 `first`。
- `deepseek_fuzzy` 在 1.0 和 1.5 档低于 `deepseek_no_time` 和 `first`，与 `random` 接近。
- 因此，目前还不能说模糊时间压力 prompt 带来了稳定收益。
- 当前 DeepSeek scheduler 的主要问题不是低层导航失败，因为 failed stops 和 abandoned targets 基本为 0；问题主要在高层目标选择顺序不够好。

阶段性结论：

```text
第一阶段框架已经跑通：
Budget-conditioned unordered long-horizon planning + oracle follower。

但是当前 text-only DeepSeek scheduler 的目标调度能力还不强，
模糊时间压力 prompt 没有稳定优于 no-time / simple baseline。
```

建议下一步先不要急着上完整 VLN，而是先改高层 scheduler 输入和 prompt：

```text
1. 让 LLM 一次输出完整目标排序，而不是每次只选一个目标。
2. 加 few-shot 示例，明确“时间紧张时优先完成近的/同房间的/能形成连续路线的目标”。
3. 做 oracle_distance ablation：给 LLM 当前到每个目标的 geodesic distance，看它是否能利用距离信息。
4. 如果 oracle_distance 明显提升，说明模型需要更结构化的空间信息。
5. 如果仍然不提升，再考虑从 LLM scheduler 转向显式搜索策略或 VLM/VLN 模型。
```

### 11.8 NaVid / Uni-NaVid 数据导出

当前新增脚本：

```bash
tools/export_scene_level_to_vlnce.py
```

它把我们的 scene-level time-aware 数据导出为 VLN-CE/NaVid 可读的 `json.gz` 格式，并为三个预算比例分别生成 instruction：

```text
0.5: time is very limited
1.0: time is moderate
1.5: time is sufficient
```

注意：instruction 中不暴露具体 step 数，只暴露模糊时间压力。

已生成文件：

```text
/file_system/nas/algorithm/Intern03/data/time_aware_scene/vlnce_navid/all_timeaware_budget_0p5.json.gz
/file_system/nas/algorithm/Intern03/data/time_aware_scene/vlnce_navid/all_timeaware_budget_1p0.json.gz
/file_system/nas/algorithm/Intern03/data/time_aware_scene/vlnce_navid/all_timeaware_budget_1p5.json.gz

/file_system/nas/algorithm/Intern03/data/time_aware_scene/vlnce_navid/all_timeaware_budget_0p5_gt.json.gz
/file_system/nas/algorithm/Intern03/data/time_aware_scene/vlnce_navid/all_timeaware_budget_1p0_gt.json.gz
/file_system/nas/algorithm/Intern03/data/time_aware_scene/vlnce_navid/all_timeaware_budget_1p5_gt.json.gz
```

同时生成了 NaVid config：

```text
/file_system/nas/algorithm/Intern03/data/time_aware_scene/vlnce_navid/uninavid_timeaware_0p5.yaml
/file_system/nas/algorithm/Intern03/data/time_aware_scene/vlnce_navid/uninavid_timeaware_1p0.yaml
/file_system/nas/algorithm/Intern03/data/time_aware_scene/vlnce_navid/uninavid_timeaware_1p5.yaml
```

这些文件目前每个 ratio 都包含：

```text
episodes: 103
```

导出的 episode 里额外保留：

```text
source_time_aware_episode_id
target_positions
target_names
target_values
time_budget
oracle_optimal_time
```

这些字段用于后续把 NaVid 的执行轨迹映射回我们的 multi-target completion 评测。

### 11.9 当前阻塞点

当前 `lhvln` 环境不能直接运行 NaVid 官方 `run.py`：

```text
ModuleNotFoundError: No module named 'habitat'
```

原因是 NaVid 官方代码需要 Habitat-Lab / VLN-CE 环境，README 推荐：

```text
python=3.8
habitat-sim=0.1.7
habitat-lab=0.1.7
```

而当前 LH-VLN 环境主要是 `habitat_sim` 路线，不包含 Habitat-Lab 的 `habitat.datasets.make_dataset` 等接口。

下一步需要二选一：

```text
A. 新建 vlnce_navid 环境，按 NaVid README 跑官方 Uni-NaVid eval。
B. 把 Uni-NaVid agent 单独接进我们自己的 TimeAwareSceneSimulator runner，保留当前 LH-VLN/HM3D 环境和 multi-target 判定。
```

更推荐先做 A，确认模型和权重可以正常推理；然后做 B，把它接回我们自己的 time-aware completion metric。

### 11.9 NavGPT-style Waypoint + RGB Candidate Smoke

为了更接近原始 NavGPT 的候选视点导航形式，当前新增了 waypoint 版 adapter：

```bash
tools/run_time_aware_navgpt_waypoint.py
tools/deepseek_waypoint_selector.py
```

这个版本不是让 LLM 直接输出 `move_forward / turn_left / turn_right`，而是：

```text
1. Habitat 根据当前位置生成若干 navigable candidate waypoint。
2. 对每个 candidate 渲染当前朝向下的 semantic observation 和 RGB front view。
3. Prompt 中列出 candidate index、方向、路径粗略长度、可见语义物体、target cue 和 RGB 图片路径。
4. LLM 输出 candidate index 或 STOP。
5. Habitat GreedyGeodesicFollower 执行到底层 waypoint。
```

新增参数：

```bash
--save-candidate-images
```

会把每一步每个候选点的真实 RGB 视角保存到：

```text
output/.../candidate_images/
```

并在 JSON trace 中记录 `rgb_image` 路径，便于之后接 VLM caption 或人工检查模型看到的候选视角。

当前 DeepSeek waypoint 小样本命令：

```bash
EGL_PLATFORM=surfaceless \
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0 \
HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet HABITAT_GPU_DEVICE_ID=0 \
/file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python \
  tools/run_time_aware_navgpt_waypoint.py \
  --episodes /file_system/nas/algorithm/Intern03/data/time_aware_scene/all_episodes_spatial_start_floor_oracle_time.jsonl \
  --split all \
  --limit 2 \
  --budget-ratios 0.5,1.0,1.5 \
  --max-decisions 20 \
  --max-candidates 6 \
  --render \
  --time-prompt-mode dynamic_fuzzy \
  --navgpt-command "/file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python tools/deepseek_waypoint_selector.py --model deepseek-chat" \
  --output-dir output/time_aware_scene/navgpt_waypoint_rgb_deepseek_limit2 \
  --summary-csv output/time_aware_scene/navgpt_waypoint_rgb_deepseek_limit2/summary.csv \
  --save-prompts \
  --save-candidate-images \
  --quiet
```

输出：

```text
output/time_aware_scene/navgpt_waypoint_rgb_deepseek_limit2/
```

小样本结果：

| Budget ratio | Episodes | Success@Budget | Completion rate | Avg. time used | Failed stops |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 2 | 0.0000 | 0.1458 | 71.0 | 0.0 |
| 1.0 | 2 | 0.0000 | 0.1458 | 139.5 | 1.5 |
| 1.5 | 2 | 0.0000 | 0.0833 | 152.0 | 0.5 |

当前观察：

- RGB candidate 保存已跑通，共保存 595 张候选视角图。
- DeepSeek 文本模型仍主要依赖 semantic object text，不能真正读取 RGB 图像。
- 更宽预算没有稳定提升，说明当前 semantic-text waypoint prompt 还不够强。
- 下一步应接 VLM/captioner，把每个 candidate RGB 图转成视觉描述，再让 LLM/VLM 做 candidate selection。

### 11.10 DeepSeek-only NavGPT Text Observation

考虑到当前可用接口主要是 DeepSeek 文本 API，后续先按原 NavGPT 的文本化视觉路线推进，而不是强行使用多模态 LLM。

当前已增强 `tools/run_time_aware_navgpt_waypoint.py` 的 candidate observation：

```text
candidate viewpoint
方向 / path 粗略长度 / 探索新颖度
可见语义物体 + 粗略显著程度
是否匹配剩余 target category
历史选择与已完成目标
模糊时间压力
```

示例 candidate 文本：

```text
- index=0, direction=back-right, path=long, exploration=new area,
  visible=cabinet (dominant), keyboard piano (clear), wardrobe (clear),
  target_matches=3:newspaper, 4:cabinet,
  cue=target 3 (newspaper) may match visible newspaper; target 4 (cabinet) may match visible cabinet
```

这更接近 NavGPT 的思路：

```text
Habitat semantic/RGB observation -> textual observation -> DeepSeek LLM selects candidate viewpoint
```

而不是：

```text
DeepSeek directly reads image
```

当前小样本结果：

```text
output/time_aware_scene/navgpt_waypoint_textnav_deepseek_limit2/
```

| Budget ratio | Episodes | Success@Budget | Completion rate | Avg. time used | Failed stops |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 2 | 0.0000 | 0.0833 | 71.0 | 0.5 |
| 1.0 | 2 | 0.0000 | 0.2708 | 120.0 | 1.5 |
| 1.5 | 2 | 0.0000 | 0.2083 | 138.5 | 0.5 |

对比上一版只给较弱 semantic object list / RGB path 的结果：

| Budget ratio | Previous completion | Enhanced text completion |
| --- | ---: | ---: |
| 0.5 | 0.1458 | 0.0833 |
| 1.0 | 0.1458 | 0.2708 |
| 1.5 | 0.0833 | 0.2083 |

观察：

- 增强文本 observation 在 1.0 和 1.5 档明显更好。
- 0.5 档更差，说明紧预算下 prompt 还需要更强调短路径、少探索、谨慎 STOP。
- 1.0 档第一个 scene 完成了 `newspaper -> cabinet -> laptop` 三个目标，说明 DeepSeek-only NavGPT 文本路线可以产生有效的多目标行为。
- 当前主要问题仍是 STOP 策略和候选选择稳定性，后续应先调 prompt，而不是立刻换模型。

后续 prompt 变量需要保持干净：老师要求实时发送模糊时间描述，但不应在不同 budget 档中混入不同策略建议。因此当前已将时间 prompt 改为中性描述：

```text
0.5: Overall time condition: very limited.
1.0: Overall time condition: moderate.
1.5: Overall time condition: sufficient.

Current time status: early in the episode.
Current time status: midway through the episode.
Current time status: late in the episode.
Current time status: nearly out of time.
```

通用导航规则仍然对所有 budget 档一致，例如优先 target match、避免重复视点、只有附近目标提示时才 STOP。这样可以更清楚地观察“模糊时间状态”本身对策略的影响。

当前 waypoint runner 默认启用 `stop_guard`：

```text
LLM 输出 STOP 后，runner 先检查是否存在 nearby target hint。
如果有目标在 stop_hint_distance 内，才真正执行 stop。
如果没有 nearby target hint，则拦截 STOP，并 fallback 到一个候选 waypoint 继续导航。
```

这样做的原因是，LLM 看到某个目标类别不等于 agent 已经到达该目标附近。成功判定仍然由系统根据 geodesic distance 完成；`stop_guard` 只是避免明显过早的 stop 浪费步数和污染 failed-stop 指标。

这个规则对所有 budget 档完全一致，因此不改变时间 prompt 的实验变量。需要做 ablation 时可以加：

```bash
--disable-stop-guard
```

当前 stop guard 小样本结果：

```text
output/time_aware_scene/navgpt_waypoint_textnav_stop_guard_deepseek_limit2/
```

| Budget ratio | Episodes | Success@Budget | Completion rate | Avg. time used | Failed stops | Guard-blocked stops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 2 | 0.0000 | 0.2083 | 71.0 | 0.0 | 3 |
| 1.0 | 2 | 0.0000 | 0.2083 | 139.5 | 0.0 | 1 |
| 1.5 | 2 | 0.0000 | 0.1458 | 154.0 | 0.0 | 1 |

观察：

- `failed_stops` 被压到 0，说明 guard 能有效过滤明显过早的 STOP。
- 0.5 档从之前的 0.0833 提升到 0.2083，紧预算下收益明显。
- 1.0 和 1.5 仍然存在小样本波动，需要扩大到 `limit=10` 后再判断曲线趋势。

进一步的主实验候选是 `auto_stop`：

```text
DeepSeek/LLM 只输出 candidate viewpoint index。
Prompt 不再提供 STOP 作为可选输出。
Runner 每次执行到 waypoint 后自动检查是否有 remaining target 进入 success distance。
如果进入，则 runner 自动执行 stop 并完成目标。
```

这会把“目标完成判定”完全交给系统，LLM 只负责 waypoint selection，更符合当前阶段研究 time-aware waypoint planning 的目标。

运行时：

```bash
--auto-stop
```

DeepSeek selector 需要配套：

```bash
tools/deepseek_waypoint_selector.py --no-stop
```

当前 auto-stop 小样本结果：

```text
output/time_aware_scene/navgpt_waypoint_auto_stop_deepseek_limit2/
```

| Budget ratio | Episodes | Success@Budget | Completion rate | Avg. time used | Failed stops | Auto-completed targets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 2 | 0.0000 | 0.0833 | 71.0 | 0.0 | 1 |
| 1.0 | 2 | 0.0000 | 0.2708 | 142.5 | 0.0 | 4 |
| 1.5 | 2 | 0.0000 | 0.2708 | 152.5 | 0.0 | 4 |

观察：

- auto-stop 将 failed stop 保持为 0。
- 1.0 和 1.5 的 completion 明显好于早期弱 semantic prompt。
- 0.5 较低，可能因为紧预算下候选 waypoint 没有精确落入 success radius；后续可尝试更密集候选或更短 waypoint 半径。
- 如果后续 `limit=10` 曲线稳定，auto-stop 可以作为主实验设定；LLM-stop + stop-guard 作为 ablation。

当前已完成 103 个 scene-level episode 的 auto-stop 全量实验：

```text
output/time_aware_scene/navgpt_waypoint_auto_stop_deepseek_full_ckpt/
```

配置：

```text
episodes: 103
budget ratios: 0.5, 1.0, 1.5
LLM: deepseek-chat
policy: NavGPT-style candidate waypoint selection
stop/completion: runner auto-stop
time prompt: dynamic neutral fuzzy
max_decisions: 20
max_candidates: 6
```

结果：

| Budget ratio | Episodes | Success@Budget | Completion rate | Avg. time used | Failed stops |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 103 | 0.0000 | 0.1531 | 56.84 | 0.0 |
| 1.0 | 103 | 0.0000 | 0.2309 | 98.83 | 0.0 |
| 1.5 | 103 | 0.0000 | 0.2590 | 117.63 | 0.0 |

观察：

- completion 随 budget 增加单调提升，符合 time-aware budget curve 的预期方向。
- success@budget 仍为 0，说明当前方法很难完成 scene-level episode 的全部 4-8 个目标。
- failed stop 为 0，说明 auto-stop 成功消除了 stop 噪声。
- 与 oracle 上限之间仍有明显差距，后续重点应放在候选 waypoint 选择质量、候选生成密度和更强文本 observation 上。

### 11.11 RAM/VFM 视觉文本与全局记忆

为了进一步贴近原始 NavGPT 的“视觉 observation 文本化”思路，当前已接入 RAM/VFM 标签生成模块。runner 对每个 candidate waypoint 渲染 RGB front view 后，使用 RAM 从图像中生成视觉 tags，再和 Habitat semantic sensor 的可见物体一起写入 candidate description。

当前 RAM checkpoint 已放在 NAS：

```text
/file_system/nas/algorithm/Intern03/models/recognize_anything/ram_swin_large_14m.pth
```

对应 runner 参数：

```bash
--vision-text-provider ram
--ram-checkpoint /file_system/nas/algorithm/Intern03/models/recognize_anything/ram_swin_large_14m.pth
--save-candidate-images
```

RAM/VFM 版 observation 形式大致为：

```text
candidate viewpoint
方向 / path 粗略长度 / 探索新颖度
Habitat semantic visible objects
RAM visual tags from candidate RGB
target_matches / target_cue
历史选择与已完成目标
模糊时间压力
```

当前 103 个 scene-level episode 的 RAM/VFM + auto-stop 全量实验输出为：

```text
output/time_aware_scene/navgpt_waypoint_ram_vfm_text_full/
```

结果：

| Budget ratio | Episodes | Success@Budget | Completion rate | Avg. time used | Failed stops |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 103 | 0.0000 | 0.1436 | 57.05 | 0.0 |
| 1.0 | 103 | 0.0000 | 0.2528 | 93.62 | 0.0 |
| 1.5 | 103 | 0.0000 | 0.3053 | 96.70 | 0.0 |

观察：

- RAM/VFM 版在 1.0 和 1.5 档比纯 semantic-text auto-stop 更好，说明视觉 tags 对 candidate selection 有帮助。
- success@budget 仍为 0，原因是 scene-level episode 通常有 4-8 个目标，完整完成难度很高；当前更应关注 completion curve。
- completion 随 budget 增加单调提升，符合 time-aware setting 的基本预期。

同时，当前已在 `tools/run_time_aware_navgpt_waypoint.py` 中加入轻量全局记忆 / 拓扑地图：

```bash
--use-global-memory
```

它不是完整 SLAM，而是 runner 在线维护的 topological memory。每次 agent 选择并到达一个 candidate waypoint 后，runner 会记录或合并一个 memory node：

```text
node position
visit count
last seen decision
进入该 node 的方向
可见 semantic objects
RAM/VFM visual tags
target-related cues
completed targets at this node
candidate RGB image path
```

后续 prompt 会新增：

```text
Global memory / topological map:
Explored topological nodes:
...

Target-related memory:
...

Current frontier summary:
...
```

这样 LLM 不再只看到当前一步周围的 candidate，而是能看到已经探索过的位置、曾经出现过的目标线索、以及当前哪些 candidate 更像 frontier。

当前已完成 mock selector smoke test：

```text
output/time_aware_scene/navgpt_waypoint_global_memory_smoke/
```

配置：

```text
episodes: 1
budget ratio: 0.5
selector: mock candidate 0
render: enabled
auto-stop: enabled
global memory: enabled
```

检查结果：

- prompt 中已出现 `Global memory / topological map`。
- 第一步 memory 为空。
- 第二步开始记录 topological node。
- memory 中能记录 target-related cue，例如 `newspaper`、`cabinet` 等可能匹配目标。

下一步建议：

1. 跑 RAM/VFM + global memory 的 103 episode 全量实验。
2. 与无 memory 的 RAM/VFM 结果比较 completion curve。
3. 如果 global memory 有提升，再作为 NavGPT-style time-aware VLN 的主版本；否则作为 ablation 分析。

## 12. OmniNav 2026 接入方向

老师认为当前 LLM waypoint 框架仍不够完整，后续主线切到 OmniNav 2026。

已确认官方仓库：

```text
https://github.com/amap-cvlab/OmniNav/
```

当前已 clone 到：

```text
third_party/OmniNav
```

OmniNav 中最相关的是：

```text
infer_ovon_slowfast/
```

它的核心不是简单从局部 waypoint 中选择，而是：

```text
360 度视觉观察
+ fog-of-war explored map
+ frontier candidates
+ Qwen2.5-VL slow planner
+ A-star / point-goal fast executor
```

这比当前 `局部候选点 + RAM/VFM tags + LLM` 更接近真实 VLN / exploration setting。

当前已新增数据 adapter：

```text
tools/omninav/export_time_aware_to_omninav.py
```

并已导出 103 个 scene-level episode：

```text
/file_system/nas/algorithm/Intern03/data/time_aware_scene/omninav_time_aware_episodes.jsonl
```

更详细的接入方案见：

```text
docs/time_aware_omninav_integration.md
```

当前 OmniNav 线已推进到两个版本：

### 12.1 QwenVL-only 版本

这个版本让本地 OmniNav / Qwen2.5-VL-3B 同时负责：

```text
读取全部 remaining targets
读取模糊时间压力
观察 360 度图像和 frontier candidates
选择下一个 frontier / waypoint
```

也就是 QwenVL 自己同时做目标取舍和视觉导航，不额外接 DeepSeek 或其他 LLM。

入口脚本：

```bash
bash scripts/omninav/run_qwenvl_all_targets_smoke.sh
```

核心参数：

```text
--planner omninav
--target-scheduler qwenvl-all-targets
```

### 12.2 LLM scheduler + QwenVL navigator 版本

这个版本把任务拆成两层：

```text
外层 LLM scheduler：
  根据模糊时间压力、remaining targets、历史尝试，选择当前 active target index。

内层 OmniNav / QwenVL navigator：
  只接收一个 active target，按原 OmniNav 单目标视觉导航方式选择 frontier / waypoint。

runner：
  由 Habitat 执行动作，并根据环境距离自动判定任意目标是否完成。
```

入口脚本：

```bash
bash scripts/omninav/run_llm_scheduler_qwen_nav_smoke.sh
```

核心参数：

```text
--planner omninav
--target-scheduler time-aware-llm-target
--target-llm-command "..."
```

`--target-llm-command` 是通用接口，不限定 DeepSeek。只要外部命令从 stdin 读取 prompt，并输出一个 remaining target index，就可以替换成任意 LLM。

当前 mock smoke 已跑通：

```text
QwenVL-only: OK
LLM scheduler + QwenVL navigator: OK
parse_failures: 0
```

更详细的环境、命令和 smoke 结果见：

```text
docs/time_aware_omninav_integration.md
```

## 13. 目前还不确定的问题

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
