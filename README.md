# Time-Aware VLN

本分支基于 LH-VLN 改造一个 time-aware VLN benchmark。核心问题是：在一条长指令里包含多个无序目标时，agent 面对限定步数或剩余时间，应该优先完成哪些目标、怎样在最短时间里做最有意义的事情。

当前阶段先把任务格式、仿真链路和 oracle-style greedy baseline 跑通，用来作为后续模型方法的基准。

## 当前目标

- 任务形式：一条 prompt，多个目标，无序完成。
- 时间约束：使用 step budget 表示限定时间或剩余时间。
- 评价重点：不是只看最终全部成功，而是看不同预算下完成了多少有价值目标。
- 当前基线：nearest-target greedy，属于搜索效率/路径规划 oracle baseline，还不是 learned VLN policy。

## 当前进度

已经完成：

- 从 `data/episode_task/*.json.gz` 生成 time-aware JSONL 数据。
- 每条样本保留原始 instruction、scene、robot、split、targets、ordered oracle steps 和多档 step budgets。
- 加入 episode 起点信息：`start_position` 和 `start_yaw`。
- target position 优先使用成功轨迹终点；缺失时 fallback 到 Habitat semantic object position。
- 加入 no-render simulator 路径，用于先稳定运行 oracle baseline；当前服务器的 RGB/depth 渲染也已通过 preload 系统 `libGLdispatch.so.0` 跑通，具体见 `docs/lhvln_environment.md`。
- 实现 nearest-target greedy baseline。
- 支持一次跑多个 budget ratios，并导出每个 ratio 的 JSON 结果和总表 CSV。
- 处理 `GreedyFollowerError`，不可达目标会记录为 `abandoned_targets`，不会中断整批实验。

## 路径约定

当前机器上的主要路径：

```bash
代码路径: /file_system/vepfs/algorithm/intern03/mhw/LH-VLN
数据路径: /file_system/nas/algorithm/Intern03/data
Conda 环境: /file_system/vepfs/algorithm/intern03/.conda/envs/lhvln
```

仓库内 `data/hm3d` 已链接到 NAS 上的 HM3D v0.2：

```bash
data/hm3d -> /file_system/nas/algorithm/Intern03/data/versioned_data/hm3d-0.2/hm3d
```

## 环境

激活环境：

```bash
source /file_system/vepfs/algorithm/dujun.nie/miniconda3/etc/profile.d/conda.sh
conda activate /file_system/vepfs/algorithm/intern03/.conda/envs/lhvln
```

也可以直接调用解释器：

```bash
/file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python
```

已验证核心包：

- Python 3.9
- Habitat-Sim 0.3.1
- Torch 2.4.1+cu121
- NumPy 1.23.5
- SciPy 1.10.1
- Numba 0.57.1

注意：上游 `requirements.txt` 里的 `deepspeed==0.6.5` 与 Torch 2.x 不兼容，会因为 `torch._six` 报错。当前 time-aware greedy/oracle baseline 不依赖 deepspeed。

更详细的本地环境记录见：

```bash
docs/lhvln_environment.md
```

## 数据

需要的数据：

```bash
data/hm3d
data/episode_task
data/task
data/step_task
data/time_aware/episodes.jsonl
```

重新生成 time-aware JSONL：

```bash
/file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python \
  tools/inspect_time_aware_data.py \
  --output data/time_aware/episodes.jsonl
```

当前生成统计：

- 总样本数：1052
- train：608
- val：41
- test：403
- 每条任务目标数：2 到 4 个
- budget ratios：`0.5, 0.75, 1.0, 1.25`

## 运行 Greedy Baseline

跑一个 smoke test：

```bash
HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet EGL_PLATFORM=surfaceless \
  /file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python \
  tools/run_time_aware_greedy.py \
  --split val \
  --limit 2 \
  --budget-ratios 0.5,1.0 \
  --output-dir output/time_aware/smoke \
  --summary-csv output/time_aware/smoke/summary.csv \
  --quiet
```

跑完整 validation sweep：

```bash
HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet EGL_PLATFORM=surfaceless \
  /file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python \
  tools/run_time_aware_greedy.py \
  --split val \
  --limit 0 \
  --budget-ratios 0.5,0.75,1.0,1.25 \
  --output-dir output/time_aware/greedy_val \
  --summary-csv output/time_aware/greedy_val/summary.csv \
  --quiet
```

输出文件：

```bash
output/time_aware/greedy_val/greedy_val_budget_0p5.json
output/time_aware/greedy_val/greedy_val_budget_0p75.json
output/time_aware/greedy_val/greedy_val_budget_1p0.json
output/time_aware/greedy_val/greedy_val_budget_1p25.json
output/time_aware/greedy_val/summary.csv
```

## 当前 Validation 结果

当前结果来自 `val / batch_6`，共 41 条任务。

| Budget ratio | Episodes | Success@Budget | Completion rate | Reward rate | Avg. time used |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 41 | 0.4634 | 0.5772 | 0.5772 | 62.17 |
| 0.75 | 41 | 0.8049 | 0.8537 | 0.8537 | 72.63 |
| 1.00 | 41 | 0.9268 | 0.9634 | 0.9634 | 75.98 |
| 1.25 | 41 | 0.9512 | 0.9715 | 0.9715 | 77.02 |

指标含义：

- `Success@Budget`：预算内是否完成全部目标。
- `Completion rate`：预算内完成目标比例。
- `Reward rate`：预算内获得的目标价值比例；当前每个目标默认 value 都是 1。
- `Avg. time used`：平均实际消耗 step。

## 当前代码入口

```bash
configs/time_aware_vln.yaml
tools/inspect_time_aware_data.py
tools/run_time_aware_greedy.py
habitat_base/time_aware_simulation.py
docs/lhvln_environment.md
```

## 下一步

- 加一个比 nearest-target greedy 更强的 budget-aware baseline，例如 value-per-step 或剩余时间规划。
- 明确最终论文实验要使用的 time-aware 指标和日志格式。
- 跑完整 test split sweep，并检查 target-position coverage 与 unreachable target 处理。
- 把当前 benchmark 格式接到模型 inference，而不只是 oracle-style greedy。
- 接通 RGB/depth 视觉输入路径，并评估真正的 VLN 模型。

## 来源

本项目基于上游 LH-VLN 改造：

```text
Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method
CVPR 2025
https://github.com/HCPLab-SYSU/LH-VLN
```
