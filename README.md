# Spatial Accelerator DNN Mapping Optimization

课程《集成电路工程算法》大作业。在 2D mesh spatial accelerator 上比较 ILP / SA / EA / Greedy+KL 四种算法求解 DNN partitioning + placement 联合优化问题的效果。

核心发现：**成本模型与实验 setup 共同决定算法排名**。主实验（含层内通信、允许闲置核心）中 Greedy+KL 12/19、SA 6/19；仅层间通信时 SA 与 Greedy 各 8/19；κ 敏感性扫描与 intra 并行假设 ablation 进一步翻转排名。

## 仓库结构

```text
EDA-alg-spatial-mapping/
├── src/                    # 源码
│   ├── model.py            # 成本模型（计算 + 层间通信 + 层内通信）
│   ├── experiment.py       # 实验入口（--inter-only / --intra-parallel）
│   ├── visualize.py        # 可视化生成（支持 --results 指定输入）
│   ├── baseline.py         # 基线启发式
│   └── solvers/            # ILP / SA / EA / Greedy+KL
├── results/                # 主实验输出
│   ├── experiment_results.json
│   ├── experiment_results_inter_only.json
│   ├── experiment_results_intra_parallel.json
│   └── sensitivity_kappa.json
├── scripts/
│   ├── run_full.sh         # 一键全量复现
│   ├── run_full.ps1
│   ├── sweep_kappa.py      # κ 敏感性扫描（已由 sweep_sensitivity 覆盖）
│   ├── sweep_sensitivity.py # κ/β 敏感性 + 报告用 β=0.1 映射图
│   └── verify_results.py
├── logs/full_reproduction.log
├── report_ieee/report.tex
├── report.md / report.pdf
└── requirements.txt
```

## 环境要求

- Python 3.10+
- `pip install -r requirements.txt`

## Quick Start

```bash
python src/experiment.py
python src/visualize.py
```

## Full Reproduction

```bash
bash scripts/run_full.sh
```

或分步：

```bash
python src/experiment.py                              # 主实验
python src/visualize.py
python src/experiment.py --inter-only                 # 仅层间通信
python src/visualize.py --results results/experiment_results_inter_only.json
python src/experiment.py --intra-parallel             # intra 并行注入 ablation
python src/visualize.py --results results/experiment_results_intra_parallel.json
python scripts/sweep_sensitivity.py                   # 通信敏感性（κ/β + 核心利用率 + β=0.1 映射图）
python scripts/verify_results.py
```

报告 Fig. mapping 左右两栏分别来自：
- 左（主模型 24/64）：`python src/visualize.py` → `mapping_detail_Large-MLP_8x8-mesh.png`
- 右（β=0.1 满核 64/64）：`python scripts/sweep_sensitivity.py --mapping-only` → `mapping_detail_Large-MLP_8x8-mesh_beta0.1.png`

## Expected Outputs

| 文件 | 说明 |
|------|------|
| `results/experiment_results.json` | 主实验（19 配置 × 4 solver × baselines） |
| `results/experiment_results_inter_only.json` | 仅层间通信对照 |
| `results/experiment_results_intra_parallel.json` | intra 项除以 $x_i$ 的 ablation |
| `results/sensitivity_study.json` | 通信敏感性汇总 |
| `results/sensitivity_comm.json` | κ/β 与核心利用率数据 |
| `results/sensitivity_kappa.png` + `sensitivity_core_usage.png` | 敏感性图表 |
| `results/mapping_detail_Large-MLP_8x8-mesh_beta0.1.png` | 报告敏感性对比（β=0.1，Greedy+KL 64/64） |
| `logs/sensitivity_study.log` | 敏感性实验运行日志 |
| `results/*.png` | 报告引用的图表 |

## 运行时长

| 步骤 | 预估时间 |
|------|---------|
| 主实验 + inter-only + intra-parallel | 30–40 分钟 |
| 通信敏感性研究 | 15–20 分钟 |
| visualize.py（全部） | 3–5 分钟 |

ILP 在 4×4 mesh 上约 60s/配置。SA/EA 各 5 次独立运行。

## 随机性说明

- SA/EA：`seed = run * 42`（run=0..4），报告取最优值
- ILP 和 Greedy+KL 确定性
- 求解器允许 $\sum x_i < K$（闲置核心）

## Known Limitations

- 绝对延迟值仅用于相对比较
- ILP 为 placement 子问题参考（固定 partitioning），仅在 ≤16 cores 运行
- 层内通信默认序列化假设；可用 `--intra-parallel` 切换
- Transformer workload 为线性链近似；执行模型为串行非 pipeline

## Report

| 文件 | 说明 |
|------|------|
| `report.pdf` | IEEE 双栏提交版 |
| `report.md` | 中文扩展版 |

重编译 PDF（需 Tectonic）：

```bash
cd report_ieee && tectonic report.tex && cp report.pdf ../report.pdf
```

## License

MIT
