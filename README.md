# Spatial Accelerator DNN Mapping Optimization

课程《集成电路工程算法》大作业。在 2D mesh spatial accelerator 上比较 ILP / SA / EA / Greedy+KL 四种算法求解 DNN partitioning + placement 联合优化问题的效果。

核心发现：**成本模型中是否包含层内 tensor-parallel 通信对算法排名有决定性影响**。仅考虑层间通信时 SA 胜出 12/19 配置；加入层内通信后 Greedy+KL 以 17/19 的优势领先。

## 仓库结构

```text
EDA-alg-spatial-mapping/
├── src/                    # 源码
│   ├── model.py            # 成本模型（计算 + 层间通信 + 层内通信）
│   ├── experiment.py       # 实验入口（支持 --inter-only 对照组）
│   ├── visualize.py        # 可视化生成（支持 --results 指定输入）
│   ├── baseline.py         # 基线启发式
│   └── solvers/            # ILP / SA / EA / Greedy+KL
├── results/                # 主实验输出
│   ├── experiment_results.json          # 主实验数据（含 18 张图）
│   └── experiment_results_inter_only.json  # 对照组数据
├── scripts/                # 辅助脚本
│   ├── run_full.sh         # 一键全量复现（Bash / Git Bash / WSL）
│   ├── run_full.ps1        # 一键全量复现（PowerShell）
│   └── verify_results.py   # 结果自检
├── report.md               # 报告源文件
├── report.pdf              # 提交版 PDF
├── gen_html.py             # Markdown → HTML 导出
├── requirements.txt
└── README.md
```

## 环境要求

- Python 3.10+
- 测试平台：Windows 11 / Python 3.12
- Linux / macOS 理论可运行（无平台特定代码）

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. 跑主实验（含层内通信，约 10-15 分钟）
python src/experiment.py

# 2. 生成主实验图
python src/visualize.py

# 3. 查看报告
# report.md 或 report.pdf
```

## Full Reproduction

完整复现主实验 + inter-only 对照组 + 全部图片：

```bash
# 主实验
python src/experiment.py
python src/visualize.py

# 对照组（仅层间通信）
python src/experiment.py --inter-only
python src/visualize.py --results results/experiment_results_inter_only.json

# 结果自检
python scripts/verify_results.py
```

或使用一键脚本：

```bash
# Bash (Git Bash / WSL / Linux / macOS)
bash scripts/run_full.sh

# PowerShell (Windows)
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1
```

## Expected Outputs

| 文件 | 说明 |
|------|------|
| `results/experiment_results.json` | 主实验数据（19 配置 × 4 solver × baselines） |
| `results/*.png` | 报告引用的 18 张图 |
| `results/experiment_results_inter_only.json` | 对照组数据（可重跑出图） |

仓库已附带作者生成的 canonical outputs，同时也支持从源码重跑。

## 运行时长

| 步骤 | 预估时间 |
|------|---------|
| `experiment.py`（主实验） | 10-15 分钟 |
| `experiment.py --inter-only`（对照组） | 5-10 分钟 |
| `visualize.py` | 1-2 分钟 |

ILP 在 4×4 mesh 上约需 30-60 秒/配置。SA/EA 各跑 3 次独立运行。

## 随机性说明

- SA 和 EA 使用确定性种子 `seed = run * 42`（run=0,1,2），报告取 3 次运行中的最优值
- ILP 和 Greedy+KL 是确定性算法
- 给定相同 Python 版本和依赖版本，结果完全可复现

## Known Limitations

- 绝对延迟值仅用于相对比较，基于分析成本模型和假设参数，不对应真实硬件性能
- ILP 因层内通信变量的二次增长，默认仅在 4×4 mesh（≤16 cores）上运行
- 层内通信模型假设简化的同步模式，未精确模拟 all-reduce / all-gather 的逐链路时序

## Report Export

`report.md` 是报告源文件。如需重新导出 PDF：

```bash
python gen_html.py
# 在浏览器中打开 report.html，Ctrl+P 另存为 PDF
```

## License

MIT
