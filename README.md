# BTOM-TransformerLens（基于 TransformerLens 架构模型的针对 Hi-ToM 数据集的可视化可解释性分析）

这是一个面向 **TransformerLens 支持模型**（如 `Qwen2.5`、`Qwen3`）的可解释性研究工作目录。  
核心目标是：在 **Hi-ToM 数据集** 上，对模型的注意力头与中间表征进行归因、聚类和可视化分析。

---

## 

- 使用 TransformerLens / HuggingFace 模型前向，缓存关键 hook 信息。
- 基于 `attribute.py` 做节点级归因（如 `attn_q / attn_k / attn_v / lm_head`）。
- 用 `vis.py` 对 head 归因结果进行聚类并可视化（如簇内/簇间指标、token 可视化）。
- 支持在同一 notebook 中对比 HF 原生模型路径与 TL 路径。

---

## 主要文件

- `test.ipynb`：主实验 notebook（数据读取、前向缓存、归因、聚类、可视化）。
- `attribute.py`：归因图与节点反传逻辑（核心分析代码）。
- `model_hooks.py`：中间层输出抓取、TL/HF 适配与注意力相关工具函数。
- `vis.py`：归因结果聚类与可视化。
- `dequant.py`：量化权重处理与缓存（用于部分量化模型场景）。
- `graph_registry.py`：图结构/缓存辅助。
- `llm.py`、`min_arc.py`、`seeds/common.py`：实验辅助依赖。

---

## 数据

默认在 notebook 中使用：

- `project/BARC-transformerlens/data_uniform/Hi_ToM_order_1.csv`

你也可以替换为自己的 Hi-ToM CSV，只要字段与 notebook 读取逻辑一致（如 `prompt/answer/choices/index`）。

---

## 环境要求（简版）

建议 Python 3.10+，CUDA 环境可用。常用依赖：

- `bitsandbytes`（4bit 场景）
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `tqdm`
- `jupyter`
- `IPython==8.12.0`
- `ipykernel==6.19.2`
- `circuitsvis==1.40.0`
- `torch==2.8.0`
- `einops==0.6.1`
- `transformers==4.55.4`
- `transformer_lens==2.16.1`

可先按你当前环境安装，再按报错补齐缺失包。

---

## 快速开始

1. 进入目录并启动 notebook：


2. 打开 `test.ipynb`，按顺序执行前部单元：
   - 模型加载（HF / TL）
   - Hi-ToM 数据读取与 `Result` 构建
   - 前向缓存与 `outputs` 生成
   - 归因树构建、head 聚类、可视化

3. 如显存紧张：
   - 降低样本数（`n_samples`）
   - 仅保留一条模型路径（HF 或 TL）
   - 减少归因节点数或层数

---
