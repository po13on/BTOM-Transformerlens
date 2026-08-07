# Final report format

**Language:** Match the user. Chinese → 任务结果 / 归因树 / 节点角色. English → Task Result / Attribution Tree / Node Roles.

Section 3 is **required** for open-ended complete-tree runs (and whenever the user asked for a full explanatory circuit).

---

## Section 1: Task Result

### Goal-directed — matches found

```markdown
## 任务结果

✅ **已找到目标注意力头（并继续向上追溯）** — 从 … 关注到 … 的注意力头如下（含多轮累计）：

| 注意力头 (Layer, Head) | 所用指标 | 分数 | 注意力模式 | 节点类型 | 归因层级 |
|------------------------|---------|------|-----------|---------|---------|
| (layer, head) | Pattern Score | 0.XX | A-->V | attn_k | 第 N 层归因 |

> **续归因说明**：首次命中后不会停止；同一目标下继续向上直到 max_attribution_rounds 或 agent-set 终止。
>
> **分数说明**：Pattern Score（DataFrame 列）≠ Positive Bound（visualize_model_heads 点击查看）。
>
> **经验门槛**：match_score_floor 由首轮同指标观测分数推导；用于判定匹配，不是循环停条件。写明指标、门槛、参考头、轮次、停止原因。
>
> **可视化审阅**：已重跑上边界 visualize_model_heads 单元格；请在 notebook 中点击圆点审阅（归因代码仅在上下边界之间）。
```

### Goal-directed — no match

```markdown
## 任务结果

❌ **未找到匹配的注意力头**

已遍历 N 轮。match_score_floor = 0.XX（指标: …；参考头: …），后续候选均未接近该门槛。

| 最接近的候选头 | 所用指标 | 分数 | 与经验门槛差距 |
|---------------|---------|------|---------------|
| (layer, head) | Pattern Score | 0.18 | -0.12 |

> 流程因 [扩展失败 / 阈值耗尽 / 归因停滞] 自动结束。
```

### Open-ended

```markdown
## 任务结果

✅ **归因完成（开放式多轮 / 位置前沿完整树）**

- 模式: open-ended（position-frontier）
- 共执行 N 轮 cluster → score → expand
- 停止原因: [达到 max_attribution_rounds=K / 扩展失败 / 阈值耗尽 / 归因停滞]
- 模型: …
- 前沿演变: 第0轮后 {A-, V}; …
- 每轮扩展摘要: …
```

Seed-only / unlabeled runs: also note which roles were **discovered** vs seeded, and any near-zero patterns skipped.

---

## Section 2: Attribution Tree

Always after Section 1. Exact `print_tree(root)` in a fenced block — **do not reformat**.

```markdown
## 归因树

自动发现的归因电路结构（`print_tree(root)` 输出）：

```
L64┐
   └L59 A-->V x1
      └L55 A-->V x1
```

> **如何阅读**：`L层号 模式 x节点数`；自上而下向输入端追溯。
> **注意**：print_tree 可能折叠同层混合模式；完整角色以 Section 3 / role ledger 为准。
```

For goal-directed: tree **after all continuation rounds**. Complete viz (`visualize_model_heads` once) before the final report when matches exist or when building a complete open-ended tree.

---

## Section 3: Node Roles（完整树必填）

One row per hung attention node (not lm_head). Same `(L,H)` with different `node_type` = separate rows.

```markdown
## 节点角色

| 注意力头 | 模式 | 节点类型 | 活跃前沿 | 角色 | 作用说明 |
|---------|------|---------|---------|------|---------|
| (26, 26) | A-->V | attn_k | V | Answer-key enabler | … |
| (32, 3) | A-->V | attn_q | A- | Answer-query former | … |
| (34, 28) | A-->A- | attn_v | A- | Final-position assembler | … |
| (20, 9) | V->V / AnswerSpan→AnswerSpan | attn_v | V | AnswerSpan-self | 勿误标为 A-->A- |
| (14, 18) | V->VK_C | attn_k | V, VK_C | Character→answer binder | … |
| (26, 7) | A-->QK_C | attn_q | A-, QK_C | Query Character aligner | … |
```

Use readable names (`AnswerSpan→Story.Object`) when the run used unlabeled discovery.
