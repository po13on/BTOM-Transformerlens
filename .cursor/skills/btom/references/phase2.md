# Phase 2 — Cluster → Score → Expand

## Attribution workspace (mandatory sandbox)

All cluster / score / expand cells live in a **fixed mid-notebook band** so the user can watch attribution without scrolling to the notebook end.

**Locate bounds by source match** (indexes drift — re-scan with `notebook_list_cells` / search before each session):

| Bound | Exact landmark source (do not edit/delete/move) |
| ----- | ----------------------------------------------- |
| **Upper** | `visualize_model_heads(root, selected_model, _results, sample=_results[0])` |
| **Lower** | `colored_tokens_multi(*show_attn(random.choice(_results), selected_model, 51, 10, downstreams=tnode.data.nodes, start=_results[0].index_map[0]['start']))#, start=100))` |

**Inside `(upper_idx, lower_idx)` only:**

- Freely **overwrite** existing workspace cells and/or **insert** new code/markdown cells for attribution.
- Why overwrite is allowed here: appending rounds at the notebook end pushes work out of the user’s view; the sandbox is the shared scratchpad for the agent’s attribution trail.
- When inserting, choose an index `i` with `upper_idx < i <= lower_idx` (insert **before** the lower landmark so it remains the bookend). After inserts, re-resolve `lower_idx` — it shifts down.
- Optional markdown `### Round N` inside the sandbox for readability.

**Outside the sandbox — forbidden for Phase 2:**

- Do **not** append cluster/score/expand cells at the notebook end or below the lower bound.
- Do **not** insert above/at the upper bound or modify either landmark cell’s source.
- Phase 1 setup cells above the upper bound stay as usual (re-run by index/id).

**End review:** after the loop, re-run the **existing upper-bound** `visualize_model_heads` cell once. Do not create a second viz cell at the end. The lower-bound `colored_tokens_multi(...)` is only a bookend / optional spot-check — not the default end-review workflow.

## Execution (mandatory)

1. Run **only in** `$PROJECT_ROOT/test.ipynb` via Notebook MCP (`notebook_edit_cell` / `notebook_insert_cell` + `notebook_run_cell` **inside the sandbox**). Pass `notebook_uri` if the tab is open but not “active”.
2. **Forbidden:** Shell / `jupyter_client` / hidden kernel / editing JSON then executing outside UI.
3. Every round `N`: **three logical steps** (never merge cluster+expand into one cell):
   - `# Round N cluster` — only `cluster_heads`
   - `# Round N score` — `d = tnode.data` then Pattern Score DataFrame
   - `# Round N expand` — exact `Node(...)` list + `add_edges` / `add_tnode` / `print_tree`
4. Prefer overwriting/recycling workspace cells when that keeps the mid-notebook trail clear; insert additional cells inside the sandbox when a round needs more space.

## Pattern Score vs Positive Bound

| Name | Source | Use |
| ---- | ------ | --- |
| **Pattern Score** | DataFrame (`attn_patterns` + `get_head_matching_scores`) | Primary screening during Phase 2 |
| **Positive Bound** | Click a circle in `visualize_model_heads` (`show_attn`) | Per-sample visual verification after the loop |

Never conflate the two. Do not batch legacy `colored_tokens_multi(*show_attn(...))` for-loops.

## `A-` indexing (critical)

`index_map['A-']` is often **one past** the cached prompt attention matrix. Helpers (including inside `visualize_model_heads`) already apply −1.

1. Do **not** patch library code to “fix” A−.
2. Ad-hoc numeric query for A−: use `index_map['A-'] - 1` (or last prompt token).
3. Prefer Pattern Score + `visualize_model_heads` for A− goals.
4. **Never** pass raw `index_map['A-']` as `pos_ids` (CUDA OOB can kill the kernel).

## Step 2a — Cluster

```python
d = tnode.data
d.groups, d.metrics, _, _ = cluster_heads(d.attn_attrs_ds,
    threshold=cluster_threshold, strengths=d.top_heads,
    model_config=selected_model.cfg, figsize=(18, 5), width_ratios=(4, 1),
    bar_height=0.5, leaf_font_size=9)
```

- Start at `cluster_threshold` (default 0.35).
- If `len(d.groups) < 2`: lower by 0.05 and re-run **the same** cluster cell.
- Record final threshold and group assignments.

## Step 2b — Pattern Score table

Adapt `attn_patterns` to keys that exist in `index_map`. Do not assume every dataset has `V->VK_*` / `A-->QK_*`.

```python
index_key_sets = [set(r.index_map[0].keys()) for r in _results]
available_keys = set.intersection(*index_key_sets) if index_key_sets else set()

pattern_requirements = {
    'A-->V': {'A-', 'V'},
    'A-->A-': {'A-'},
    'V->VK_C': {'V', 'VK_C'},
    'V->VK_I': {'V', 'VK_I'},
    'A-->QK_C': {'A-', 'QK_C'},
    'A-->QK_I': {'A-', 'QK_I'},
}

attn_patterns = [
    pattern for pattern, required_keys in pattern_requirements.items()
    if required_keys <= available_keys
]

# Goal-directed example: attn_patterns = ['A-->V']

if not attn_patterns:
    missing = {
        pattern: sorted(required_keys - available_keys)
        for pattern, required_keys in pattern_requirements.items()
        if not required_keys <= available_keys
    }
    raise ValueError(f'No patterns scorable. Missing: {missing}')

df = pd.DataFrame([(l, h, round(score, 4),
    round(mr(eval_head_lens)(_results, selected_model, l, h, strict=False).item(), 4),
    round(mr(eval_head_lens)(_results, selected_model, l, h, strict=True).item(), 4),
    *[round(ap_scores[p].mean().item(), 4) for p in attn_patterns])
    for (l, h), score in d.top_heads.items()
    if (ap_scores := mr(get_head_matching_scores)(_results, attn_patterns, selected_model, l, h))],
    columns=['layer', 'head', 'score', 'acc', 'acc0'] + attn_patterns)
print(df.to_csv(sep='\t', index=True))
```

**Always score self / previous-token baselines** when keys exist: `A-->A-` and especially **`V->V` / AnswerSpan→AnswerSpan**. Do not force-label a head `A-->A-` when `V->V` is equal or stronger.

**Weak columns:** if a pattern is ~0 across candidates (often `A-->QK_I` when the story object is unique), do **not** invent heads for it; note the absence. Object binding often lives on the V / AnswerSpan side (`V->VK_I`).

For unlabeled / seed-only datasets, see [unlabeled-discovery.md](unlabeled-discovery.md).

## Step 2c — Select + expand (position-frontier)

Sources: (1) clusters → dominant pattern → `Node`; (2) DataFrame columns above `attn_pattern_threshold` (default 0.3; ±0.05).

Frontier filter after Stage 0: cover **both** A−-side and V-side when candidates exist. Cap ~2–6 nodes/round; prefer **new roles/frontiers**. Dedupe by `(layer, head, node_type, pattern)`.

```python
nodes = [Node(l, h, node_type, attn_pattern=pattern) for ...]
for n in nodes:
    add_edges(graph, n, tnode.data.nodes, tnode.data.attr)
tnode = add_tnode(_results, selected_model, nodes, parent=tnode)
print_tree(root)
# update active_frontiers; append role ledger rows
```

### Cluster → node type

| Dominant pattern | Node type | Typical role |
| ---------------- | --------- | ------------ |
| `A-->V` (high) | `attn_k` and/or `attn_q` (exits: prefer q+k+v union) | Answer mover |
| `A-->A-` (high) | `attn_v` | Final-position assembler |
| `V->V` / AnswerSpan self (high) | `attn_v` | AnswerSpan-self — **not** A-->A- |
| `V->VK_*` (high) | `attn_k` | Character/Item→answer binder |
| `A-->QK_*` (high) | `attn_q` | Query Character/Item aligner |

Skip clusters with no clear dominant pattern.

### Frontier updates after hang

| Hung pattern | Frontiers |
| ------------ | --------- |
| `A-->V` | `A-`, `V` |
| `A-->A-` | `A-` |
| `V->VK_C` / `V->VK_I` | `V`, `VK_*` |
| `A-->QK_C` / `A-->QK_I` | `A-`, `QK_*` |

## Stage 0 / Stage 1+ (open-ended)

**Stage 0:** Score `A-->V`; top **2** exits; hang **each** × `{attn_k, attn_q, attn_v}` (~6 nodes). Frontiers `{A-, V}`. Do **not** split exits only-k vs only-q.

**Stage 1+:** Expand both frontiers; prefer same-pathway consistency across multiple Stage 0 exits. Prefer union of pathways on clear hubs when budget allows.

**Robustness** = same pathway, multiple exits agree — **not** swap-q/k-then-intersect.

## Bookkeeping

- `round_i` starts at 1; after each successful expand, record round, hung nodes, frontiers, thresholds, tree snippet.
- Default: skip mid-loop viz; after the loop, re-run the existing upper-bound `visualize_model_heads` cell once.
- Zero candidates after lowering thresholds → **exhausted thresholds** stop.

## Agent-set stops

1. **Expansion failure** — `add_tnode` / `add_edges` error
2. **Exhausted thresholds** — `cluster_threshold`≤0.1 and `attn_pattern_threshold`≤0.05 still empty
3. **Stagnation** — same `top_heads` / empty deduped candidates / no new roles

On stop: report tree + roles (open-ended) or accumulated matches (goal-directed); re-run upper-bound viz once if not done; **STOP**.
