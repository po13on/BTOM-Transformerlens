---
name: btom
description: Run and supervise BTOM-TransformerLens test.ipynb experiments. Use when the user asks the agent to run, continue, debug, or monitor BTOM-TransformerLens/test.ipynb (located via BTOM_PROJECT_ROOT env var or ~/.btom/config.json), especially when execution should proceed cell-by-cell based on outputs, errors, model-loading state, cache generation, attribution results, clustering summaries, or visualization outputs.
---
# BTOM Notebook Runner

Use this skill to run the BTOM-TransformerLens test.ipynb (located via `BTOM_PROJECT_ROOT` env var or `~/.btom/config.json`) as a supervised experiment rather than as a blind "run all" notebook.

The notebook is stateful, GPU-heavy, and partially exploratory. Treat each code cell output as evidence for whether to continue, skip an optional branch, fix a local setup issue, or stop and report.

## First Principles

1. Prefer running cells incrementally. Do not run the whole notebook blindly unless the user explicitly asks.
2. Preserve the user's notebook unless they ask for edits. If a temporary change is needed for execution, explain it first.
3. **Reuse existing cells for the same function.** When the notebook already contains a cell that does clustering, pattern scoring, tree expansion, tokenization helpers, etc., edit that cell in place (if parameters must change) and re-run it with `notebook_run_cell`. Do **not** insert a new cell that rewrites the same logic. Overwriting that cell's previous output is expected and preferred; creating many near-duplicate cells makes results hard to locate.
4. Treat model loading, CUDA memory, local model paths, and missing weights as stopping conditions unless the fix is obvious and non-destructive.
5. Use the newest user request to define the goal. Do not continue into later attribution or visualization sections if the requested goal was only model loading, data validation, or forward cache generation.
6. **Attribution mode:**
   - **goal-directed** — user names a directed search (e.g. "找 A-→V 的头", "from token X to Y"). Find matching heads, materialize them, then **keep attributing upward** on the new `tnode` to discover more heads with the **same** goal pattern / metric. Do **not** stop at the first match.
   - **open-ended** — user only asks to attribute / discover the circuit / 完善归因树 **without** naming heads or (source, target). Iterate Phase 2 until an agent-set stop or `max_attribution_rounds`. Do **not** stop after a single expand.

## Project Location

This skill does not hardcode the project path. When invoked:

1. Read the `BTOM_PROJECT_ROOT` environment variable.
2. If not set, read `~/.btom/config.json` and extract the `project_root` value.
3. If neither exists, ask: "What is the absolute path to your BTOM-Transformerlens project directory?"
4. Use the resolved path as `$PROJECT_ROOT` throughout: notebook is at `$PROJECT_ROOT/test.ipynb`, data at `$PROJECT_ROOT/data_uniform/`, etc.

To register the project permanently, the user can run:

```bash
mkdir -p ~/.btom && echo '{"project_root": "/absolute/path/to/BTOM-Transformerlens"}' > ~/.btom/config.json
```

Or set the environment variable:

```bash
export BTOM_PROJECT_ROOT=/absolute/path/to/BTOM-Transformerlens
```

## Parameters

### User-set parameters

Extract these from the user's natural language request. Do NOT change these during execution — they reflect the user's explicit intent.

| Parameter               | Default                                           | How to recognize                                                                         |
| ----------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `model_name`          | *(required)* — ask if not provided             | "use Qwen...", "load model from...", "模型路径是...", "用...模型"                        |
| `dataset_path`        | `$PROJECT_ROOT/data_uniform/Hi_ToM_order_1.csv` | "data at...", "数据集在...", "用...csv"                                                  |
| `use_4bit`            | `true`                                          | Set to`false` when user says: "不用量化", "full precision", "no quantization"          |
| `use_transformerlens` | `true`                                          | Set to`false` when user says: "只用 HF", "不用 TL", "skip transformerlens", "原生模型" |
| `source_position`     | *(required if goal specified)* — token index   | "第 42 个 token 关注到...", "from token 15 to...", "主语位置..."                         |
| `target_token`        | *(required if goal specified)* — token index   | "...关注到第 80 个 token", "...attends to 'Mary'", "...to token 80"                      |
| `attribution_mode`    | `open-ended` if no (source,target)/pattern goal; else `goal-directed` | "进行归因", "完善归因树", "自动发现电路", "多轮归因" → open-ended; "找 A-→V", "从 X 关注到 Y" → goal-directed |

### Agent-adjustable parameters

These start with the defaults below. During execution, adjust them based on observed outputs to improve results without asking the user.

| Parameter                  | Default                                | Cell(s)                                                                            | Adjustment guidance                                                                                                                                              |
| -------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `n_samples`              | `max(20, len(df_tom))` (all samples) | `64823fec`                                                                       | Reduce if OOM or execution too slow. Increase if results have high variance across samples.                                                                      |
| `cluster_threshold`      | `0.35`                               | `3697e73c`, `f09e6bbd`, `981ec62f`, `d99547c5`, `409bf345`, `464476a7` | Higher → fewer, tighter clusters. Lower → more, looser clusters. Adjust by ±0.05 if clusters are too granular or too coarse.                                  |
| `attn_pattern_threshold` | `0.3`                                | `06273ae5`, `f0702206`, `96ff4065`                                           | Higher → fewer candidate heads selected (stricter). Lower → more candidates. Adjust by ±0.05 if too many or too few heads are selected for circuit expansion. |
| `match_score_floor`      | *(derived, not user-set)*            | Phase 3 goal check                                                               | **Goal-directed only.** Empirically derived after the first Phase 2 pass from candidate scores of the goal metric. Used to decide which heads **count as matches** each round; finding a match does **not** end the loop — continue upward for more same-pattern matches until a hard stop. |
| `max_attribution_rounds` | `5`                                  | Phase 2 loop                                                                     | Max Phase 2 iterations (cluster→score→expand). Applies to **both** open-ended and goal-directed continuation after the first match. Raise to 6–8 if the tree is still shallow; lower to 3 if OOM / too slow. User may override: "多跑几轮", "最多 3 轮". |

### Mapping user-set parameters to notebook cells

When executing, modify the cell's variables before running:

**model_name** → cell `ce9370d7`: set `model_name = "<value>"`. Also drives cell `67f9525e` TL conversion.

**dataset_path** → cell `64823fec`: set `hi_tom_path = "<value>"`.

**use_4bit = false** → cell `ce9370d7`: remove `quantization_config=bnbconfig`, keep `torch_dtype=torch.float16`. Cell `67f9525e`: remove `load_in_4bit=True`.

**use_transformerlens = false** → Skip cells `67f9525e` through `e203b118` (TL loading and `model_base` deletion). Do NOT delete `model_base`. The default attribution loop is TL-first and expects `selected_model.cfg`, TL cache outputs, and TL attention hooks; only use the HF-only path for explicit HF debugging/sanity checks, or after adapting the TL-dependent cells in Phase 1 and Phase 2.

## Automated Circuit Discovery Loop

This is the core execution loop. After one-time initialization, iterate: cluster → score → expand → repeat.

**Mode switch (decide once after parsing the user request):**

| Mode | When | Loop behavior |
| ---- | ---- | ------------- |
| **goal-directed** | User names a directed attention goal (`source_position`→`target_token`, or a symbolic pattern like A-→V) | After each Phase 2 scoring pass, evaluate the goal metric. Heads at/above `match_score_floor` are **matches**: materialize them into the tree (Step 6), append to the accumulated match list, then **continue** Phase 2 on the new `tnode` looking for **more** same-goal heads further up. If no match this round: still expand via Step 2c (exploration) and continue. Stop only on agent-set conditions or `max_attribution_rounds`. Batch-visualize **all** accumulated matches at the end (Step 7). |
| **open-ended** | User asks only to attribute / build / deepen the circuit tree, **without** naming heads or a (source, target) | After each Phase 2 scoring pass, **always** select candidates and expand (Step 2c), then return to Step 2a on the new `tnode`. Continue until an **agent-set** stop condition or `max_attribution_rounds`. Do not wait for the user to pick heads between rounds. |

### Phase 1: Initialization (run once per session)

Run cells from the beginning through root node creation, respecting all user-set parameters.

When using the `olavovieiradecarvalho.notebook-mcp-server` extension, execute cells by 0-based `index` with `notebook_run_cell`. Use the `cell id` as a verification anchor before running; if the index no longer contains the expected id or preview, call `notebook_list_cells` / inspect the notebook and update the index table before executing.

Only run the rows listed below by default. Current unlisted cells `10` (`9465a27b`, optional GPTQ comparison model) and `16` (`c6f7f769`, commented token-inspection helper) are skipped unless the user explicitly asks for those branches.

| Phase                        | MCP index | Cell id      | Purpose / notes                                                                                       |
| ---------------------------- | --------- | ------------ | ----------------------------------------------------------------------------------------------------- |
| Imports & autoreload         | `0`     | `030bdbc8` | Autoreload and IPython display setup.                                                                 |
| Imports & project modules    | `1`     | `9341f87e` | Environment variables, local imports,`pptree`, `vis`, `attribute`, etc.                         |
| Torch / transformers imports | `2`     | `09d30546` | Torch, Transformer imports, disable grad.                                                             |
| Model cache init             | `3`     | `03d4e10e` | `models = {}`.                                                                                      |
| 4bit config                  | `4`     | `c372a448` | `BitsAndBytesConfig`; skip or edit only if `use_4bit=false`.                                      |
| HF model load                | `5`     | `ce9370d7` | Apply`model_name` and `use_4bit` here. Stop on CUDA/OOM/missing weight errors.                    |
| HF model metadata            | `6`     | `6997eba9` | Cache`model_base`, set `L`, `H`, `V`.                                                         |
| TransformerLens imports      | `7`     | `1d0aa69f` | Import`HookedTransformer`, TL utils, `gc`.                                                        |
| TL conversion                | `8`     | `67f9525e` | Run only if`use_transformerlens=true`; apply `use_4bit` here too.                                 |
| Release HF model             | `9`     | `e203b118` | Run only after TL conversion succeeds. If`use_transformerlens=false`, skip and keep `model_base`. |
| Data loading                 | `11`    | `64823fec` | Apply`dataset_path` and `n_samples`; builds `results`.                                          |
| Hook filter                  | `12`    | `a91ab7f2` | Defines`names_filter` for TL cache; TL-dependent.                                                   |
| Result alias                 | `13`    | `e7af6efa` | `_results = results`.                                                                               |
| TL forward cache             | `14`    | `0904d383` | Run with TL model; writes`r.outputs`; TL-dependent.                                                 |
| HF forward cache (optional)  | `15`    | `e1db90eb` | Run only for HF-only path where`model_base` still exists; otherwise skip.                           |
| Result filter / alias        | `17`    | `b40f1e25` | Default keeps all samples; edit only if user asks for a subset.                                       |
| Graph init                   | `18`    | `6283ffc5` | Initializes`graph`, model device/dtype fields; edit if running a non-TL model.                      |
| Attribution patch            | `19`    | `d1b04a55` | Patches dequant context for TL attention; TL-dependent.                                               |
| Root node                    | `20`    | `8cbc7c04` | Creates`lm_head`; `root = tnode = add_tnode(...)`; prints `root`.                               |

After Phase 1, `tnode` is the current attribution tree node. Proceed to Phase 2.

### Phase 2: Cluster → Score → Expand (loop body)

For the current `tnode`, execute these steps in order.

**Cell reuse rule (mandatory for Phase 2):** Prefer the notebook's existing reusable cells. Locate them with `notebook_search` if indexes drift, then `notebook_edit_cell` + `notebook_run_cell` on that same cell. Do not keep inserting new cluster/score/expand clones.

| Function | Preferred existing cell (current `test.ipynb`) | How to reuse |
| -------- | --------------------------------------------- | ------------ |
| Cluster (`cluster_heads`) | index `21` (first post-root cluster cell; also appears later as copies — still reuse one chosen cell, typically `21` or the user's currently focused cluster cell) | Edit `threshold=` / figsize if needed, then re-run the **same** cell every iteration. |
| Pattern Score DataFrame | index `25` (first `attn_patterns` + `df` cell; later copies exist — reuse one chosen cell, typically `25` or the user's currently focused score cell) | Edit `attn_patterns` for the goal/dataset, then re-run the **same** cell. This yields **Pattern Score**, not Positive Bound. |
| Attention visualization / Positive Bound | index `24` (`layer`/`heads` arrays + `for` loop over `show_attn`; later copies may exist) | Fill parallel arrays `layer = [...]` and `heads = [...]` with all heads to review (same order), then re-run the **same** cell once. For A-→* goals, **omit** `pos_ids` (default is already A−1 / last prompt token). Use for Positive Bound checks and for the mandatory post-match review of every found head (Phase 3 Step 7). |
| Tree expand (`add_edges` / `add_tnode` / `print_tree`) | reuse an existing expand cell in the current section; only insert if none exists | Edit the node list in place and re-run. |

If the user has a specific cluster/score/visualization cell focused or referenced, reuse **that** cell rather than creating another copy.

**Pattern Score vs Positive Bound (do not conflate):**

| Name | Source cell / API | Meaning |
| ---- | ----------------- | ------- |
| **Pattern Score** | DataFrame cell (`attn_patterns` + `get_head_matching_scores`, e.g. index `25`) | Dataset-level symbolic pattern match (e.g. column `A-->V`) aggregated over samples via `index_map` keys. |
| **Positive Bound** | Visualization cell (`colored_tokens_multi(*show_attn(...))`, e.g. index `24`) | Per-sample attention strength **from** the query token at `pos_ids` **to** other tokens. When `pos_ids` is omitted, the source is A−1 (last prompt token). Read the value at the target token from the visualization (typically the third label row, e.g. `answer✓`). |

- Pattern Score is for screening / ranking heads against symbolic patterns such as `A-->V`.
- Positive Bound is for visual verification of a specific head's attention from a chosen source position to a target token.
- Never treat a DataFrame `A-->V` column value as a Positive Bound, and never treat a `show_attn` Positive Bound as a Pattern Score.

### `A-` indexing and `show_attn` defaults (critical)

In the current Hi-ToM-style datasets, `index_map['A-']` is the answer/logit position and is **one past the end** of the cached prompt attention matrix (OOB if used raw as a query row). Project helpers that consume `A-` already apply `-1` internally. Follow these rules:

1. **Do not patch library / notebook helper code** to “fix” A−. When you need a numeric query index for A− yourself (ad hoc slicing, custom checks), use `index_map['A-'] - 1` (equivalently `answer_indices[-1] - 1` / last prompt token).
2. **`show_attn` default `pos_ids` is already A−1** (last prompt token via `answer_indices - 1`). For goals whose source is A− (e.g. A− → V), call `show_attn` **without** passing `pos_ids`. Do not pass raw `index_map['A-']` — that triggers CUDA index-OOB and can kill the kernel.
3. **Symbolic Pattern Score paths** (`get_head_matching_scores` / `A-->V` etc.) already handle A− correctly via dataset helpers; keep using pattern columns for screening. Only apply the manual `-1` when you touch raw attention rows/indices yourself.
4. Only pass an explicit `pos_ids` when the source is a **non-A−** position (e.g. a numeric token index, `V`, `QK_C`). For those keys, use `index_map` values as stored unless you observe the same OOB pattern.

**Step 2a — Cluster with auto-adjusted threshold:**

Reuse the existing `cluster_heads` cell (see table above). Edit only parameters such as `threshold`, then re-run:

```python
d = tnode.data
d.groups, d.metrics, _, _ = cluster_heads(d.attn_attrs_ds,
    threshold=cluster_threshold, strengths=d.top_heads,
    model_config=selected_model.cfg, figsize=(18, 5), width_ratios=(4, 1),
    bar_height=0.5, leaf_font_size=9)
```

- Start with `cluster_threshold` (default 0.35).
- If `len(d.groups) < 2`: lower `cluster_threshold` by 0.05 and re-run **the same cell**.
- Repeat until ≥2 groups are formed (each group represents a semantically distinct set of attention heads).
- Record the final `cluster_threshold` and group assignments.

**Step 2b — Compute per-head attention pattern scores:**

Reuse the existing DataFrame / `attn_patterns` cell (see table above). That cell is intentionally editable: adapt `attn_patterns` to the user's requested goal and to the keys that actually exist in the dataset's `index_map`, then re-run it. Do **not** assume every dataset has `V->VK_C`, `V->VK_I`, `A-->QK_C`, or `A-->QK_I`. Do **not** open a new cell just to print another `df`.

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

# If the user requested a specific symbolic pattern, keep only that pattern
# when it can be scored from the current dataset.
# Example: attn_patterns = ['A-->V'] for "A- attends to V".
#
# requested_patterns = ['A-->V']
# attn_patterns = [p for p in requested_patterns if p in attn_patterns]

if not attn_patterns:
    missing = {
        pattern: sorted(required_keys - available_keys)
        for pattern, required_keys in pattern_requirements.items()
        if not required_keys <= available_keys
    }
    raise ValueError(f'No attention patterns can be scored from this dataset. Missing keys by pattern: {missing}')

df = pd.DataFrame([(l, h, round(score, 4),
    round(mr(eval_head_lens)(_results, selected_model, l, h, strict=False).item(), 4),
    round(mr(eval_head_lens)(_results, selected_model, l, h, strict=True).item(), 4),
    *[round(ap_scores[p].mean().item(), 4) for p in attn_patterns])
    for (l, h), score in d.top_heads.items()
    if (ap_scores := mr(get_head_matching_scores)(_results, attn_patterns, selected_model, l, h))],
    columns=['layer', 'head', 'score', 'acc', 'acc0'] + attn_patterns)
print(df.to_csv(sep='\t', index=True))
```

This produces a table with columns: `layer`, `head`, `score`, `acc`, `acc0`, plus whichever pattern columns were valid for the current dataset and goal. If no symbolic pattern can be formed from `index_map`, stop and report which expected keys were missing rather than forcing the fixed Hi-ToM pattern list.

**Step 2c — Select candidates and expand the attribution tree:**

Determine candidate nodes using TWO sources:

1. **From clusters (`d.groups`)**: For each cluster, map to a node type and attention pattern based on the cluster's dominant attention pattern (see mapping table below). Construct `Node(l, h, node_type, attn_pattern=pattern)`.
2. **From DataFrame (`df`)**: Select heads where `attn_pattern_score > attn_pattern_threshold` (default 0.3). If too few candidates, lower `attn_pattern_threshold` by 0.05; if too many (>10), raise it by 0.05.

Merge both sources, deduplicate, then expand:

```python
nodes = [Node(l, h, node_type, attn_pattern=pattern) for ...]
for n in nodes: add_edges(graph, n, tnode.data.nodes, tnode.data.attr)
tnode = add_tnode(_results, selected_model, nodes, parent=tnode)
print_tree(root)
```

This produces a new `tnode` — return to Step 2a for the next iteration.

**Open-ended mode — round bookkeeping (required):**

1. Keep a counter `round_i` starting at 1 for the first cluster→score→expand cycle on the root.
2. After each successful `add_tnode` / `print_tree`, increment `round_i` and record: round index, expanded `(layer, head, pattern, node_type)` list, `cluster_threshold`, `attn_pattern_threshold`, and the printed tree snippet.
3. Optional per-round visualization: only if useful for debugging; default is to **skip** mid-loop `show_attn` in open-ended mode and visualize only a small set of strongest heads at the end (or omit viz if the user only asked for the tree).
4. Reuse the **same** cluster / score / expand cells every round — edit parameters in place; do not insert a new cell per round.
5. If Step 2c yields zero candidates even after lowering thresholds within this round, treat it as the **exhausted thresholds** stop (do not spin empty rounds).

**Cluster → node type mapping:**

Map each cluster to a node type and attention pattern based on its dominant pattern from the DataFrame:

| Dominant pattern    | Node type                | Typical cluster |
| ------------------- | ------------------------ | --------------- |
| `A-->V` (high)    | `attn_q` or `attn_k` | Cluster 1, 3    |
| `A-->A-` (high)   | `attn_v`               | Cluster 2       |
| `V->VK_C` (high)  | `attn_k`               | —              |
| `A-->QK_C` (high) | `attn_q`               | —              |

If a cluster's heads show no clear dominant pattern, skip that cluster.

### Phase 3: End Conditions

The loop terminates under: goal-directed soft/hard completion, open-ended completion, or agent-set hard stops. **Finding the first matching head is never by itself a stop** in goal-directed mode.

#### User-set goal (goal-directed search) — match, then keep going upward

When the user specifies a search goal like "find heads where token X attends to token Y":

1. **Extract goal from user request:**

   - `source_position`: the token index where attention comes FROM (e.g., "from token 42", "from 'John'"). If the user names an `index_map` key such as `A-`, `QK_C`, or `VK_I`, use that key (see A− rule below for raw indexing).
   - `target_token`: the token index being attended TO (e.g., "attends to token 80", "to 'Mary'"). If the user names an `index_map` key such as `V`, use that key's per-sample value.
   - Do **not** ask the user for `min_positive_bound`, and do **not** hardcode a fixed cutoff such as `0.3`. The acceptance bar is derived empirically from first-round candidate scores (see Step 4).
2. **Resolve token positions before scoring:**

   Prefer dataset-provided positions over ad hoc token search.

   - If the user names a position already present in `r.index_map[0]` (for example `A-`, `V`, `QK_C`, `QK_I`, `VK_C`, `VK_I`, or `start`), do **not** run tokenization helper cells to rediscover it.
   - For symbolic Pattern Score (`A-->V` etc.), keep using the pattern APIs / `index_map` keys as-is (helpers apply A− adjustments).
   - For **raw** attention indexing when the source is `A-`, use `index_map['A-'] - 1` yourself. Do **not** edit project code to insert that `-1`.
   - For `show_attn` when the source is `A-`, omit `pos_ids` entirely (default is already A−1 / last prompt token).
   - Only run tokenization / token-search helper cells when the user refers to a token string or position that is not already represented in `index_map`.
   - The tokenization helper cells are exploratory and may be modified freely to inspect token positions. They should not change model state, `_results`, `graph`, `root`, or `tnode`.

   Example helper for locating a user-named token when no suitable `index_map` key exists:

   ```python
   def tokenize_plus(model: HookedTransformer, inputs: List[str]):
       tokens = model.to_tokens(inputs, prepend_bos=False, padding_side='left')
       attention_mask = get_attention_mask(model.tokenizer, tokens, True)
       input_lengths = attention_mask.sum(1)
       n_pos = attention_mask.size(1)
       return tokens, attention_mask, input_lengths, n_pos

   sample = random.choice(_results)
   token_ids, attention_mask, input_lengths, n_pos = tokenize_plus(selected_model, sample.prompt)
   tokens = selected_model.tokenizer.convert_ids_to_tokens(token_ids[0])
   for idx, token in enumerate(tokens):
       print(idx, repr(token))
   ```

   If searching for a particular token string, modify the loop as needed:

   ```python
   matches = [(idx, token) for idx, token in enumerate(tokens) if token == 'Ġ' + sample.answers[0]]
   print(matches)
   ```

   When token text repeats, inspect surrounding tokens and, if needed, check multiple `_results` samples. Do not assume one random sample's token index applies to every sample unless the dataset construction guarantees fixed positions.
3. **Check after each Phase 2 iteration** — after Step 2b produces the DataFrame of per-head **Pattern Scores**, evaluate candidates against the goal. Keep Pattern Score and Positive Bound separate.

   **A. Pattern Score screening (symbolic `index_map` goals)** — reuse the DataFrame cell (e.g. index `25`):

   - `A-` → `V`: use the `A-->V` column from `get_head_matching_scores`.
   - `A-` → `A-`: use the `A-->A-` column.
   - `V` → `VK_C` / `VK_I`: use `V->VK_C` / `V->VK_I`.
   - `A-` → `QK_C` / `QK_I`: use `A-->QK_C` / `A-->QK_I`.

   This matters because `index_map['A-']` is OOB as a raw attention query row; Pattern Score helpers already account for that. Prefer Pattern Score for these symbolic goals during screening.

   **B. Positive Bound visualization / verification** — reuse the existing `show_attn` cell (e.g. index `24`). That cell uses parallel arrays `layer` / `heads` and a `for` loop for batch review. Fill the arrays (and only set `pos_ids` when the source is **not** A−), then re-run **once**:

   ```python
   # A- → * 目标：不要传 pos_ids（默认已是 A- - 1 / 最后一位 prompt token）
   sample = _results[0]
   layer = [28, 31]   # 待审阅头的 layer，按顺序
   heads = [0, 28]    # 对应 head，与 layer 一一对齐
   for l, h in zip(layer, heads):
       colored_tokens_multi(*show_attn(
           sample, selected_model, l, h,
           downstreams=root.data.nodes,
           start=_results[0].index_map[0]['start'],
       ))

   # 非 A- 源位置时才显式传 pos_ids（例如某个 numeric index 或 index_map['V']）
   # for l, h in zip(layer, heads):
   #     colored_tokens_multi(*show_attn(
   #         sample, selected_model, l, h,
   #         downstreams=root.data.nodes,
   #         start=_results[0].index_map[0]['start'],
   #         pos_ids=source_position,
   #     ))
   ```

   - Read **Positive Bound** at the target token from this visualization (typically the third label row, e.g. `answer✓`).
   - Use this whenever the user asks to verify attention at a specific token position, or for numeric token goals that are not covered by a symbolic pattern column.
   - Never pass raw `index_map['A-']` as `pos_ids`. If you must index A− outside `show_attn`, subtract 1 yourself; do not modify `vis.py` / notebook helpers for this.
4. **Derive the acceptance bar from the first clustering round (required):**

   After the **first** Phase 2 clustering + scoring pass:

   1. Collect first-round candidate scores using the metric that matches the goal:
      - symbolic pattern goal → Pattern Score from the DataFrame column;
      - numeric / visual position goal → Positive Bound from `show_attn(..., pos_ids=...)` at the target token.
   2. Treat those observed scores as the empirical reference set. Typical ways to set `match_score_floor`:
      - use the best first-round score, or
      - use a high quantile / near-top cluster of first-round scores when several candidates look similarly strong.
   3. Record the reference heads, which metric was used, their scores, and the chosen `match_score_floor` in the final report.
   4. Do **not** invent a preset absolute threshold independent of the observed first-round scores, and do **not** mix Pattern Score with Positive Bound when setting or applying the floor.

   In later Phase 2 iterations, keep searching for heads whose score on that **same metric** is **close to or above** this empirical floor. Prefer heads that meet or exceed the strongest first-round references; also accept heads that are clearly comparable to that first-round top group rather than far weaker outliers.
5. **Decision (match ≠ stop):**

   Keep an accumulated list `matched_heads` across rounds (empty at start).

   - If one or more heads meet or exceed `match_score_floor` (or are clearly comparable to the strongest first-round reference scores): they are **confirmed matches for this round**.
     1. Materialize **this round's** matched heads into the tree (Step 6) so the next attribution step runs from them.
     2. Append them to `matched_heads` (dedupe by `(layer, head)`).
     3. **Do not stop.** Return to Step 2a on the new `tnode` and search for **more** heads that satisfy the **same** goal (same Pattern Score column or same Positive Bound source→target). Prefer newly discovered matches at deeper / earlier layers along the attribution path.
     4. Optional: skip mid-loop visualization; run Step 7 once at the end over **all** accumulated `matched_heads`.
   - If no head this round meets the empirical bar: still run Step 2c with the best exploratory candidates (cluster + near-threshold pattern scores), then iterate Phase 2 again with the same `match_score_floor` (raise it only if later rounds show a clearly stronger reference set).
   - **Stop goal-directed continuation only when:**
     - `round_i >= max_attribution_rounds`, or
     - an **agent-set** condition fires (expansion failure / exhausted thresholds / stagnation), or
     - a later round produces **no** new match and **no** useful exploratory expansion (treat as stagnation).

   After the loop ends, if `matched_heads` is non-empty: finish Step 7 (batch visualize all matches), then report every accumulated match with layer, head, pattern, metric, score, derived floor / references, and which attribution depth it entered. If `matched_heads` is empty: report goal-not-found as before.
6. **Materialize matched heads each time they are found (required for continuation):**

   Whenever a round confirms matches, add them to the tree **before** the next Phase 2 iteration. Do not leave matches only in a table while `tnode` stays at a pre-match node.

   Construct `Node` objects for **this round's** matched heads using the matched attention pattern, connect them to the current `tnode`, create the next tree node, and print the updated tree:

   ```python
   matched_nodes = [Node(layer, head, node_type, attn_pattern=pattern) for ...]
   for n in matched_nodes:
       add_edges(graph, n, tnode.data.nodes, tnode.data.attr)
   tnode = add_tnode(_results, selected_model, matched_nodes, parent=tnode)
   print_tree(root)
   ```

   Choose `node_type` with the same mapping used in Step 2c. If the pattern is ambiguous, prefer the node type that matches the current investigation direction; for `A-->V`, use the same local convention as nearby notebook cells (often `attn_k` when manually expanding from root, or `attn_q` when following query-side clusters). Record the chosen node type in the report.

   If `add_edges` or `add_tnode` fails, stop further continuation and report all matches found so far plus the expansion error. The Attribution Tree section must use the deepest successfully printed tree and note that later matches could not be materialized.
7. **Visualize all accumulated matched heads for user review (required at end):**

   After the goal-directed loop finishes (and preferably after the last successful materialization), reuse the existing visualization cell (index `24`). Do **not** insert a new visualization cell. Do **not** require a full viz pass after every intermediate match round.

   Put **all** accumulated `matched_heads` into parallel arrays `layer` and `heads` (same order), then re-run the cell **once**:

   ```python
   # 指定 head 可视化其注意力（A- → V：省略 pos_ids，默认即为 A- - 1）
   sample = _results[0]
   layer = [28, 31]   # 全部累计匹配头的 layer，按审阅顺序
   heads = [0, 28]    # 对应 head，与 layer 一一对齐
   for l, h in zip(layer, heads):
       colored_tokens_multi(*show_attn(
           sample, selected_model, l, h,
           downstreams=root.data.nodes,
           start=_results[0].index_map[0]['start'],
       ))
   ```

   - Fill `layer` / `heads` with every accumulated match; lengths must match.
   - For A−-sourced goals, **do not** add `pos_ids`. Only pass `pos_ids` for non-A− sources. If `sample` is undefined, use `random.choice(_results)` or an existing sample variable.
   - After materialization, prefer `downstreams=root.data.nodes` (lm_head) when reviewing heads that feed the root; for deeper nodes, `downstreams` may follow the parent context used when those heads were expanded.
   - Overwriting the previous visualization output is expected and preferred.
   - In the final report, note that visualizations for all matched heads were rendered in the notebook for user review. If a visualization run fails for some head, report which heads succeeded and which failed; do not skip this step silently.

#### Open-ended end condition (no user goal — multi-round tree building)

When `attribution_mode = open-ended` (user did not name specific heads or a directed pattern goal):

1. **Do not** derive or apply `match_score_floor` as a stop criterion. Pattern scores still guide **which** heads to expand each round, but finding a strong `A-->V` (or any pattern) is **not** a reason to stop.
2. Each round **must** attempt Step 2c expansion after scoring, unless an agent-set condition already applies.
3. Prefer expanding a **small diverse set** of candidates per round (typically 2–6 heads): merge cluster representatives + DataFrame heads above `attn_pattern_threshold`, dedupe, cap at ~6 to avoid huge `add_tnode` graphs / OOM. Prefer heads that introduce a **new** dominant pattern or a clearly deeper layer than already on the path.
4. Continue Phase 2 until **any** of:
   - `round_i >= max_attribution_rounds` (default 5) — **soft complete**: report the tree built so far as successfully deepened;
   - an **agent-set** condition below fires — **hard stop**.
5. Before the final report in open-ended mode:
   - Capture `print_tree(root)` after the last successful expansion.
   - Optionally batch-visualize the heads added in the **last** round (or the globally strongest heads on the tree) via cell `24` arrays; skip if the user only asked for the tree / attribution structure.
6. Final report must state: mode = open-ended, rounds completed, stop reason (`max_attribution_rounds` / expansion failure / exhausted thresholds / stagnation), and the exact `print_tree(root)` output.

#### Agent-set end condition (automatic stop)

The loop stops automatically when further progress is impossible:

1. **Expansion failure**: `add_tnode` or `add_edges` raises an error (shape mismatch, missing data, attribute error) — the tree cannot be expanded further upward.
2. **Exhausted thresholds**: After lowering `cluster_threshold` to 0.1 and `attn_pattern_threshold` to 0.05, still no candidate heads are produced.
3. **Stagnation**: The current iteration produces the same `tnode.data.top_heads` as the previous iteration (no new heads discovered), or the candidate set after dedupe is empty / identical to the nodes already expanded at this `tnode`.

When any agent-set condition triggers:

- If a user goal was specified and `matched_heads` is non-empty: report all accumulated matches, rounds completed, stop reason, then run end-of-loop Step 7 visualization if not yet done.
- If a user goal was specified but `matched_heads` is empty: report "Goal not found after N iterations" with a summary of the deepest heads discovered.
- If open-ended / no goal was specified: report the full circuit tree discovered so far, with N = rounds completed and the stop reason.
- In all cases, **STOP**. Do not ask the user to manually pick the next heads unless they explicitly requested interactive selection.

## Notebook-Specific Guidance

Key local files:

- `test.ipynb`: main experiment notebook.
- `data_uniform/Hi_ToM_order_1.csv`: current Hi-ToM data source.
- `model_hooks.py`: HF/TL output collection and attention adaptation.
- `attribute.py`: graph, node, and attribution logic.
- `vis.py`: head scoring, clustering, and visualization.

**`A-` / `show_attn` reminder:** `index_map['A-']` is OOB for prompt attention rows. Helpers already `-1` internally. Agent code that touches raw indices must `-1` itself. For A−-sourced `show_attn`, omit `pos_ids` (default = last prompt token = A−1). Do not patch project code for this.

Common model-loading rule:

- If using a local ordinary model directory, pass the absolute path directly to `from_pretrained`.
- Do not assume `cache_dir` plus a HuggingFace repo id will find a ModelScope-style directory.
- If `model.safetensors.index.json` exists, verify the referenced shard files also exist before diagnosing deeper.

## Output Format

When the loop terminates, produce a final report in TWO sections:

### Section 1: Task Result

**If the user specified a search goal** (find heads attending from `source_position` to `target_token`) and heads were found:

```markdown
## 任务结果

✅ **已找到目标注意力头（并继续向上追溯）** — 从 token / `index_map` key `source_position` 关注到 token / `index_map` key `target_token` 的注意力头如下（含多轮累计）：

| 注意力头 (Layer, Head) | 所用指标 | 分数 | 注意力模式 | 节点类型 | 归因层级 |
|------------------------|---------|------|-----------|---------|---------|
| (layer, head) | Pattern Score 或 Positive Bound | 0.XX | A-->V | attn_k | 第 N 层归因 |

> **续归因说明**：首次命中后不会停止；agent 会在同一目标模式/指标下继续向上扩展，直到 `max_attribution_rounds` 或 agent-set 终止条件。上表为全部累计匹配头。
>
> **分数说明（二者不同，勿混用）**：
> - **Pattern Score**：来自 DataFrame cell（`attn_patterns` + `get_head_matching_scores`），例如 `A-->V` 列；用于 symbolic `index_map` 目标的筛选。
> - **Positive Bound**：来自可视化 cell `colored_tokens_multi(*show_attn(...))`；A− 源时省略 `pos_ids`（默认 A−1）。表示该 head 从源位置出发、在目标 token 上的关注强度；用于位置级可视化验证。
>
> **经验门槛**：`match_score_floor` 不是用户预设值，而是首轮后根据**同一指标**的候选头观测分数推导出的门槛；用于判定“是否算匹配”，不是循环停条件。最终报告需写明指标名、门槛、参考头分数、完成轮次与停止原因。
>
> **可视化审阅**：已对全部累计匹配头复用 `show_attn` cell（将 layer/heads 填入数组后一次性 for 循环运行）完成注意力可视化，请在 notebook 中查看最新输出。
```

**If the user specified a goal but NO head matched:**

```markdown
## 任务结果

❌ **未找到匹配的注意力头**

已遍历 N 轮归因。首轮后根据候选头观测分数得到经验门槛 `match_score_floor = 0.XX`（指标: Pattern Score 或 Positive Bound；参考头: …），后续候选头均未接近或超过该门槛。

| 最接近的候选头 | 所用指标 | 分数 | 与经验门槛差距 |
|---------------|---------|------|---------------|
| (layer, head) | Pattern Score | 0.18 | -0.12 |

> 流程因 [扩展失败 / 阈值耗尽 / 归因停滞] 自动结束。
```

**If no goal was specified** (open-ended / general circuit discovery):

```markdown
## 任务结果

✅ **归因完成（开放式多轮）**

- 模式: open-ended
- 共执行 N 轮 cluster → score → expand
- 停止原因: [达到 max_attribution_rounds=K / 扩展失败 / 阈值耗尽 / 归因停滞]
- 模型: …
- 每轮扩展摘要: 第1轮 …; 第2轮 …; …
```

> 未指定具体注意力目标时，agent 会自动多轮完善归因树，而不是在首轮扩展后停止。
```

### Section 2: Attribution Tree

Always output after Section 1:

```markdown
## 归因树

自动发现的归因电路结构（`print_tree(root)` 输出）：
```

L64┐
   └L59 A-->V x1
      └L55 A-->V x1

```

> **如何阅读**：每个节点 `L层号 注意力模式 x节点数` 表示一个或多个注意力头。顶层（L64）为 lm_head 输出层，箭头从上到下表示归因信号向模型输入端方向追溯。`A-->V` 等标记表示该 head 的注意力模式。
```

Use the exact `print_tree(root)` output — do not modify or reformat it. For goal-directed searches, this must be the tree **after all continuation rounds** (every accumulated match materialized), not the tree from only the first match. Also complete Phase 3 Step 7 (batch-visualize all accumulated matched heads via the reused `show_attn` cell) before the final report.

## Do Not

- Do not run optional API/client cells unless the user asks.
- Do not download models unless the user explicitly allows network/downloads.
- Do not clear or overwrite user results unless asked.
- Do not treat the **first** goal match as “任务完成” in **goal-directed** mode; materialize it, then continue upward for more same-pattern matches until `max_attribution_rounds` or an agent-set stop.
- Do not treat a single Phase 2 expand as “归因完成” in **open-ended** mode; keep iterating until `max_attribution_rounds` or an agent-set stop.
- Do not stop open-ended or goal-directed continuation to ask the user which heads to expand next, unless they explicitly requested interactive / manual selection.
- Do not keep executing after the notebook reaches a **user-requested** manual branch that requires choosing heads, thresholds, or circuit nodes.
- Do not treat rich visualization HTML as a failure if the cell completed and produced a render object.
- Do not insert new cells that rewrite functionality already present in the notebook (clustering, pattern-score DataFrame, tree expansion, tokenization helpers, etc.). Edit and re-run the existing cell instead; overwriting that cell's prior output is fine and preferred.
- Do not create a growing trail of near-duplicate Phase 2 cells across iterations; each repeated operation should bounce through the same reusable cell.
- Do not pass raw `index_map['A-']` into `show_attn(..., pos_ids=...)` or other raw attention indexers; omit `pos_ids` for A−-sourced viz, or subtract 1 yourself. Do not rewrite project helpers to insert that `-1`.
- Do not apply `match_score_floor` as a **loop stop** rule (it only decides which heads count as matches). In open-ended mode it is not used at all.
