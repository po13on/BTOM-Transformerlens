---
name: btom
description: >-
  Supervise BTOM-TransformerLens circuit discovery in test.ipynb (load model,
  TL cache, cluster/score/expand attribution, visualize heads). Use whenever the
  user mentions BTOM, test.ipynb, 归因, 电路发现, 归因树, A-->V / A-→V,
  Hi-ToM, TransformerLens hooks/cache, clustering attention heads, Pattern Score,
  position-frontier, Q/K/V pathway attribution, or continuing a GPU notebook
  experiment in this project — even if they never say "skill", "notebook", or
  "btom". Prefer this skill over ad-hoc scripts. Resolve project root via
  BTOM_PROJECT_ROOT or ~/.btom/config.json.
---

# BTOM Notebook Runner

Run `$PROJECT_ROOT/test.ipynb` as a **supervised** experiment: incremental cells, evidence-driven stops, notebook-visible Phase 2.

Details that would bloat this file live under `references/` — **read them when needed** (pointers below).

## Quick examples

| User says | Agent does |
| --------- | ---------- |
| 「加载 Qwen3-8B，跑 Hi-ToM」 | Phase 1 only → report L/H/V + n_samples → **stop** |
| 「找 A-→V 的头」 | Phase 1 → **goal-directed** loop **inside attribution workspace** → accumulate matches → re-run upper-bound viz → report |
| 「完整归因树 / 自动发现电路」 | Phase 1 → **open-ended** Stage0 `A-->V`×{q,k,v} in workspace → frontier expand → role ledger → re-run upper-bound viz → report |
| 「假设数据只有 A-->V，重新归因」 | Seed `Answer`/`AnswerSpan` only → discover readable roles → expand (workspace only) → compare to labeled run |
| 「点一下可视化」 | Re-run the upper-bound `visualize_model_heads(...)` cell; do **not** click the widget for the user |

## First principles

1. Run cells incrementally; never blind “run all” unless asked.
2. Preserve the notebook; explain temporary edits first.
3. **Phase 2 = notebook UI only.** `notebook_insert_cell` + `notebook_run_cell` (pass `notebook_uri` if the tab is open but not “active”). **Forbidden:** Shell / `jupyter_client` / hidden-kernel sandbox for cluster·score·expand.
4. **Attribution workspace (mandatory sandbox).** All cluster / score / expand work happens **only between** two landmark cells in `test.ipynb` (find by source match; indexes drift):
   - **Upper bound** (do not edit/delete): `visualize_model_heads(root, selected_model, _results, sample=_results[0])`
   - **Lower bound** (do not edit/delete): `colored_tokens_multi(*show_attn(random.choice(_results), selected_model, 51, 10, downstreams=tnode.data.nodes, start=_results[0].index_map[0]['start']))#, start=100))`
   - Inside `(upper, lower)`: freely **overwrite** existing cells and/or **insert** new ones for attribution. Prefer keeping the three logical steps visible and readable for the user watching mid-notebook.
   - **Never** append Phase 2 cells at the notebook end, above the upper bound, or below the lower bound. When inserting, place at an index strictly before the lower-bound cell so both landmarks stay put as bookends.
5. **Every Phase 2 round = three logical steps** (never merge cluster+expand into one cell):
   - `# Round N cluster` → only `cluster_heads`
   - `# Round N score` → `d = tnode.data` then Pattern Score table
   - `# Round N expand` → exact `Node(...)` list + `add_edges` / `add_tnode` / `print_tree`
     Reuse/overwrite workspace cells when that keeps the mid-notebook trail clearer; insert extra cells inside the sandbox when a round needs more space.
6. OOM / missing weights / CUDA issues → stop unless the fix is obvious and safe.
7. Scope to the **newest** user ask (load-only ≠ full attribution).
8. Modes:
   - **goal-directed** — named pattern/positions; match → materialize → **keep going upward** for more same-goal heads.
   - **open-ended** — no named goal; build a **role-complete** position-frontier tree until stop.
9. Stage 0 exits: **multi-pathway union** — each selected `(L,H)` gets `attn_q`+`attn_k`+`attn_v`. Robustness = same pathway across exits, **not** swap-q/k-then-intersect.
10. `print_tree` often **collapses** mixed patterns on a depth; truth is the **role ledger** / on-tree `(pattern, type)` list. Always report Section 3 for complete trees.

## Project location

1. `BTOM_PROJECT_ROOT` env, else `~/.btom/config.json` → `project_root`, else ask.
2. Notebook: `$PROJECT_ROOT/test.ipynb`. Prefer free GPU in `CUDA_VISIBLE_DEVICES` before Phase 1 if the default device is busy.

```bash
mkdir -p ~/.btom && echo '{"project_root": "/absolute/path/to/BTOM-Transformerlens"}' > ~/.btom/config.json
```

## Parameters (short)

**User-set (do not invent):** `model_name` (required), `dataset_path` (default Hi-ToM csv), `use_4bit=true`, `use_transformerlens=true`, optional `source_position` / `target_token`, `attribution_mode` (open-ended if no goal).

**Agent-adjustable:** `n_samples`, `cluster_threshold≈0.35`, `attn_pattern_threshold≈0.3`, `max_attribution_rounds≈5`, goal-directed `match_score_floor` (derived from first-round scores — match bar, **not** loop stop).

Cell variable mapping and Phase 1 index↔id table → [references/phase1.md](references/phase1.md).

## Glossary (Hi-ToM `index_map`)

| Key | Meaning |
| --- | ------- |
| `A-` | Answer / logit position (often OOB as raw attn row; use helpers or `Answer = answer_indices-1`) |
| `V` | Answer **value** in story (often location) |
| `VK_C` / `VK_I` | Story **Character** / **Item** near the V event |
| `QK_C` / `QK_I` | Question **Character** / **Item** |

**C = Character, I = Item** (not container). Location/container ≈ `V`.

Readable aliases when discovering without labels: `Answer`, `AnswerSpan`, `Story.Name`, `Story.Object`, `Question.Name`, `Question.Object`.

## Circuit discovery (open-ended default)

### Labeled data (full `index_map`)

1. **Stage 0:** score `A-->V`; take top **2** exits; hang **each** × `{attn_k, attn_q, attn_v}` (~6 nodes). Frontiers `{A-, V}`.
2. **Stage 1+:** cluster/score/expand inside the attribution workspace; cover **both** A−-side and V-side when candidates exist.
3. Prefer heads that open **new roles/frontiers**; cap ~2–6 nodes/round.
4. Score **self / previous-token baselines** every round (`A-->A-` and especially **`V->V` / `AnswerSpan->AnswerSpan`**). Do **not** force-label a head `A-->A-` when `V->V` is equal or stronger — that mis-tags previous-token/self heads (e.g. (20,9)).
5. **Weak patterns:** if a column is near-zero across candidates (often `A-->QK_I` / `Answer->Question.Object` when the story object is unique), **do not invent heads** for it; note the absence. Object binding often lives on the **V / AnswerSpan** side (`V->VK_I`), not on Answer→Question.Object.
6. Stop: `max_attribution_rounds`, expansion failure (e.g. only L0 left), exhausted thresholds, or stagnation.

### Unlabeled / seed-only data

When the user says the dataset “only has A-->V” (or other `index_map` keys are missing/untrusted):

1. Seed positions: **`Answer`** (last prompt / logit−1) + **`AnswerSpan`** (answer string match in context). Stage 0 still uses `A-->V` (or `Answer->AnswerSpan`).
2. After each expand, **discover** new edges from frontier queries: attention top-k → filter stopwords → **role-align across samples** (`Story.Name`, not exact token “Jacob”) → inject into a pos registry with **human-readable names**.
3. Re-score with seed + discovered edges; expand; iterate.
4. Report both code patterns and readable names in the role ledger.

Full unlabeled protocol → [references/unlabeled-discovery.md](references/unlabeled-discovery.md).

### Role catalog (Section 3)

| Pattern + type | Role |
| -------------- | ---- |
| `A-->V` + k/q/v | Answer-key enabler / query former / value reader |
| `A-->A-` or `Answer->Answer` + v | Final-position / Answer assembler |
| `V->V` or `AnswerSpan->AnswerSpan` + v | AnswerSpan-self (do not call this A-->A- if V-side dominates) |
| `V->VK_*` or `AnswerSpan->Story.*` + k | Character/Item → answer binder |
| `A-->QK_*` or `Answer->Question.*` + q | Query Character/Item aligner |

## Phase 1 / 2 / 3 (where to read)

| Topic | File |
| ----- | ---- |
| Phase 1 cell table, CUDA, model load | [references/phase1.md](references/phase1.md) |
| Cluster / score code, Pattern Score vs Positive Bound, A− indexing, expand selection | [references/phase2.md](references/phase2.md) |
| Goal-directed match floor, continuation, viz | [references/goal-directed.md](references/goal-directed.md) |
| Final report templates (中/EN) | [references/report-format.md](references/report-format.md) |

**Hard loop rules (keep in mind without opening refs):**

- Pattern Score screens; Positive Bound is for click-review in the upper-bound `visualize_model_heads` — never conflate.
- Never pass raw `index_map['A-']` as `pos_ids`.
- Phase 2 cells stay inside the attribution workspace; end review = re-run the **existing** upper-bound `visualize_model_heads` cell (do not append a duplicate at the notebook end). The lower-bound `colored_tokens_multi(...)` is a landmark only — not the preferred end-review pattern.
- Goal-directed: first match ≠ stop. Open-ended: first `A-->V` expand ≠ done.

## Do not

- Phase 2 outside the notebook; merge cluster+expand into one cell.
- Append cluster/score/expand (or duplicate viz) **below the lower bound** or anywhere outside `(upper, lower)`.
- Edit, delete, or move the two landmark boundary cells.
- Stage 0 only-k on some exits and only-q on others; swap-then-intersect “robustness”.
- Skip Section 3 on complete-tree asks; trust collapsed `print_tree` labels over the role ledger.
- Force-hang near-zero patterns; mislabel `V->V` hubs as `A-->A-` without comparing columns.
- Download models / clear user results / run optional API cells unless asked.
