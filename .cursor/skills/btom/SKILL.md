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
6. OOM / missing weights / CUDA issues → stop unless the fix is obvious and safe. **Qwen attribution is 4bit.** This repo’s hooks / dequant / TL path are already adapted for Qwen 4bit IG. Default `use_4bit=true`; do **not** treat “32B must be fp16” as a reason to skip attribution or to keep `n_samples` tiny. Do **not** `merge_and_unload` on a 4bit base — CPU fp16 merge first if a merged dir is required, then load that merged dir in 4bit+TL.
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

**User-set (do not invent):** `model_name` (required), `dataset_path` (default Hi-ToM csv), `use_4bit=true` (**required for Qwen attribution**; code is 4bit-adapted), `use_transformerlens=true`, optional `source_position` / `target_token`, `attribution_mode` (open-ended if no goal).

**Agent-adjustable:** `n_samples`, `cluster_threshold≈0.35`, `attn_pattern_threshold≈0.3`, goal-directed `match_score_floor` (derived from first-round scores — match bar, **not** loop stop). **Do not use a round cap.** Stop only on evidence (see “How to attribute”).

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

## How to attribute (learned; keep updating)

This section is the live method. After **every** attribution session, append what went wrong and the rule that would have prevented it. Do not reset these rules to a round budget.

### Stop on evidence, not a round cap

**Forbidden:** `max_attribution_rounds≈5` (or any preset “K rounds then done”). That cut `scat_targeted` at L47 while residual was still finite and Demo edges had not surfaced.

Continue while `attr` is finite **and** the next score table still has a **new** cross-span role, a stronger copy of a required role, or a frontier that the required circuit still needs. Stop only when one of these is true:

1. **Role-complete + stagnant.** Required edges are already on the tree, and the new top heads are only self/previous-token baselines (`Final->Final`, `Copy->Copy`, `A-->A-`, `V->V`) or near-zero cross-span.
2. **Genuine expansion failure.** After keeping only `attr.max() > 0` heads, `get_downstreams()` is empty (no positive residual edge).
3. **Exhausted thresholds.** `cluster_threshold≤0.1` and `attn_pattern_threshold≤0.05` still empty.
4. **Health failure.** `isnan(attr).sum() > 0` after an expand — do **not** hang the NaN top-k. Shrink the batch / stay in fp32 and retry; never treat `(0,0)(0,1)(0,2)` as “only L0 left”.

A complete `scat_targeted` tree must hang all six families:

1. **`Final->Copy`**
2. **`Final->QLast`**
3. **`QLast->WinDigit`**
4. **`Final->DemoOutN` / `Final->DemoOut1`**
5. **`QLast->DemoQLast`**
6. **`QLast->DemoOutN`**

Weak Demo columns at L47 are not absence — hang **k** and keep going; Final→DemoOut often jumps to 0.1–0.4 by L41–44. **`QLast->DemoOutN` is a weak needle** (~0.01 on n=8, **0.022** on n=40 at (20,36)) — still hang the best modest-residual candidate; do not wait for 0.3. Pattern Score is a single-cell pin; Demo heads usually sweep the whole last-example Output span.

**Do not keep `INTERP_N=8`.** 4bit + CPU-pinned `r.outputs` is specifically so IG can average more prompts. n=8 made Final→QLast look like 0.04–0.06 and DemoOutN like a 0.01 fluke. n=40 and **n=100** (fwd 108, keep 100) both close the six families in 4 expands. n=100: DemoOut1 **0.576** at (41,11), DemoQLast **0.243**, DemoOutN **0.103**, QLast→DemoOutN **0.029** at (20,36). Stage-0 Pattern Score winner (58,60) still has **negative** residual — hang only `attr.max()>0`. After QLast-k, `|attr|` top-16 again hits 200–550; keep using the modest band.

### Fewest expands (scat_targeted, 2026-09-11)

Do **not** pile more Copy-k after Stage 0. On n=8 the shortest complete tree was **5 successful expands**. On **n=40** the same circuit closed in **4** (DemoOut-k and DemoQLast-q in one modest-band round):

| Round | Hang | Why this is the cheap move |
| ----- | ---- | -------------------------- |
| 1 | top-2 `Final->Copy` × `{q,k,v}` | Write-head residual; Stage 0 union |
| 2 | `Final->QLast` **k** immediately (n=40: 0.21–0.33; n=8 looked like 0.04) + at most one Copy-k + `QLast->WinDigit` **q** | Opens QLast; WinDigit can appear the same round |
| 3 | modest-band `Final->DemoOut1` **k** + `Final->DemoOutN` **k** + `QLast->DemoQLast` **q** (n=40 can do all three here) | DemoOut lives at resid ≈0.05–25, not `|attr|` top-16 |
| 4 | best modest `QLast->DemoOutN` **q** | Completes the sixth family; n=40 ≈0.02, n=8 ≈0.01 |

Failed shortcuts that *increase* total expands: hanging `|attr|` bombs after QLast opens (NaN, must rebuild/retry); hanging 4 nodes + 8 long caches in one `add_tnode` (CUDA OOM). After QLast is open, prefer **1–2 nodes/round** and `get_attn_weights_cache().clear()` + `empty_cache()` first.

### Health checks (every expand)

Print `attr nan`, `max`, `minL`. Residual einsum must be **fp32**. `get_top_heads`: `nan_to_num`, **attn columns only** (`h < H`), never MLP. Stage 0 may hang 2 exits × `{q,k,v}`. After that: ≤4 nodes/round **only while resid max is modest**; once QLast is open or `attr.max()` is already tens–hundreds, hang **1–2**. Only hang if `attr[l,h].max() > 0` **and** `get_downstreams()` is non-empty (`add_edges` requires score > 0; `abs()` ranking will propose large **negative** heads that intern with no edge).

**After QLast opens, do not trust `|attr|` top-16.** Those slots are IG bombs (resid 100–400). Hunting missing Demo/QLast roles: scan heads whose **positive** residual is in **(0.05, 25)** and pick by Pattern Score. Diagnostic loops must not reuse names `L` / `H` — they clobber the model width.

**`attn_weights_ds` / `attn_attrs_ds` are per-sample lists, not dense tensors** (Option B, 2026-09-11: `get_attn_attrs_on_dataset` returns `[tensor_per_sample]`, normalized per sample, no `_cat_pad_seq`). `compute_js_matrix` / `compute_cosine_matrix` / `visualize_group_patterns` accept both lists and legacy dense tensors; notebook cells must not call `.shape` / `.mean()` on dict values directly. Verified on n=100: identical tree/ledger/scores vs padded run; JS distances ~1–2% higher (undiluted), same groups.

### q / k / v by intended frontier

| Intent | Type | What happens |
| ------ | ---- | ------------ |
| Stay at the current query position and ask “what else does this q look at?” | `attn_q` | Frontier stays |
| Open the destination as the next residual site | `attn_k` (or `v`) | Frontier jumps |
| Stage 0 exits | **each** × `{q,k,v}` | Union, not swap-then-intersect |

`scat_targeted` mapping:

| Pattern | Hang | Why |
| ------- | ---- | --- |
| `Final->Copy` | Stage 0 q+k+v; later **k** to walk Copy | Write first digit from extracted block |
| `Final->QLast` | **k** | Open Test Query last digit |
| `Final->DemoOutN` / `DemoOut1` | **k** | Open last-example Output; do not wait for the column to exceed 0.3 |
| `QLast->WinDigit` | **q** | Query pointer into winner grid |
| `QLast->DemoQLast` | **q** | Align Test Query with last demo Query |
| `QLast->DemoOutN` | **q** | Weak needle from QLast into last-example Output; hang best modest-resid hit |

Do **not** hang `Final->QLast` as q if the goal is to read the demo rule at QLast — q stays on Final and never sees `QLast->Demo*`. Do **not** read `Final->DemoOut*` as “this head located test Copy”.

### Cross-dataset playbook (2026-09-17–18)

Lessons from `scat_targeted` DemoOut↔Demo2Out↔Demo1Out. Apply the **method**, not the layer ids.

**1. Name the rules before naming heads.** ICL tasks usually split:

| Kind | What it does | Typical edges (any dataset) |
| ---- | ------------ | --------------------------- |
| **Match / query analog** | Current query → corresponding demo query; query → evidence in the chosen structure | `TestQuery→DemoQuery`, `Query→WinCell` / `Query→StoryCue` |
| **Answer / schema** | Write the output; demo outputs as templates | `Final→Copy` / `Answer→AnswerSpan`; `Final→LastDemoOut` |
| **Corresponding-span hop** | Same role, previous example | `LastOut→PrevOut`, `PrevOut→PrevPrevOut` (first/last token of that span) |

Do not hunt “the rule head” as one node. Pointer, write, and previous-example hop are different frontiers.

**2. Pattern Score is site-independent; residual is not.** Score any `(L,H)` on any named edge without a tree. `attr[L,H]` is only about the **current residual site**. A GOAL table built on the wrong frontier (small, self-dominated, or empty modest pool) is not evidence the edge is absent — and must not be reported as the hop.

**3. Open the source span of the target edge first.** For hop `P→Q`: hang `*→P` as **k** so residual sits on `P`, then score `P→Q`. Hang that hit as **q** to stay on `P`, or **k** to walk to `Q` for the next hop. If you first hang some other **k** that leaves `P` (e.g. `Final→QLast` before `Final→LastDemoOut`), `Final→LastDemoOut` resid often goes **negative**. That is absence-at-this-site. Recover with another path that hangs the needed **k** while the write-site residual still exists (right after Stage 0, or stay-`q` on Final).

**4. Multi-path when one expand order kills a residual.** Rebuild root on the **same** samples; vary only hang order. Scan a hop only after its source span is interned with `resid>0`. Peek the intended `(L,H)` after every expand. Rewrite the report after each path.

**5. Shrink `n` only after a known-head compare.** Forward + CPU-pinned caches is not where n=100 died; `get_attn_attrs` / full `[seq,seq]` at `k>0` is. `add_tnode(..., k=0)` skips attn maps (residual IG still runs). Before trusting a smaller `n`, rescore the previous run’s **core** edges: most within ~35% relative and still above half the old score. Mix lengths; do not lock one `n_tokens`.

**6. Do not split a Stage-0 pair across expands.** `attribute_residual` only fills layers `< max(hung_layer)`. Hang an earlier write-head (e.g. L53) then index a later one (L58) from that child → **OOB**. Two Final-side write-heads must be hung from the **same parent** (usually root).

**7. Filters for ICL hops.** Require `goal > adjacent/later-example control` and `goal > self`. Example: hunting `LastOut→PrevOut`, drop heads that prefer `LastOut→PrevPrevOut` or `LastOut→LastOut`. A 0.04 hit with self>goal on the wrong frontier is a false positive; a 0.03 needle that beats controls **on the right frontier** can be real.

**8. One induction head often implements the whole corresponding-span chain.** A head that does last-out→prev-out frequently also does prev-out→prev-prev (same role, first or last token). Score those hung hop-heads on the next corresponding edge **before** a modest scan. Empty modest band after a weak-`k` open (`resid` ~0.02) means the next residual is outside `(0.005, 0.15)`, not “no hop” — print `attr` max/min and read the Pattern Score on the same heads.

**9. “Which example understands the rule” ≠ “which example Final attends”.** At write time, **match** often analogizes to the **last** demo query; **answer schema** is often summarized into the **last** demo output (which Final reads), while earlier outputs feed that summary via hops. One demo can *specify* both rules; the second demo *closes* induction (two corresponding spans); the last demo is the *readout pointer*. Mid-prompt residual at demo-*i* is a different experiment than Final-only attribution.

**10. Long GPU jobs.** Notebook when the user is watching cells; tmux + a status/report file when SSH drops. Do not overwrite a finished hop report when starting the next hop — new filename.

### Naming (this dataset)

Do **not** reuse Hi-ToM `A-` / `V` / `QK_I` as pattern names. New keys must **not** start with `A`/`V`/`QK_`. Prefer exact-key lookup. `Final` is `seq_len-1` (here `'\n\n'` after `</think>`), not the first Test Output digit. **`DemoOut1-`** is the token immediately before last-demo Output's first digit (same relative slot as Final). It is stored as an exact index; `get_ranges_tom` must **not** extra-shift exact keys that end with `-`. Unstored `Span-` still means `Span` minus one.

### Session loop

1. If the kernel is alive and residual is finite, **continue the current tree** — do not rebuild from Stage 0.
2. After each session: write the mistake + the better rule into **this section**, then keep going until the required circuit is on the tree and the next layer is stagnant.
3. `print_tree` collapses mixed patterns; **role ledger** is the truth.
4. Mix sequence lengths. Do not filter to one `n_tokens`.

### Mistakes already burned (do not repeat)

- fp16 residual → NaN → fake L0 `(0,0)(0,1)(0,2)`.
- One expand with 6 mixed q/k nodes down to L47; Demo frontier never opened.
- Stopping because “4–5 rounds is the skill default” while `nan=0` and minL was still ~37–47.
- Treating a near-zero `Final->DemoOutN` column as “no such head”.
- Ranking with `abs()` then hanging negative-residual heads (`IndexError` / empty intern).
- Leaving a failed `add_tnode` child on the tree and reading `print_tree` as gospel.
- After the required `scat_targeted` edges are on the tree, still hanging weaker copies (`Final->QLast≈0.03`, `QLast->DemoQLast≈0.07`) when unused `index_map` edges (`WinDigit->Copy`, `QLast->CandDigits`, `DemoQLast->DemoOutN`, …) are ~0 and self-attn is 0.9+. Residual was already ~1000; another expand would recreate the NaN / fake-L0 failure. **Before declaring complete:** re-score unused role keys on the current top heads; if they are near-zero and only baselines remain, stop.
- After opening QLast, hanging `|attr|` top-16 (resid 100–400, e.g. (51,20)/(51,16)) → **attr nan**. The heads that actually carry DemoOut / DemoQLast sat in the **modest positive residual band (≈0.05–25)**.
- One `add_tnode` of 4 modest nodes on 8 mixed-length (~550) caches → **CUDA OOM**. Edges may already be interned; `tnode` / `print_tree` can grow a ghost sibling. Retry 1–2 nodes after clearing the attn-weight cache. Ledger, not `print_tree`, is the hung-role truth.
- Treating `QLast->DemoOutN≈0.01` as “no such head” after the other five families are on the tree. It is a weak single-cell pin; hang the best modest-resid q and stop if unused roles + self-attn say stagnant.
- Splitting Stage-0 Copy into two expands `(53,19)` then `(58,59)` → `IndexError` (`attr` width stops at 53). Hang both Final-side write-heads from the same parent.
- After `QLast-k` then `WinDigit-q`, scanning `DemoOut→Demo2Out` on that frontier → 0.04 hits with self>goal. DemoOut-k resid was already negative. Those are not the hop. Open `Final→DemoOut` **k** while Final residual still exists, then scan.
- `pick_hits` that only rejects `self≥0.5`. A 0.045 GOAL with self=0.078 must lose. Always `goal > self` and `goal > later/adjacent control`.
- Empty modest pool after opening Demo2 with a weak **k**, then concluding “no `Demo2Out→Demo1Out`”. The same hop-heads already score 0.37–0.54 on that edge. Score hung heads on the next corresponding span first.
- Treating `DemoOut1-` (the `Output:\n` token, Final-analog) as what Final reads. n=50: (41,11) `Final→DemoOut1=0.560` vs `Final→DemoOut1-=0.018`. Same-slot ICL from Final is **earlier** demos’ pre-write: (37,1) `Final→Demo2Out1-=0.495`. Content hops stay on digits (`DemoOut1-→Demo2Out1=0.437`), not pre-write-to-pre-write.
- Concluding `DemoOutN→Demo2OutN` tops out at (26,54)≈0.11 because that was the modest-band hit. A full-head Pattern Score sweep finds the same corresponding-cell hop at **0.61 / 0.57 / 0.48** on (45,33) / (41,14) / (42,51); those heads also do `DemoOut1→Demo2Out1`. `Final→DemoOutN` still has no sharp high pin (best (48,16)≈0.15, and that head’s mass is on the whole last Output span).

## Circuit discovery (open-ended default)

### Labeled data (full `index_map`)

1. **Stage 0:** score `A-->V`; take top **2** exits; hang **each** × `{attn_k, attn_q, attn_v}` (~6 nodes). Frontiers `{A-, V}`.
2. **Stage 1+:** cluster/score/expand inside the attribution workspace; cover **both** A−-side and V-side when candidates exist.
3. Prefer heads that open **new roles/frontiers**; cap ~2–6 nodes/round.
4. Score **self / previous-token baselines** every round (`A-->A-` and especially **`V->V` / `AnswerSpan->AnswerSpan`**). Do **not** force-label a head `A-->A-` when `V->V` is equal or stronger — that mis-tags previous-token/self heads (e.g. (20,9)).
5. **Weak patterns:** if a column is near-zero across candidates (often `A-->QK_I` / `Answer->Question.Object` when the story object is unique), **do not invent heads** for it; note the absence. Object binding often lives on the **V / AnswerSpan** side (`V->VK_I`), not on Answer→Question.Object.
6. Stop only on evidence — **never** a preset round budget such as 5. See “How to attribute”.

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
- ICL hops: open the **source** span (`k`) before scoring `P→Q`; `goal > self` and `goal > later/adjacent control`; same induction head may cover the whole corresponding-span chain (see “Cross-dataset playbook”).

## Do not

- Phase 2 outside the notebook; merge cluster+expand into one cell.
- Append cluster/score/expand (or duplicate viz) **below the lower bound** or anywhere outside `(upper, lower)`.
- Edit, delete, or move the two landmark boundary cells.
- Stage 0 only-k on some exits and only-q on others; swap-then-intersect “robustness”.
- Skip Section 3 on complete-tree asks; trust collapsed `print_tree` labels over the role ledger.
- Force-hang near-zero patterns; mislabel `V->V` hubs as `A-->A-` without comparing columns.
- Report a GOAL table from the wrong residual site; split Stage-0 later-layer heads onto an earlier-layer child; shrink `n` without rescoring the previous core edges.
- Download models / clear user results / run optional API cells unless asked.
- Switch Qwen attribution off 4bit, or refuse larger `n_samples` on the grounds that 32B IG needs fp16. 4bit Qwen attribution is supported.
