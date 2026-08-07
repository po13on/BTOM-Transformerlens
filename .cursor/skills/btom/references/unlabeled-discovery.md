# Unlabeled / seed-only discovery

Use when the user says the dataset “only has A-->V”, other `index_map` keys are missing/untrusted, or asks to **auto-discover** readable roles without relying on full Hi-ToM labels.

**Why this exists:** exact token strings (e.g. “Jacob”) do **not** align across samples. Discovery must use **role-based** names (`Story.Name`, `Question.Object`) that resolve per-sample, then inject into a position registry for Pattern Score.

## Seed (Stage 0)

1. Positions:
   - **`Answer`** — last prompt / logit−1 (same role as labeled `A-`)
   - **`AnswerSpan`** — answer string match in context (same role as labeled `V`)
2. Stage 0 scoring still uses `A-->V` or `Answer→AnswerSpan`.
3. Hang top ~2 exits × `{attn_q, attn_k, attn_v}` (same multi-pathway union as labeled runs).

Do **not** invent Character/Item keys before evidence appears.

## After each expand — discover new edges

From each active frontier query position:

1. Read attention top-k destinations across samples.
2. Filter stopwords / punctuation / boilerplate.
3. **Role-align across samples** — cluster by narrative role, not by surface string:
   - Character in story near AnswerSpan event → `Story.Name`
   - Item in that event → `Story.Object`
   - Character / item named in the question → `Question.Name` / `Question.Object`
4. Inject discovered spans into a per-sample pos registry with **human-readable** names.
5. Re-score Pattern Score with seed + discovered edges; expand; iterate.

If `Node.set_pos_ids` / `get_ranges` only allow a limited key set, patch **in notebook cells** for this session (or extend helpers carefully) so dynamic role names can be used as `attn_pattern` / pos ids. Prefer readable edge labels in expand cells and the role ledger.

## Scoring discipline

Always include baselines:

| Edge | Meaning |
| ---- | ------- |
| `Answer→Answer` / `A-->A-` | Final-position self |
| `AnswerSpan→AnswerSpan` / `V->V` | Answer-value self — often the true label for previous-token hubs |

Compare columns before naming a head. Example: Pattern Score `A-->A-` may measure last-token self-attn while the head is actually **`V→V` / AnswerSpan→AnswerSpan** (e.g. (20,9)).

## Weak / near-zero patterns

- `Answer→Question.Object` can be ~0 while `Answer→Question.Name` is strong (~0.27) when the story object is unique — Answer may not need Q.Object.
- Object binding often shows as **`AnswerSpan→Story.Object`**, not Answer→Question.Object.
- Skip inventing heads for near-zero columns; note the absence in the report.

## Report

- Role ledger: both code-style patterns (`A-->V`) and readable names (`AnswerSpan→Story.Name`).
- Note seeded vs discovered frontiers.
- Compare core heads to a labeled run when the user asked for a re-attribution under seed-only assumptions — expect overlap on exits, possible differences on mid-tree hubs.
