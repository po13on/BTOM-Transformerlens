# Goal-directed mode

Use when the user names a directed goal (`找 A-→V`, `from token X to Y`, symbolic pattern).

**Core rule:** first match ≠ stop. Materialize matches, then keep attributing **upward** for more heads with the **same** goal metric until a hard stop.

## Extract goal

- `source_position` — FROM (token index or `index_map` key like `A-`)
- `target_token` — TO (token index or key like `V`)
- Do **not** ask for `min_positive_bound` or hardcode `0.3` as acceptance. Derive `match_score_floor` from first-round scores.

## Resolve positions

Prefer `index_map` keys over ad-hoc search.

- Symbolic Pattern Score (`A-->V` etc.): use pattern APIs / keys as-is (helpers handle A−).
- Raw attention when source is `A-`: use `index_map['A-'] - 1` yourself; do not edit project helpers.
- End review: `visualize_model_heads` (defaults handle A−). Never pass raw `A-` as `pos_ids`.
- Only run tokenization helpers when the user names a string/position **not** in `index_map`. Those cells must not mutate `_results` / `graph` / `root` / `tnode`.

```python
def tokenize_plus(model, inputs):
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

When a token repeats, check surrounding context and multiple samples — do not assume one sample's index applies to all.

## Screen after each Phase 2 score cell

Prefer Pattern Score columns:

| Goal | Column |
| ---- | ------ |
| A− → V | `A-->V` |
| A− → A− | `A-->A-` |
| V → VK_* | `V->VK_C` / `V->VK_I` |
| A− → QK_* | `A-->QK_C` / `A-->QK_I` |

Defer Positive Bound to end-of-loop `visualize_model_heads`. Do not batch `colored_tokens_multi` for-loops.

## Derive `match_score_floor` (first round only)

After first cluster+score:

1. Collect first-round candidate scores on the **goal metric**.
2. Set floor from best score or a high quantile / near-top cluster.
3. Record reference heads, metric, scores, and floor in the final report.
4. Do not invent a preset absolute threshold; do not mix Pattern Score with Positive Bound when setting the floor.

Later rounds: accept heads at/near this floor on the **same** metric. Floor decides “is match”, **not** “stop loop”.

## Decision each round

Keep `matched_heads` across rounds.

**If matches this round:**

1. Materialize into tree (expand cell) before next iteration.
2. Append to `matched_heads` (dedupe `(layer, head)`).
3. Continue Phase 2 on new `tnode` for more same-goal heads (prefer earlier layers).
4. Skip mid-loop viz.

**If no match:** still expand exploratory candidates (cluster + near-threshold), keep same floor (raise only if a clearly stronger reference set appears).

**Stop only when:**

- `round_i >= max_attribution_rounds`, or
- agent-set (expansion failure / exhausted thresholds / stagnation), or
- no new match **and** no useful exploratory expansion.

Then: if `matched_heads` non-empty → viz once → report all matches with depth. If empty → goal-not-found.

## Materialize

```python
matched_nodes = [Node(layer, head, node_type, attn_pattern=pattern) for ...]
for n in matched_nodes:
    add_edges(graph, n, tnode.data.nodes, tnode.data.attr)
tnode = add_tnode(_results, selected_model, matched_nodes, parent=tnode)
print_tree(root)
```

Choose `node_type` with the same mapping as open-ended Step 2c. If `add_edges`/`add_tnode` fails, stop continuation and report matches so far + error.

## End visualization (once)

```python
ui = visualize_model_heads(root, selected_model, _results, sample=_results[0])
```

- Insert + run **once** after the loop. Widget is for the **user** to click; agent does not click.
- Do not fill `layer`/`heads` arrays or insert `colored_tokens_multi` loops.
- Note in the report that viz ran (or report the error if it failed).

## Switching modes

- Ambiguous request → default **open-ended**; do not ask before starting.
- Scope to newest ask only (load-only → stop after Phase 1).
- Later “找 A-→V” → switch to goal-directed from that point; existing tree stays valid.
- When unsure whether to attribute: finish the clear task, report, then ask — never silently run the whole notebook.
