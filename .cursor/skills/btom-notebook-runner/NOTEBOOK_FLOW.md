# test.ipynb Execution Map

This file summarizes the current execution flow of `/home/hushengchun/project/BTOM-transformerlens/test.ipynb`.

## Stage 1: Initialization

Typical cells:
- `%autoreload`
- imports from `common_utils`, `llm`, `min_arc`, `model_hooks`, `attribute`, `vis`, `seeds.common`
- torch / transformers imports
- `bnbconfig = BitsAndBytesConfig(...)`

Success signs:
- No import traceback.
- `bnbconfig` exists.

Stop signs:
- Missing local project module.
- Missing critical package such as `torch`, `transformers`, `transformer_lens`, `bitsandbytes`.

## Stage 2: HF Model Loading

Typical variables:
- `model_name`
- `model_base`
- `tokenizer`
- `L, H, V`

Success signs:
- `AutoModelForCausalLM.from_pretrained(...)` completes.
- `tokenizer` loads from the same path.
- `model_base.tokenizer = tokenizer`.
- `L/H/V` match model config.

Common blocker:
- `cache_dir` is not the same as a local model path.
- For ordinary local directories, use the absolute model directory as `model_name`.
- If only `model.safetensors.index.json` exists but shard files are missing, stop and report missing weights.

## Stage 3: TransformerLens Loading

Typical cell:
- `HookedTransformer.from_pretrained(model_name, hf_model=model_base, ...)`

Success signs:
- Output like `Loaded pretrained model ... into HookedTransformer`.
- `model.cfg.n_layers`, `model.cfg.n_heads`, `model.cfg.d_vocab` are available.

Stop signs:
- `model_base` was deleted before TL conversion.
- TL does not support the architecture or local adapter.
- CUDA OOM.

## Stage 4: Hi-ToM Data Loading

Typical variables:
- `hi_tom_path`
- `df_tom`
- `results`
- `_results`
- `candidate_ids`
- `labels`
- `index_map`

Success signs:
- CSV loads.
- `index` is parsed with `ast.literal_eval`.
- `results` contains `Result` objects.
- `_results = results`.

Checks:
- Current `Hi_ToM_order_1.csv` is multiline CSV with about 100 samples.
- Do not infer sample count from text line count.

## Stage 5: TransformerLens Forward Cache

Typical variables:
- `names_filter`
- `output_logits`
- `cache`
- `r.outputs`
- `r.responses`
- `r.is_corrects`
- `r.logprobs`

Success signs:
- tqdm reaches 100%.
- Every `Result` has `outputs`.
- `responses`, `is_corrects`, and `logprobs` are filled.

Stop signs:
- Missing hook names.
- CUDA OOM.
- Shape mismatch in `get_outputs_from_cache`.
- Model/tokenizer path mismatch changes answer token indices unexpectedly.

## Stage 6: Optional HF Forward Cache

This branch requires `model_base`.

Success signs:
- `set_hooks(model_base)` then forward pass completes.
- `get_outputs(model_base, output)` fills HF-style outputs.

Stop signs:
- `model_base` was deleted to free memory.
- The user only asked for TL execution.

## Stage 7: Graph Initialization

Typical variables:
- `graph = Graph(dataset_size=len(_results), hidden_size=model.cfg.d_model)`
- `model.device`
- `model.dtype`
- patched `attribute.use_dequant_projections`

Success signs:
- `graph` exists and current graph registry points to it.
- TL dequant patch is installed when using TL attention.

## Stage 8: Root Attribution

Typical cell:
- `lmhead = Node(L, None, 'lm_head', attn_pattern='A-->A-')`
- `root = tnode = add_tnode(_results, selected_model, nodes)`
- `print_tree(root)`

Success signs:
- Tree prints a root layer, often like `L36` or `L64` depending on model.
- `tnode.data.top_heads` and attribution data are populated.

Stop signs:
- `selected_model` points to `model_base` after it has been deleted.
- `attribute.py` shape/device mismatch.

## Stage 9: Clustering and Pattern Scoring

Typical functions:
- `cluster_heads`
- `eval_head_lens`
- `get_head_matching_scores`

Success signs:
- Output includes `CLUSTERING SUMMARY`.
- Metrics such as cohesion, separation, ratio are printed.
- DataFrame lists `layer`, `head`, `score`, `acc`, and pattern columns.

Stop signs:
- The next step requires selecting specific heads or thresholds. Report results and ask for the user's choice unless a policy was explicitly provided.

## Stage 10: Manual Circuit Expansion

Typical cells:
- Construct `Node(..., 'attn_k'/'attn_q'/'attn_v', attn_pattern=...)`
- `tnode = add_tnode(_results, selected_model, nodes, parent=tnode)`

Success signs:
- Tree expands.
- New `tnode.data` has top heads and attention attribution.

Stop signs:
- The notebook reaches hard-coded research choices.
- A chosen head is invalid for the current model size.
- The user did not ask to continue circuit expansion.

## Stage 11: Visualization

Typical functions:
- `show_attn`
- `colored_tokens_multi`
- `merge_gqa_groups`
- `visualize_graph`

Success signs:
- Circuitsvis HTML object or pyvis graph is produced.
- Visualization rendering object is not a failure by itself.

Stop signs:
- Browser/UI rendering is unavailable but data was produced. Report the generated object or file path instead.
