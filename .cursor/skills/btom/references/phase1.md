# Phase 1 — Initialization

Run once per session through root node creation. Apply user-set parameters before running.

Use Notebook MCP `notebook_run_cell` by 0-based `index`. **Cell id is authoritative** — indexes drift; verify with `notebook_list_cells` before running.

Skip unlisted cells by default: `10` (`9465a27b`, optional GPTQ) and `16` (`c6f7f769`, token helper) unless the user asks.

| Phase | MCP index | Cell id | Purpose / notes |
| ----- | --------- | ------- | --------------- |
| Imports & autoreload | `0` | `030bdbc8` | Autoreload, IPython display |
| Imports & project modules | `1` | `9341f87e` | Env, local imports, `pptree`, `vis`, `attribute` |
| Torch / transformers | `2` | `09d30546` | Torch, disable grad |
| Model cache init | `3` | `03d4e10e` | `models = {}` |
| 4bit config | `4` | `c372a448` | Run only if `use_4bit=true` |
| HF model load | `5` | `ce9370d7` | Apply `model_name` / `use_4bit`. Stop on CUDA/OOM/missing weights |
| HF model metadata | `6` | `6997eba9` | Cache `model_base`; set `L`, `H`, `V` |
| TL imports | `7` | `1d0aa69f` | `HookedTransformer`, TL utils, `gc` |
| TL conversion | `8` | `67f9525e` | Only if `use_transformerlens=true`; apply `use_4bit` |
| Release HF model | `9` | `e203b118` | After TL succeeds. If HF-only, skip and keep `model_base` |
| Data loading | `11` | `64823fec` | Apply `dataset_path` / `n_samples`; builds `results` |
| Hook filter | `12` | `a91ab7f2` | `names_filter` for TL cache |
| Result alias | `13` | `e7af6efa` | `_results = results` |
| TL forward cache | `14` | `0904d383` | Writes `r.outputs` |
| HF forward (optional) | `15` | `e1db90eb` | Only HF-only path |
| Result filter | `17` | `b40f1e25` | Keep all unless user asks subset |
| Graph init | `18` | `6283ffc5` | `graph`, device/dtype |
| Attribution patch | `19` | `d1b04a55` | Dequant context for TL attn |
| Root node | `20` | `8cbc7c04` | `lm_head`; `root = tnode = add_tnode(...)` |

After Phase 1, `tnode` is the current attribution node → go to Phase 2.

## Parameter → cell mapping

| Param | Where |
| ----- | ----- |
| `model_name` | cell `ce9370d7`; also drives TL cell `67f9525e` |
| `dataset_path` | cell `64823fec` → `hi_tom_path` |
| `use_4bit=false` | Remove `quantization_config` / `load_in_4bit` in load + TL cells; keep `float16` |
| `use_transformerlens=false` | Skip `67f9525e`–`e203b118`; keep `model_base`. Default loop is TL-first |

## Agent-adjustable defaults

| Param | Default | Guidance |
| ----- | ------- | -------- |
| `n_samples` | `max(20, len(df_tom))` | Reduce on OOM/slow; increase if high variance |
| `cluster_threshold` | `0.35` | ±0.05 if clusters too coarse/fine |
| `attn_pattern_threshold` | `0.3` | ±0.05 if too many/few candidates |
| `max_attribution_rounds` | `5` | Raise 6–8 if tree shallow; lower on OOM |

## CUDA / model path

- Prefer a free GPU in `CUDA_VISIBLE_DEVICES` before Phase 1 if the default device is busy.
- Local ordinary model dir → absolute path to `from_pretrained`. Do not assume `cache_dir` + HF repo id finds ModelScope-style dirs.
- If `model.safetensors.index.json` exists, verify shard files before deeper diagnosis.

## Key local files

- `test.ipynb`, `data_uniform/Hi_ToM_order_1.csv`
- `model_hooks.py`, `attribute.py`, `vis.py`
