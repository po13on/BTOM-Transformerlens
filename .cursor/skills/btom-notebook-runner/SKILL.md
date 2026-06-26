---
name: btom
description: Run and supervise BTOM-TransformerLens test.ipynb experiments. Use when the user asks the agent to run, continue, debug, or monitor /home/hushengchun/project/BTOM-transformerlens/test.ipynb, especially when execution should proceed cell-by-cell based on outputs, errors, model-loading state, cache generation, attribution results, clustering summaries, or visualization outputs.
---

# BTOM Notebook Runner

Use this skill to run `/home/hushengchun/project/BTOM-transformerlens/test.ipynb` as a supervised experiment rather than as a blind "run all" notebook.

The notebook is stateful, GPU-heavy, and partially exploratory. Treat each code cell output as evidence for whether to continue, skip an optional branch, fix a local setup issue, or stop and report.

## First Principles

1. Work from the project root:
   `/home/hushengchun/project/BTOM-transformerlens`.
2. Prefer running cells incrementally. Do not run the whole notebook blindly unless the user explicitly asks.
3. Preserve the user's notebook unless they ask for edits. If a temporary change is needed for execution, explain it first.
4. Treat model loading, CUDA memory, local model paths, and missing weights as stopping conditions unless the fix is obvious and non-destructive.
5. Use the newest user request to define the goal. Do not continue into later attribution or visualization sections if the requested goal was only model loading, data validation, or forward cache generation.

## Execution Workflow

1. Inspect notebook state.
   - Identify the target cell range from the user request.
   - Check whether required variables already exist in the kernel if using an active Jupyter session.
   - If no active kernel is available, run a fresh kernel and start from the required prerequisites.

2. Classify the current stage.
   - Initialization/imports.
   - Model loading.
   - TransformerLens conversion.
   - Hi-ToM data loading.
   - Forward cache generation.
   - HF/TL cache comparison.
   - Graph initialization and attribution.
   - Head clustering and pattern scoring.
   - Manual node selection and circuit expansion.
   - Visualization or reporting.

3. Run only the next necessary cell or small contiguous group.
   - After each run, inspect stdout, stderr, traceback, displayed objects, and key variables.
   - Continue only if the output satisfies the success condition for that stage.
   - Record key results in the response: model path, sample count, cache fields, accuracy/logprob summaries, top heads, cluster metrics, or the exact blocker.

4. Decide whether to continue.
   Continue when:
   - The cell completed without traceback.
   - Required variables for the next stage are present.
   - Outputs have expected shapes or summaries.
   - The user asked for the downstream result.

   Stop and report when:
   - A model directory is missing weight shards or config files.
   - `local_files_only=True` prevents loading a missing model.
   - CUDA OOM occurs or memory is clearly insufficient.
   - A cell depends on a deleted variable such as `model_base`.
   - A manual research choice is needed, such as choosing top heads or selecting an attribution branch.
   - The current output already answers the user's request.
   - A long-running cell appears hung or repeatedly fails.

## Notebook-Specific Guidance

Read `NOTEBOOK_FLOW.md` before running unfamiliar sections of the notebook.

Key local files:
- `test.ipynb`: main experiment notebook.
- `data_uniform/Hi_ToM_order_1.csv`: current Hi-ToM data source.
- `model_hooks.py`: HF/TL output collection and attention adaptation.
- `attribute.py`: graph, node, and attribution logic.
- `vis.py`: head scoring, clustering, and visualization.

Common model-loading rule:
- If using a local ordinary model directory, pass the absolute path directly to `from_pretrained`.
- Do not assume `cache_dir` plus a HuggingFace repo id will find a ModelScope-style directory.
- If `model.safetensors.index.json` exists, verify the referenced shard files also exist before diagnosing deeper.

## Output Format

When reporting progress or a result, use:

```markdown
## 当前阶段
[one sentence]

## 已观察到的输出
[important stdout/stderr/metrics, concise]

## 判断
[continue / stop / needs user decision, with reason]

## 下一步
[the exact next cell or action, if continuing is appropriate]
```

For a blocker, include the exact file/path/model/cell context and the safest fix.

## Do Not

- Do not run optional API/client cells unless the user asks.
- Do not download models unless the user explicitly allows network/downloads.
- Do not clear or overwrite user results unless asked.
- Do not keep executing after the notebook reaches a manual branch that requires choosing heads, thresholds, or circuit nodes.
- Do not treat rich visualization HTML as a failure if the cell completed and produced a render object.
