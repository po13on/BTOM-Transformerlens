#!/usr/bin/env python3
"""Compatibility check: varlen-patched attribute.py/vis.py on fixed-length Hi-ToM.

Uses Qwen3-8B 4bit, first 20 rows of Hi_ToM_order_1.csv (same as last saved
notebook forward). Open-ended 5 rounds. Writes hitom_compat_report.md.
"""
from __future__ import annotations

import ast
import gc
import io
import json
import os
import sys
import traceback
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib

matplotlib.use("Agg")

ROOT = Path("/home/hushengchun/project/BTOM-transformerlens")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pptree"))

CSV = ROOT / "data_uniform" / "Hi_ToM_order_1.csv"
MODEL_DIR = Path("/data0/modelscope/qwen3/models/Qwen/Qwen3-8B")
REPORT = ROOT / "hitom_compat_report.md"
STATUS = ROOT / "hitom_compat_status.md"
TZ = timezone(timedelta(hours=8))
N_SAMPLES = 20
MAX_ROUNDS = 5
CLUSTER_THRESHOLD = 0.35
ATTN_PATTERN_THRESHOLD = 0.3
TOP_K = 30  # original add_tnode default, not the scat k=12

PREV_AV = {(26, 26), (27, 18), (28, 0), (29, 11), (32, 3), (32, 2), (28, 3), (24, 31), (29, 12)}
PREV_TREE = (
    "L36┐\n"
    "   └L32,29,28,27,26 A-->V x6┐\n"
    "                            └L30,29,28,26,25,24 A-->V x6┐\n"
    "                                                        └L24,22,20,19,18 A-->A- x6┐\n"
    "                                                                                  └L22,21,19,17,16,14 V->VK_I x6┐\n"
    "                                                                                                                └L0 A-->A- x6"
)


def now():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")


def write_status(text: str):
    STATUS.write_text(
        f"# Hi-ToM 兼容性复跑\n\n- 时间：{now()}\n- {text}\n- 模型：{MODEL_DIR}\n- 数据：{CSV} n={N_SAMPLES}\n",
        encoding="utf-8",
    )
    print(f"[status] {text}", flush=True)


def capture_tree(root):
    from pptree import print_tree

    buf = io.StringIO()
    with redirect_stdout(buf):
        print_tree(root)
    return buf.getvalue().rstrip()


def patch_attribute():
    import attribute
    from contextlib import contextmanager as cm

    orig = attribute.use_dequant_projections

    @cm
    def _patched(module, layer, grad_inputs):
        if hasattr(module, "W_O") and not hasattr(module, "o_proj"):
            yield
            return
        with orig(module, layer, grad_inputs):
            yield

    attribute.use_dequant_projections = _patched


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformer_lens import HookedTransformer

    write_status("加载 Qwen3-8B 4bit → TransformerLens")
    tok = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True, trust_remote_code=True, use_fast=False
    )
    tok.pad_token = tok.pad_token or tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    hf = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        attn_implementation="eager",
        output_hidden_states=True,
        device_map="auto",
        quantization_config=bnb,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True,
    )
    hf.tokenizer = tok
    tl_name = "/".join(MODEL_DIR.parts[-2:])
    model = HookedTransformer.from_pretrained(
        tl_name,
        hf_model=hf,
        use_split_qkv_attention=False,
        load_in_4bit=True,
        tokenizer=tok,
        device="cuda",
        device_map="auto",
        move_to_device=True,
        torch_dtype=torch.float16,
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        fold_value_biases=False,
        trust_remote_code=True,
    )
    del hf
    gc.collect()
    torch.cuda.empty_cache()
    model.device = model.cfg.device
    model.dtype = model.cfg.dtype
    print(f"TL {tl_name} L={model.cfg.n_layers} H={model.cfg.n_heads}", flush=True)
    return model, tok


def build_results(model, tokenizer):
    import pandas as pd
    import torch
    from min_arc import Result
    from model_hooks import get_outputs_from_cache

    write_status(f"读取 Hi-ToM 前 {N_SAMPLES} 条并前向")
    df = pd.read_csv(CSV)
    df["index"] = df["index"].apply(ast.literal_eval)
    df = df.head(N_SAMPLES)
    lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in df["prompt"]]
    print("encode lengths", Counter(lens), flush=True)

    names_filter = (
        [f"blocks.{l}.ln1.hook_scale" for l in range(model.cfg.n_layers)]
        + [f"blocks.{l}.hook_mlp_out" for l in range(model.cfg.n_layers)]
        + [f"blocks.{l}.hook_resid_pre" for l in range(model.cfg.n_layers)]
        + [f"blocks.{l}.ln2.hook_scale" for l in range(model.cfg.n_layers)]
        + ["ln_final.hook_scale", "ln_final.hook_normalized"]
        + [f"blocks.{l}.attn.hook_z" for l in range(model.cfg.n_layers)]
    )
    results = []
    for i, row in df.iterrows():
        prompt = row["prompt"]
        answer = row["answer"]
        choices = [c.split(". ")[1].strip() for c in row["choices"].split(", ")]
        candidate_ids = [tokenizer.encode(" " + c)[0] for c in choices]
        label = tokenizer.encode(" " + answer)[0]
        n_tok = len(tokenizer.encode(prompt))
        spans = dict(row["index"])
        spans["A-"] = n_tok - 1
        r = Result(
            index=len(results),
            model="qwen3-8b",
            prompt=prompt,
            answers=[answer],
            answer_indices=[n_tok],
            candidate_ids=candidate_ids,
            labels=[label],
            n_tokens=n_tok,
            index_map=[spans],
        )
        inp = model.to_tokens(r.prompt, prepend_bos=False, padding_side="left")
        logits, cache = model.run_with_cache(inp, names_filter=names_filter)
        r.outputs = get_outputs_from_cache(model, cache)
        seq = int(inp.shape[1])
        r.n_tokens = seq
        r.answer_indices = [seq]
        r.index_map[0]["A-"] = seq - 1
        pred = tokenizer.decode(logits[0, seq - 1].argmax().item())
        r.responses = [pred]
        r.is_corrects = [pred.strip().strip('"').strip("'") == answer]
        del logits, cache, inp
        gc.collect()
        torch.cuda.empty_cache()
        results.append(r)
    n_ok = sum(any(x.is_corrects) for x in results)
    print(f"correct {n_ok}/{len(results)} live_n_tokens={Counter(x.n_tokens for x in results)}", flush=True)
    return results, f"{n_ok}/{len(results)}"


def available_patterns(results):
    keys = set.intersection(*(set(r.index_map[0].keys()) for r in results))
    req = {
        "A-->V": {"A-", "V"},
        "A-->A-": {"A-"},
        "V->V": {"V"},
        "V->VK_C": {"V", "VK_C"},
        "V->V": {"V"},
        "V->VK_I": {"V", "VK_I"},
        "A-->QK_C": {"A-", "QK_C"},
        "A-->QK_I": {"A-", "QK_I"},
    }
    return [p for p, need in req.items() if need <= keys]


def score_df(tnode, results, model, patterns):
    import pandas as pd
    from common_utils import mr
    from vis import eval_head_lens, get_head_matching_scores

    rows = []
    for (l, h), score in tnode.data.top_heads.items():
        ap = mr(get_head_matching_scores)(results, patterns, model, l, h)
        rows.append(
            {
                "layer": int(l),
                "head": int(h),
                "score": float(score),
                "acc": float(mr(eval_head_lens)(results, model, l, h, strict=False).item()),
                **{p: float(ap[p].mean().item()) for p in patterns},
            }
        )
    return pd.DataFrame(rows).sort_values("A-->V", ascending=False).reset_index(drop=True)


def pick_nodes(df, hung, rnd, patterns):
    from attribute import Node

    nodes, ledger = [], []
    # Stage 0（与 skill / 上次开放式一致）：A-->V 列 top-2 出口 × {q,k,v}，不设绝对分数门槛。
    if rnd == 1 and "A-->V" in df.columns:
        for _, row in df.sort_values("A-->V", ascending=False).head(2).iterrows():
            l, h = int(row["layer"]), int(row["head"])
            s = float(row["A-->V"])
            for ntype in ("attn_k", "attn_q", "attn_v"):
                key = (l, h, ntype, "A-->V")
                if key in hung:
                    continue
                hung.add(key)
                n = Node(l, h, ntype, attn_pattern="A-->V")
                nodes.append(n)
                ledger.append((n, rnd, s))
        return nodes, ledger

    order = [p for p in ("A-->V", "V->VK_I", "V->VK_C", "A-->QK_C", "A-->QK_I", "V->V", "A-->A-") if p in patterns]
    ntype_of = {
        "A-->V": "attn_k",
        "V->VK_I": "attn_k",
        "V->VK_C": "attn_k",
        "A-->QK_C": "attn_q",
        "A-->QK_I": "attn_q",
        "V->V": "attn_v",
        "A-->A-": "attn_v",
    }
    # Stage 1+：先沿 A-->V 通路（上次 canvas 连续两层都是 A-->V），再补其它角色。
    for _, row in df.sort_values("A-->V", ascending=False).iterrows():
        if len(nodes) >= 6:
            break
        l, h = int(row["layer"]), int(row["head"])
        scores = {p: float(row[p]) for p in order}
        av = scores.get("A-->V", 0.0)
        best_p = max(scores, key=scores.get)
        best_s = scores[best_p]
        if av >= 0.002 and av >= 0.25 * max(best_s, 1e-9):
            pattern, ntype, score = "A-->V", "attn_k", av
        elif best_s >= ATTN_PATTERN_THRESHOLD:
            pattern, ntype, score = best_p, ntype_of[best_p], best_s
        else:
            continue
        key = (l, h, ntype, pattern)
        if key in hung or any(n.layer == l and n.head == h and n.type == ntype for n in nodes):
            continue
        hung.add(key)
        n = Node(l, h, ntype, attn_pattern=pattern)
        nodes.append(n)
        ledger.append((n, rnd, score))
    return nodes, ledger


def run_loop(model, results):
    import torch
    from attribute import Graph, Node, add_edges, add_tnode
    from vis import cluster_heads

    patch_attribute()
    patterns = available_patterns(results)
    print("patterns", patterns, flush=True)
    graph = Graph(dataset_size=len(results), hidden_size=model.cfg.d_model)
    lm = Node(model.cfg.n_layers, None, "lm_head", attn_pattern="A-->A-")
    graph.add_node(lm)
    write_status("构建 lm_head 根节点（变长补丁代码路径）")
    root = tnode = add_tnode(results, model, [lm], k=TOP_K)
    print(capture_tree(root), flush=True)

    hung, ledger = set(), []
    stop = f"达到 max_attribution_rounds={MAX_ROUNDS}"
    last_top = None
    n_rounds = 0
    for rnd in range(1, MAX_ROUNDS + 1):
        n_rounds = rnd
        write_status(f"Round {rnd} cluster/score/expand")
        d = tnode.data
        th = CLUSTER_THRESHOLD
        try:
            groups, *_ = cluster_heads(
                d.attn_attrs_ds,
                threshold=th,
                strengths=d.top_heads,
                model_config=model.cfg,
                show_plot=False,
            )
            if len(groups) < 2:
                th = max(0.15, th - 0.05)
                groups, *_ = cluster_heads(
                    d.attn_attrs_ds,
                    threshold=th,
                    strengths=d.top_heads,
                    model_config=model.cfg,
                    show_plot=False,
                )
        except Exception as e:
            groups = {0: list(d.top_heads)}
            print(f"[cluster] fallback: {e}", flush=True)
        print(f"n_groups={len(groups)} th={th}", flush=True)
        df = score_df(tnode, results, model, patterns)
        print(df.to_csv(sep="\t", index=True), flush=True)
        prev_hits = []
        if "A-->V" in df.columns:
            for _, row in df.iterrows():
                lh = (int(row["layer"]), int(row["head"]))
                if lh in PREV_AV:
                    prev_hits.append((lh, float(row["A-->V"]), float(row.get("acc", 0))))
        print(f"[prev A-->V in this top_k] {prev_hits}", flush=True)
        (ROOT / "hitom_compat_rounds.jsonl").open("a", encoding="utf-8").write(
            json.dumps({"round": rnd, "prev_av_hits": [(list(a), b, c) for a, b, c in prev_hits], "df": df.to_dict(orient="records")}, ensure_ascii=False)
            + "\n"
        )
        nodes, more = pick_nodes(df, hung, rnd, patterns)
        ledger.extend(more)
        if not nodes:
            stop = "归因停滞（无新节点）"
            break
        for n in nodes:
            add_edges(graph, n, tnode.data.nodes, tnode.data.attr)
        tnode = add_tnode(results, model, nodes, parent=tnode, k=TOP_K)
        tree = capture_tree(root)
        print(tree, flush=True)
        top_now = tuple(tnode.data.top_heads.keys())
        if last_top == top_now:
            stop = "归因停滞（top_heads 未变）"
            break
        last_top = top_now
        gc.collect()
        torch.cuda.empty_cache()
    return root, capture_tree(root), ledger, stop, n_rounds, patterns


def write_report(tree, ledger, stop, n_rounds, acc, patterns, crash):
    from collections import Counter

    new_heads = {(n.layer, n.head) for n, _, _ in ledger}
    av_heads = {(n.layer, n.head) for n, _, s in ledger if n.attn_pattern == "A-->V"}
    overlap = sorted(av_heads & PREV_AV)
    missed = sorted(PREV_AV - av_heads)
    extra = sorted(av_heads - PREV_AV)
    pat_c = Counter(n.attn_pattern for n, _, _ in ledger)
    rows = "\n".join(
        f"| ({n.layer}, {n.head}) | {n.attn_pattern} | {n.type} | {s:.4f} | r{rnd} |"
        for n, rnd, s in ledger
    )
    ok = crash is None
    text = f"""# Hi-ToM 定长兼容性复跑

生成时间：{now()} (UTC+8)

- 模型：Qwen3-8B 4bit (`{MODEL_DIR}`)
- 数据：`Hi_ToM_order_1.csv` 前 {N_SAMPLES} 条（csv 中 `prompt_token_length` 全为 205）
- 代码：当前变长补丁（`_cat_pad_seq` / 每样本 `set_pos_ids` / JS pad trim）
- 前向正确率：{acc}
- 轮次：{n_rounds}；停止：{stop}
- 可评分模式：{patterns}
- 是否跑通（无 cat/OOB 崩溃）：{'是' if ok else '否：' + crash}

## 上次树（canvas）

```
{PREV_TREE}
```

## 本次树

```
{tree}
```

## 与上次 A-->V 头重叠

- 上次 canvas 列出的 A-->V 头：{sorted(PREV_AV)}
- 本次挂上的 A-->V 头：{sorted(av_heads)}
- **重叠** {len(overlap)}/{len(PREV_AV)}：{overlap}
- 上次有、本次没有：{missed}
- 本次有、上次没有：{extra}

## 本次模式计数

{dict(pat_c)}

上次 canvas：A-->V 9 / A-->A- 8 / V->VK_I 8 / A-->QK_C 3 / V->VK_C 2（节点数，含同一头多种 type）

## 节点

| 头 | 模式 | 类型 | 分数 | 轮次 |
|---|---|---|---|---|
{rows}
"""
    REPORT.write_text(text, encoding="utf-8")
    print(f"[report] {REPORT}", flush=True)
    return overlap, missed, extra, ok


def main():
    write_status("启动 Hi-ToM 兼容性复跑")
    crash = None
    try:
        model, tok = load_model()
        results, acc = build_results(model, tok)
        root, tree, ledger, stop, n_rounds, patterns = run_loop(model, results)
        write_report(tree, ledger, stop, n_rounds, acc, patterns, None)
        write_status(f"完成 — 见 {REPORT.name}；stop={stop}")
        print("DONE True", stop, flush=True)
    except Exception:
        crash = traceback.format_exc()
        print(crash, flush=True)
        write_report("CRASH", [], str(crash.splitlines()[-1] if crash else "err"), 0, "n/a", [], crash)
        write_status(f"失败\n```\n{crash[-1500:]}\n```")
        raise


if __name__ == "__main__":
    main()
