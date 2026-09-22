#!/usr/bin/env python3
"""Detached scat_targeted attribution driver for tmux/screen.

Uses the QLoRA adapter from fine-tune.ipynb:
  /data0/hushengchun/qwen3-32b-lora-scat_targeted/adapter

Variable-length handling: mix all Query×Output shapes from the 2000-row
test CSV (round-robin). Roles are Final/Copy/QLast/WinDigit (not Hi-ToM QK_I).

Survives SSH disconnect. Writes attribution_status.md / attribution_report.md
and syncs Phase 2 cluster/score/expand cells into test.ipynb (between landmarks).
"""
from __future__ import annotations

import gc
import io
import json
import os
import sys
import traceback
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("MODELSCOPE_CACHE", "/data0/modelscope/AI-ModelScope")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib

matplotlib.use("Agg")

ROOT = Path("/home/hushengchun/project/BTOM-transformerlens")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pptree"))

NB_PATH = ROOT / "test.ipynb"
STATUS_PATH = ROOT / "attribution_status.md"
REPORT_PATH = ROOT / "attribution_report.md"
JSONL = ROOT / "data_uniform" / "scat_targeted.csv"
MODEL_DIR = Path("/data0/modelscope/qwen3/models/Qwen/Qwen3-32B")
ADAPTER_DIR = Path("/data0/hushengchun/qwen3-32b-lora-scat_targeted/adapter")
TZ = timezone(timedelta(hours=8))

TARGET_CORRECT = 8
MAX_FORWARD = 40
MAX_ROUNDS = 5
CLUSTER_THRESHOLD = 0.35
TOP_K = 12
ATTN_PATTERNS = [
    "Final->Copy",
    "Final->DemoOut1",
    "Final->DemoOutN",
    "Final->QLast",
    "Final->WinDigit",
    "QLast->WinDigit",
    "QLast->DemoQLast",
    "Final->Final",
    "Copy->Copy",
]
CROSS_SPAN = (
    "Final->Copy",
    "Final->DemoOut1",
    "Final->DemoOutN",
    "Final->QLast",
    "Final->WinDigit",
    "QLast->WinDigit",
    "QLast->DemoQLast",
)
AV_EXPLAIN_MIN = 0.03
MIXED_LENGTH = True


def now_str():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")


def gpu_mem():
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return f"alloc={alloc:.2f}GiB reserved={reserved:.2f}GiB"


def write_status(text: str):
    STATUS_PATH.write_text(
        f"# scat_targeted 归因进度\n\n"
        f"- 时间：{now_str()} (UTC+8)\n"
        f"- 会话：`tmux attach -t btom-scat-attr`（断线后任务仍在跑）\n"
        f"- {text}\n"
        f"- 笔记本：`{NB_PATH}`\n"
        f"- 数据：`{JSONL}`\n"
        f"- 合并微调模型：`{MERGED_DIR}`（fp16 merge 后再 4bit + TL）\n"
        f"- LoRA 源：`{ADAPTER_DIR}`\n"
        f"- CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} {gpu_mem()}\n"
        f"- 日志：`{ROOT / 'logs' / 'scat_attr_tmux.log'}`\n"
        f"- 最终报告：`{REPORT_PATH}`\n",
        encoding="utf-8",
    )
    print(f"[status] {text}", flush=True)


def capture_tree(root):
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_tree(root)
    return buf.getvalue().rstrip()


def sync_workspace_cells(blocks: list[tuple[str, str]]):
    """Overwrite cells strictly between upper/lower landmarks. Do not touch landmarks."""
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]
    upper = lower = None
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if "visualize_model_heads(root, selected_model, _results, sample=_results[0]" in src and upper is None:
            upper = i
        if "colored_tokens_multi(*show_attn(random.choice(_results), selected_model, 51, 10" in src:
            lower = i
    if upper is None or lower is None or lower <= upper + 1:
        print(f"[warn] landmarks not found upper={upper} lower={lower}", flush=True)
        return
    span = lower - upper - 1
    while len(blocks) < span:
        blocks.append(("# (reserved workspace)\npass\n", ""))
    blocks = blocks[:span]
    for offset, (source, output) in enumerate(blocks):
        idx = upper + 1 + offset
        cells[idx]["cell_type"] = "code"
        cells[idx]["source"] = [line + "\n" for line in source.rstrip("\n").split("\n")]
        cells[idx]["outputs"] = [
            {"name": "stdout", "output_type": "stream", "text": [output] if output else [""]}
        ] if output else []
        cells[idx]["execution_count"] = offset + 1
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[notebook] synced {len(blocks)} workspace cells ({upper+1}..{lower-1})", flush=True)


def score_dataframe(tnode, results, model):
    d = tnode.data
    rows = []
    for (l, h), score in d.top_heads.items():
        ap = mr(get_head_matching_scores)(results, ATTN_PATTERNS, model, l, h)
        acc = mr(eval_head_lens)(results, model, l, h, strict=False).item()
        acc0 = mr(eval_head_lens)(results, model, l, h, strict=True).item()
        rows.append(
            {
                "layer": int(l),
                "head": int(h),
                "score": float(score),
                "acc": float(acc),
                "acc0": float(acc0),
                **{p: float(ap[p].mean().item()) for p in ATTN_PATTERNS},
            }
        )
    df = pd.DataFrame(rows)
    if len(df) and "Final->Copy" in df.columns:
        df["_cross"] = df[[c for c in CROSS_SPAN if c in df.columns]].max(axis=1)
        df = df.sort_values(["_cross", "Final->Copy"], ascending=False).reset_index(drop=True)
        df = df.drop(columns=["_cross"])
    else:
        df = df.reset_index(drop=True)
    return df


def role_for(pattern, ntype):
    catalog = {
        ("Final->Copy", "attn_k"): ("Copy-key enabler", "Final, Copy"),
        ("Final->Copy", "attn_q"): ("Copy-query former", "Final"),
        ("Final->Copy", "attn_v"): ("Copy value reader", "Copy"),
        ("Final->Final", "attn_v"): ("Final-position assembler", "Final"),
        ("Copy->Copy", "attn_v"): ("Copy-span self", "Copy"),
        ("Final->QLast", "attn_q"): ("Final reads Test Query last digit", "Final, QLast"),
        ("Final->DemoOut1", "attn_k"): ("Final reads last-demo Output first cell", "Final, DemoOut1"),
        ("Final->DemoOutN", "attn_k"): ("Final reads last-demo Output last cell", "Final, DemoOutN"),
        ("Final->WinDigit", "attn_k"): ("Final reads winner query-digit", "Final, WinDigit"),
        ("QLast->WinDigit", "attn_q"): ("Query last digit points at winner same digit", "QLast, WinDigit"),
        ("QLast->DemoQLast", "attn_q"): ("Test Query aligns to last-demo Query", "QLast, DemoQLast"),
    }
    return catalog.get((pattern, ntype), catalog.get((pattern, "attn_q"), ("other", "")))


def write_report(payload: dict):
    tree = payload["tree"]
    rows = payload["ledger"]
    matches = payload["matches"]
    floor = payload["floor"]
    viz_ok = payload["viz_ok"]
    explain = payload["explain"]
    stop = payload["stop"]
    attempt = payload["attempt"]
    cohort = payload.get("cohort", "n/a")
    if matches and explain:
        table = "\n".join(
            f"| ({l}, {h}) | Pattern Score | {s:.4f} | {p} | {t} | 第 {r} 层归因 |"
            for (l, h, p, t, s, r) in matches
        )
        sec1 = (
            "## 任务结果\n\n"
            "✅ **已找到跨位置注意力头（并继续向上追溯）** — `Final->Copy` / `QLast->WinDigit` 等如下（含多轮累计）：\n\n"
            "| 注意力头 (Layer, Head) | 所用指标 | 分数 | 注意力模式 | 节点类型 | 归因层级 |\n"
            "|------------------------|---------|------|-----------|---------|---------|\n"
            f"{table}\n\n"
            f"> **续归因说明**：首次命中后不会停止；同一目标下继续向上直到 max_attribution_rounds={MAX_ROUNDS} 或 agent-set 终止。\n"
            f"> **分数说明**：Pattern Score（DataFrame 列）≠ Positive Bound（visualize_model_heads 点击查看）。\n"
            f"> **经验门槛**：match_score_floor={floor:.4f}（指标 Final->Copy；参考头 {payload['ref_head']}）。停止原因：{stop}。尝试次数：{attempt}。\n"
            f"> **数据**：{cohort}（全量 2000 条变长混合，不再锁同一 Query×Output 长度）。\n"
            f"> **可视化审阅**：{'已调用 visualize_model_heads（tmux 无 widget，请重连后在上界 cell 再点一次）' if viz_ok else 'visualize_model_heads 在无显示环境下未成功渲染 widget；请重连后重跑上界 cell'}。\n"
        )
    else:
        sec1 = (
            "## 任务结果\n\n"
            "❌ **未找到能稳定解释数据集的匹配注意力头**\n\n"
            f"已遍历 {payload['n_rounds']} 轮。match_score_floor = {floor:.4f}（指标: Final->Copy；参考头: {payload['ref_head']}）。\n"
            f"停止原因：{stop}。尝试次数：{attempt}。\n"
            f"首 token 正确率：{payload['acc_str']}。QLast->WinDigit 列最大值：{payload['max_qki']:.4f}。\n"
            f"数据：{cohort}。Final->Copy 最大值：{payload.get('max_av', 0):.4f}。\n"
        )
    ledger_md = "\n".join(
        f"| ({n.layer}, {n.head}) | {n.attn_pattern} | {n.type} | {role_for(n.attn_pattern, n.type)[1]} | {role_for(n.attn_pattern, n.type)[0]} | hung in round {rnd} |"
        for n, rnd in rows
    )
    text = (
        f"# scat_targeted 归因报告\n\n"
        f"生成时间：{now_str()} (UTC+8)\n\n"
        f"模型：Qwen3-32B + LoRA `{ADAPTER_DIR}`\n\n"
        f"{sec1}\n"
        f"## 归因树\n\n"
        f"自动发现的归因电路结构（`print_tree(root)` 输出）：\n\n"
        f"```\n{tree}\n```\n\n"
        f"> **如何阅读**：`L层号 模式 x节点数`；自上而下向输入端追溯。\n"
        f"> **注意**：print_tree 可能折叠同层混合模式；完整角色以节点角色为准。\n\n"
        f"## 节点角色\n\n"
        f"| 注意力头 | 模式 | 节点类型 | 活跃前沿 | 角色 | 作用说明 |\n"
        f"|---------|------|---------|---------|------|---------|\n"
        f"{ledger_md}\n"
    )
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"[report] wrote {REPORT_PATH}", flush=True)


def _free_cuda():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        from graph_registry import clear_attn_weights_cache

        clear_attn_weights_cache()
    except Exception:
        pass


MERGED_DIR = Path("/data0/hushengchun/qwen3-32b-lora-scat_targeted-merged-fp16")
MERGED_DIR_TMP = Path("/tmp/qwen3-32b-lora-scat_targeted-merged")


def _ensure_fp16_merged(tokenizer):
    """CPU fp16 合并 LoRA 后落盘，再 4bit 加载。不要在已量化权重上 merge。"""
    import shutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    for reused in (MERGED_DIR, MERGED_DIR_TMP):
        if (reused / "config.json").exists():
            print(f"[load] reuse merged dir {reused}", flush=True)
            return reused
    free_gb = shutil.disk_usage("/tmp").free / 1024**3
    if free_gb < 70:
        raise RuntimeError(f"/tmp only {free_gb:.1f}GiB free; need ~65GiB for fp16 merge")
    write_status(f"进行中 — CPU fp16 合并 LoRA 得到微调模型（/tmp 剩余 {free_gb:.0f}GiB）")
    base_cpu = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    peft_cpu = PeftModel.from_pretrained(base_cpu, str(ADAPTER_DIR))
    write_status("进行中 — merge_and_unload（fp16，adapter 折进基座）")
    merged = peft_cpu.merge_and_unload()
    del base_cpu, peft_cpu
    gc.collect()
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    write_status(f"进行中 — 写出微调模型到 {MERGED_DIR}")
    merged.save_pretrained(str(MERGED_DIR), safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(str(MERGED_DIR))
    del merged
    gc.collect()
    print(f"[load] saved merged model -> {MERGED_DIR}", flush=True)
    return MERGED_DIR


def load_model():
    write_status("进行中 — 先 fp16 合并得到微调模型，再 4bit 加载并归因")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformer_lens import HookedTransformer

    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {ADAPTER_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    merged_dir = _ensure_fp16_merged(tokenizer)

    bnbconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    write_status(f"进行中 — 从合并后的微调模型 4bit 加载 ({merged_dir})")
    model_base = AutoModelForCausalLM.from_pretrained(
        str(merged_dir),
        attn_implementation="eager",
        output_hidden_states=True,
        device_map="auto",
        quantization_config=bnbconfig,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True,
    )
    model_base.tokenizer = tokenizer
    write_status(f"进行中 — 微调模型已 4bit 加载，转 TransformerLens ({gpu_mem()})")
    tl_name = "/".join(MODEL_DIR.parts[-2:])
    model = HookedTransformer.from_pretrained(
        tl_name,
        hf_model=model_base,
        use_split_qkv_attention=False,
        load_in_4bit=True,
        tokenizer=tokenizer,
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
    del model_base
    _free_cuda()
    model.device = model.cfg.device
    model.dtype = model.cfg.dtype
    print(
        f"Loaded TL+merged-LoRA {tl_name} L={model.cfg.n_layers} H={model.cfg.n_heads} {gpu_mem()}",
        flush=True,
    )
    return model, tokenizer


def _forward_one(model, tokenizer, row, loc, shapes, digit_ids, names_filter):
    from min_arc import Result
    from model_hooks import get_outputs_from_cache
    import torch

    n_tokens = len(tokenizer.encode(row["prompt"], add_special_tokens=False))
    loc = dict(loc)
    loc["Final"] = loc["A-"] = n_tokens - 1
    first_id = tokenizer.encode(row["answer"], add_special_tokens=False)[0]
    spans = dict(row.get("spans") or {})
    spans.update(loc)
    spans["Final"] = spans["A-"] = n_tokens - 1
    spans["start"] = spans.get("start", loc.get("QLast", 0))
    spans["cohort"] = list(shapes["cohort"])
    r = Result(
        index=0,
        model="qwen3-32b-lora-scat_targeted",
        prompt=row["prompt"],
        answers=[tokenizer.decode([first_id])],
        answer_indices=[n_tokens],
        candidate_ids=digit_ids,
        labels=[first_id],
        n_tokens=n_tokens,
        rel_fn=row.get("dataset", "scat_targeted"),
        puzzle=row["task"],
        index_map=[spans],
    )
    model_inputs = model.to_tokens(r.prompt, prepend_bos=False, padding_side="left")
    output_logits, cache = model.run_with_cache(model_inputs, names_filter=names_filter)
    r.outputs = get_outputs_from_cache(model, cache)
    seq_len = int(model_inputs.shape[1])
    if seq_len == n_tokens + 1:
        from export_mtm_csv import SCAT_SHIFT_KEYS

        for k in SCAT_SHIFT_KEYS:
            if k in r.index_map[0] and isinstance(r.index_map[0][k], int):
                r.index_map[0][k] = int(r.index_map[0][k]) + 1
        if "CandDigits" in r.index_map[0]:
            r.index_map[0]["CandDigits"] = [int(x) + 1 for x in r.index_map[0]["CandDigits"]]
    elif seq_len != n_tokens:
        print(f"[warn] to_tokens len {seq_len} != encode {n_tokens}", flush=True)
    r.n_tokens = seq_len
    r.answer_indices = [seq_len]
    r.index_map[0]["Final"] = r.index_map[0]["A-"] = seq_len - 1
    logits = output_logits[0, seq_len - 1]
    response = tokenizer.decode(logits.argmax(dim=-1).item())
    r.responses = [response]
    r.is_corrects = [response.strip().strip('"').strip("'") == r.answers[0]]
    r.logprobs = [logits.log_softmax(dim=-1)[first_id].item()]
    del output_logits, cache, model_inputs
    _free_cuda()
    return r


def build_results(model, tokenizer, cohort=None):
    from collections import Counter
    from export_mtm_csv import (
        load_prompt_dataset,
        locate_attribution_spans,
        task_io_shapes,
        verify_span_tokens,
    )

    digit_ids = [tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    names_filter = (
        [f"blocks.{layer}.ln1.hook_scale" for layer in range(model.cfg.n_layers)]
        + [f"blocks.{layer}.hook_mlp_out" for layer in range(model.cfg.n_layers)]
        + [f"blocks.{layer}.hook_resid_pre" for layer in range(model.cfg.n_layers)]
        + [f"blocks.{layer}.ln2.hook_scale" for layer in range(model.cfg.n_layers)]
        + ["ln_final.hook_scale"]
        + [f"blocks.{layer}.attn.hook_z" for layer in range(model.cfg.n_layers)]
        + ["ln_final.hook_normalized"]
    )

    candidates = []
    n_skip = 0
    write_status("进行中 — 扫描全部 2000 条：定位 Final/Copy/QLast/WinDigit（不按长度分桶）")
    for row in load_prompt_dataset(str(JSONL)):
        shapes = task_io_shapes(row["task"])
        if shapes is None:
            n_skip += 1
            continue
        loc = locate_attribution_spans(row["prompt"], row["task"], tokenizer)
        ok, reasons = verify_span_tokens(
            tokenizer, row["prompt"], loc, shapes["gold"], shapes["query"]
        )
        if "Copy" not in loc or "QLast" not in loc or "WinDigit" not in loc or "DemoOut1" not in loc or not ok:
            n_skip += 1
            continue
        candidates.append((row, loc, shapes))

    counts = Counter(item[2]["cohort"] for item in candidates)
    print("cohort counts:", counts.most_common(8), "skipped", n_skip, flush=True)
    if MIXED_LENGTH or cohort is None:
        from collections import defaultdict

        by_c = defaultdict(list)
        for item in candidates:
            by_c[item[2]["cohort"]].append(item)
        pooled, progressed = [], True
        idxs = {c: 0 for c in by_c}
        while progressed:
            progressed = False
            for c in list(by_c):
                i = idxs[c]
                if i < len(by_c[c]):
                    pooled.append(by_c[c][i])
                    idxs[c] = i + 1
                    progressed = True
        cohort = "mixed"
        print(f"using MIXED lengths n={len(pooled)} from {len(by_c)} shapes", flush=True)
        write_status(f"进行中 — 全量变长混合 {len(pooled)} 条（{len(by_c)} 种形状），前向目标 {TARGET_CORRECT} 条正确")
    else:
        pooled = [item for item in candidates if item[2]["cohort"] == cohort]
        print(f"using cohort {cohort} n={len(pooled)}", flush=True)
        write_status(f"进行中 — 同形状队列 {cohort} 共 {len(pooled)} 条，开始前向（目标 {TARGET_CORRECT} 条首 token 正确）")

    kept, n_fwd = [], 0
    n_wrong = 0
    for row, loc, shapes in pooled:
        r = _forward_one(model, tokenizer, row, loc, shapes, digit_ids, names_filter)
        n_fwd += 1
        ok = any(r.is_corrects)
        # live-token verify after to_tokens (A- already reset)
        live_ok, live_reasons = verify_span_tokens(
            tokenizer, r.prompt, r.index_map[0], shapes["gold"], shapes["query"]
        )
        if r.index_map[0].get("Copy", 10**9) >= r.n_tokens or r.index_map[0].get("QLast", 10**9) >= r.n_tokens:
            live_ok = False
            live_reasons = live_reasons + ["span >= seq_len"]
        print(
            f"fwd {n_fwd} keep={len(kept)} last={'Y' if ok else 'N'} liveV={'Y' if live_ok else live_reasons} "
            f"n_tok={r.n_tokens} gold0={shapes['gold'][0][0]}",
            flush=True,
        )
        if ok and live_ok:
            r.index = len(kept)
            kept.append(r)
        else:
            r.outputs = None
            if not ok:
                n_wrong += 1
        if len(kept) >= TARGET_CORRECT:
            break
        if n_fwd >= MAX_FORWARD:
            break
    for i, r in enumerate(kept):
        r.index = i
    n_ok = sum(any(r.is_corrects) for r in kept)
    print(
        f"kept {len(kept)} correct={n_ok} wrong_seen={n_wrong} skipped_no_V={n_skip} {gpu_mem()}",
        flush=True,
    )
    if n_ok == 0:
        raise RuntimeError("LoRA 前向 0 首 token 正确：adapter 可能没合并进 TL")
    rate = n_ok / max(n_fwd, 1)
    print(f"first-token rate {n_ok}/{n_fwd} = {rate:.2%}", flush=True)
    if rate < 0.5:
        print(
            f"[warn] 首 token 正确率 {n_ok}/{n_fwd}={rate:.2%} < 50%，仍继续归因",
            flush=True,
        )
    return kept, f"{n_ok} / {len(kept)} ({rate:.0%} of {n_fwd} fwd)", cohort, counts


def patch_attribute():
    import attribute
    from contextlib import contextmanager as cm

    orig = attribute.use_dequant_projections

    @cm
    def _patched(module, layer, grad_inputs):
        is_tl_attn = hasattr(module, "W_O") and not hasattr(module, "o_proj")
        if is_tl_attn:
            yield
            return
        with orig(module, layer, grad_inputs):
            yield

    attribute.use_dequant_projections = _patched
    attribute.add_tnode = attribute.add_tree_node


def screen_av_heads(results, model, layer_from=32):
    """Direct A- → V attention mass, cheaper than full Pattern Score over all heads."""
    from model_hooks import get_attn_weights
    import torch

    write_status(f"进行中 — 全头探测 A-→V 注意力 L{layer_from}..{model.cfg.n_layers - 1}")
    H = model.cfg.n_heads
    acc = {(l, h): 0.0 for l in range(layer_from, model.cfg.n_layers) for h in range(H)}
    for r in results:
        a_pos = int(r.index_map[0]["Final"])
        v_pos = int(r.index_map[0]["Copy"])
        for l in range(layer_from, model.cfg.n_layers):
            aw = get_attn_weights(model, r, l, head=None, pos_ids=None, use_cache=False)
            # TL: [bsz, n_heads, q, k] — must index the A- query row, not q=0
            if aw.ndim == 4:
                col = aw[0, :, int(a_pos), v_pos]
            else:
                col = aw[0, :, v_pos]
            for h in range(H):
                acc[(l, h)] += float(col[h].item())
            del aw
        _free_cuda()
    n = max(len(results), 1)
    ranked = sorted(((k, v / n) for k, v in acc.items()), key=lambda x: -x[1])
    print("A-→V probe top10:", [(lh, round(s, 4)) for lh, s in ranked[:10]], flush=True)
    return ranked


def run_loop(model, results, attempt: int, seed_exits=None):
    import torch
    from attribute import Graph, Node, add_edges, add_tnode

    _free_cuda()
    patch_attribute()
    graph = Graph(dataset_size=len(results), hidden_size=model.cfg.d_model)
    L = model.cfg.n_layers
    lmhead = Node(L, None, "lm_head", attn_pattern="Final->Final")
    for n in [lmhead]:
        graph.add_node(n)
    write_status(f"进行中 — 第 {attempt} 次尝试，构建 lm_head 根节点")
    root = tnode = add_tnode(results, model, [lmhead], k=TOP_K)
    print(capture_tree(root), flush=True)

    hung = set()
    matches = []
    ledger = []
    nb_blocks = []
    floor = 0.0
    ref_head = "n/a"
    max_qki = 0.0
    max_av = 0.0
    stop = "达到 max_attribution_rounds"
    n_rounds = 0
    last_top = None

    for rnd in range(1, MAX_ROUNDS + 1):
        n_rounds = rnd
        write_status(f"进行中 — 第 {attempt} 次尝试 Round {rnd} cluster/score/expand {gpu_mem()}")
        d = tnode.data
        th = CLUSTER_THRESHOLD
        groups = {}
        try:
            groups, metrics, _, _ = cluster_heads(
                d.attn_attrs_ds,
                threshold=th,
                strengths=d.top_heads,
                model_config=model.cfg,
                figsize=(18, 5),
                width_ratios=(4, 1),
                bar_height=0.5,
                leaf_font_size=9,
                show_plot=False,
            )
            if len(groups) < 2:
                th = max(0.15, th - 0.05)
                groups, metrics, _, _ = cluster_heads(
                    d.attn_attrs_ds,
                    threshold=th,
                    strengths=d.top_heads,
                    model_config=model.cfg,
                    figsize=(18, 5),
                    width_ratios=(4, 1),
                    bar_height=0.5,
                    leaf_font_size=9,
                    show_plot=False,
                )
        except Exception as e:
            groups = {0: list(d.top_heads)}
            print(f"[cluster] fallback: {e}", flush=True)
        cluster_src = (
            f"# Round {rnd} cluster\n"
            f"cluster_threshold = {th}\n"
            f"d = tnode.data\n"
            f"d.groups, d.metrics, _, _ = cluster_heads(d.attn_attrs_ds, threshold=cluster_threshold, "
            f"strengths=d.top_heads, model_config=selected_model.cfg, figsize=(18, 5), "
            f"width_ratios=(4, 1), bar_height=0.5, leaf_font_size=9)\n"
        )
        cluster_out = f"n_groups={len(groups)} threshold={th}\n{groups}\n"
        nb_blocks.append((cluster_src, cluster_out))

        df = score_dataframe(tnode, results, model)
        if len(df):
            if "QLast->WinDigit" in df.columns:
                max_qki = max(max_qki, float(df["QLast->WinDigit"].max()))
            if "Final->Copy" in df.columns:
                max_av = max(max_av, float(df["Final->Copy"].max()))
        score_src = (
            f"# Round {rnd} score\n"
            f"attn_patterns = {ATTN_PATTERNS!r}\n"
            f"d = tnode.data\n"
            f"df = pd.DataFrame([...])  # Pattern Score Final->Copy / QLast->WinDigit / ...\n"
            f"print(df.to_csv(sep='\\t', index=True))\n"
        )
        score_out = df.to_csv(sep="\t", index=True)
        nb_blocks.append((score_src, score_out))
        print(score_out, flush=True)

        if rnd == 1:
            if seed_exits:
                pick = seed_exits[:2]
                best = float(df["Final->Copy"].max()) if len(df) and "Final->Copy" in df.columns else 0.0
                floor = best * 0.5 if best > 0 else AV_EXPLAIN_MIN
                ref_head = f"probe{pick[0]}" if pick else "n/a"
            else:
                top2 = df.head(2)
                best = float(df["Final->Copy"].max()) if len(df) and "Final->Copy" in df.columns else 0.0
                floor = best * 0.5 if best > 0 else 0.0
                ref_head = (
                    f"({int(top2.iloc[0]['layer'])}, {int(top2.iloc[0]['head'])})"
                    if len(top2)
                    else "n/a"
                )
                pick = []
                for _, row in top2.iterrows():
                    best_p, best_s = "Final->Copy", 0.0
                    for p in CROSS_SPAN:
                        if p in row and float(row[p]) > best_s:
                            best_p, best_s = p, float(row[p])
                    pick.append((int(row["layer"]), int(row["head"]), best_s, best_p))
            if (not seed_exits) and best < 0.01:
                print(f"[iter] residual max Final->Copy={best:.4f} < 0.01; screening heads", flush=True)
                ranked = [(lh, s) for lh, s in screen_av_heads(results, model) if s >= 0.01][:2]
                if ranked:
                    pick = [(l, h, s, "Final->Copy") for (l, h), s in ranked]
                    floor = max(ranked[0][1] * 0.5, floor)
                    ref_head = f"probe({pick[0][0]},{pick[0][1]})"
                else:
                    print("[iter] probe 全近 0，仍使用残差 top-2", flush=True)
            nodes = []
            for item in pick:
                l, h = int(item[0]), int(item[1])
                sc = float(item[2]) if len(item) > 2 else 0.0
                pat = item[3] if len(item) > 3 else "Final->Copy"
                for ntype in ("attn_k", "attn_q", "attn_v"):
                    key = (l, h, ntype, pat)
                    if key in hung:
                        continue
                    hung.add(key)
                    nodes.append(Node(l, h, ntype, attn_pattern=pat))
                    matches.append((l, h, pat, ntype, sc if sc else max_av, rnd))
                    ledger.append((nodes[-1], rnd))
        else:
            nodes = []
            prefer = [
                ("Final->Copy", "attn_k"),
                ("QLast->WinDigit", "attn_q"),
                ("Final->DemoOut1", "attn_k"),
                ("QLast->DemoQLast", "attn_q"),
                ("Final->QLast", "attn_q"),
                ("Final->WinDigit", "attn_k"),
                ("Final->DemoOutN", "attn_k"),
            ]
            cand = df.copy()
            if "_cross" not in cand.columns and any(c in cand.columns for c in CROSS_SPAN):
                cand["_cross"] = cand[[c for c in CROSS_SPAN if c in cand.columns]].max(axis=1)
                cand = cand.sort_values("_cross", ascending=False)
            for _, row in cand.iterrows():
                if len(nodes) >= 6:
                    break
                l, h = int(row["layer"]), int(row["head"])
                pattern = ntype = None
                score = 0.0
                for p, nt in prefer:
                    if p not in row:
                        continue
                    val = float(row[p])
                    if val >= max(floor, 0.05):
                        pattern, ntype, score = p, nt, val
                        break
                if pattern is None:
                    continue
                key = (l, h, ntype, pattern)
                if key in hung or any(n.layer == l and n.head == h and n.type == ntype for n in nodes):
                    continue
                hung.add(key)
                nodes.append(Node(l, h, ntype, attn_pattern=pattern))
                matches.append((l, h, pattern, ntype, score, rnd))
                ledger.append((nodes[-1], rnd))
            if not nodes:
                stop = "归因停滞（无新匹配且无可用扩展）"
                expand_src = f"# Round {rnd} expand\n# no new nodes; stop={stop}\n"
                nb_blocks.append((expand_src, stop + "\n"))
                sync_workspace_cells(nb_blocks)
                break

        expand_src = (
            f"# Round {rnd} expand\n"
            + "nodes = [\n"
            + "".join(f"    Node({n.layer}, {n.head}, '{n.type}', attn_pattern='{n.attn_pattern}'),\n" for n in nodes)
            + "]\n"
            + "for n in nodes:\n    add_edges(graph, n, tnode.data.nodes, tnode.data.attr)\n"
            + "tnode = add_tnode(_results, selected_model, nodes, parent=tnode)\n"
            + "print_tree(root)\n"
        )
        try:
            _free_cuda()
            for n in nodes:
                add_edges(graph, n, tnode.data.nodes, tnode.data.attr)
            tnode = add_tnode(results, model, nodes, parent=tnode, k=TOP_K)
            tree = capture_tree(root)
            print(tree, flush=True)
            nb_blocks.append((expand_src, tree + "\n"))
        except Exception as e:
            stop = f"扩展失败: {e}"
            print(stop, flush=True)
            traceback.print_exc()
            nb_blocks.append((expand_src, stop + "\n"))
            sync_workspace_cells(nb_blocks)
            break

        top_now = tuple(tnode.data.top_heads.keys()) if hasattr(tnode.data, "top_heads") else None
        if last_top is not None and top_now == last_top:
            stop = "归因停滞（top_heads 未变）"
            sync_workspace_cells(nb_blocks)
            break
        last_top = tuple(tnode.data.top_heads.keys())
        sync_workspace_cells(nb_blocks)
    else:
        stop = f"达到 max_attribution_rounds={MAX_ROUNDS}"

    tree = capture_tree(root)
    av_match = [m for m in matches if m[2] in CROSS_SPAN]
    explain = ("扩展失败" not in stop) and any(m[4] >= AV_EXPLAIN_MIN for m in av_match)

    viz_ok = False
    try:
        write_status("进行中 — 调用 visualize_model_heads（无显示环境可能无 widget）")
        visualize_model_heads(root, model, results, sample=results[0], attn_fn=show_attn_grid)
        viz_ok = True
    except Exception as e:
        print(f"[viz] {e}", flush=True)

    return {
        "root": root,
        "tree": tree,
        "ledger": ledger,
        "matches": matches,
        "floor": floor,
        "ref_head": ref_head,
        "viz_ok": viz_ok,
        "explain": explain,
        "stop": stop,
        "attempt": attempt,
        "n_rounds": n_rounds,
        "max_qki": max_qki,
        "max_av": max_av,
        "acc_str": "",
    }


def main():
    (ROOT / "logs").mkdir(exist_ok=True)
    write_status("进行中 — tmux 驱动脚本启动（LoRA scat_targeted + 同形状队列）")
    global pd, mr, cluster_heads, get_head_matching_scores, eval_head_lens
    global visualize_model_heads, show_attn_grid, print_tree

    from common_utils import mr
    from pptree import print_tree
    from vis import (
        cluster_heads,
        eval_head_lens,
        get_head_matching_scores,
        show_attn_grid,
        visualize_model_heads,
    )
    import pandas as pd

    model, tokenizer = load_model()
    results, acc_str, cohort, counts = build_results(model, tokenizer)
    payload = run_loop(model, results, attempt=1)
    payload["acc_str"] = acc_str
    payload["cohort"] = f"mixed-length from 2000; shapes={dict(counts.most_common(8))}"

    write_report(payload)
    if payload["explain"]:
        write_status("完成 — 报告已写入 attribution_report.md。可 `tmux attach -t btom-scat-attr` 看日志")
    else:
        write_status("无法解释（已迭代）— 详见 attribution_report.md")
    print("DONE", payload["explain"], payload["stop"], "max_av", payload.get("max_av"), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        write_status(f"失败 — 见 logs/scat_attr_tmux.log\n```\n{traceback.format_exc()[-1500:]}\n```")
        raise
