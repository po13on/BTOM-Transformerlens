#!/usr/bin/env python3
"""Exact-match grid accuracy on move_to_match test split."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DATA_ROOT = Path("/home/hushengchun/project/BTOM-transformerlens/data")
MODEL_PATH = "/data0/modelscope/qwen3/models/Qwen/Qwen3-32B"
SEP = 10

SYSTEM_PROMPT = """You solve a three-candidate grid puzzle.

Each puzzle has one hidden rule shared by all of its examples. Every example shows:
- three 3x3 candidate grids (A, B, C)
- a smaller query grid
- for the three training examples, the correct output grid

How to solve:
1. Compare the query with A/B/C and decide which candidate matches.
2. From the winning candidate, extract a small value grid using the same rule as the examples.
3. Emit only that value grid. Cells are colors 0-9.

Color ids: 0 black, 1 blue, 2 red, 3 green, 4 yellow, 5 grey, 6 pink, 7 orange, 8 teal, 9 maroon.
Color 10 is a canvas separator/padding only. It never appears in candidates, query, or the answer. Never copy 10.

Reply with the test output grid alone: one row per line, integers separated by a single space. No labels, no candidate letter, no explanation."""


def grid_to_text(grid) -> str:
    return "\n".join(" ".join(str(int(v)) for v in row) for row in grid)


def parse_grid(text: str):
    rows = []
    for line in text.strip().splitlines():
        line = line.strip().replace(",", " ")
        if not line:
            if rows:
                break
            continue
        parts = line.split()
        if not parts or not all(p.lstrip("-").isdigit() for p in parts):
            if rows:
                break
            continue
        rows.append([int(p) for p in parts])
    return rows


def _trim_query(block):
    rows = [list(map(int, row)) for row in block]
    while rows and all(v == SEP for v in rows[0]):
        rows.pop(0)
    while rows and all(v == SEP for v in rows[-1]):
        rows.pop()
    cols = list(zip(*rows)) if rows else []
    while cols and all(v == SEP for v in cols[0]):
        cols.pop(0)
    while cols and all(v == SEP for v in cols[-1]):
        cols.pop()
    return [list(row) for row in zip(*cols)] if cols else rows


def split_canvas(grid):
    arr = [list(map(int, row)) for row in grid]
    n_rows = len(arr)
    n_cols = len(arr[0]) if arr else 0
    sep_cols = [c for c in range(n_cols) if all(arr[r][c] == SEP for r in range(n_rows))]
    blocks, start = [], 0
    for c in sep_cols + [n_cols]:
        if c > start:
            blocks.append([row[start:c] for row in arr])
        start = c + 1
    if len(blocks) >= 4:
        return blocks[:3], _trim_query(blocks[3])
    return None, None


def format_panels(grid) -> str:
    cands, query = split_canvas(grid)
    if cands is None:
        raise ValueError("bad canvas")
    parts = [f"Candidate {name}:\n{grid_to_text(g)}" for name, g in zip("ABC", cands)]
    parts.append(f"Query:\n{grid_to_text(query)}")
    return "\n\n".join(parts)


def format_example_block(title: str, inp, output=None) -> str:
    body = f"{title}\n{format_panels(inp)}"
    if output is None:
        return body + "\n\nOutput:"
    return body + f"\n\nOutput:\n{grid_to_text(output)}"


def task_to_prompt_parts(task: dict, gold):
    chunks = [
        format_example_block(f"Example {i}", ex["input"], ex["output"])
        for i, ex in enumerate(task["train"], 1)
    ]
    chunks.append(format_example_block("Test", task["test"][0]["input"]))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(chunks)},
    ]
    return messages, gold


def looks_like_grid(rows) -> bool:
    if not rows or not rows[0]:
        return False
    w = len(rows[0])
    return w > 0 and all(len(r) == w for r in rows) and all(0 <= v <= 9 for r in rows for v in r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    folder = DATA_ROOT / args.dataset
    challenges = json.loads((folder / "test_challenges.json").read_text())
    solutions = json.loads((folder / "test_solutions.json").read_text())
    items = []
    for tid, task in challenges.items():
        if tid not in solutions:
            continue
        items.append((tid, task, solutions[tid][0]))
    if args.limit:
        items = items[: args.limit]
    print(f"dataset={args.dataset} n_test={len(items)} adapter={args.adapter}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    chat_kw = {}
    try:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        chat_kw["enable_thinking"] = False
    except TypeError:
        pass

    prompts, golds, tids = [], [], []
    for tid, task, gold in items:
        messages, g = task_to_prompt_parts(task, gold)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_kw
        )
        prompts.append(text)
        golds.append(g)
        tids.append(tid)

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print("loading base + adapter ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    print("ready", type(model).__name__, flush=True)

    n_exact = n_fmt = n_done = 0
    records = []
    t0 = time.time()
    bs = args.batch_size
    for start in range(0, len(prompts), bs):
        batch_p = prompts[start : start + bs]
        batch_g = golds[start : start + bs]
        batch_id = tids[start : start + bs]
        enc = tokenizer(
            batch_p,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        in_len = enc["input_ids"].shape[1]
        for i, seq in enumerate(out):
            pred_text = tokenizer.decode(seq[in_len:], skip_special_tokens=True)
            pred = parse_grid(pred_text)
            gold = [[int(x) for x in row] for row in batch_g[i]]
            fmt = looks_like_grid(pred)
            exact = pred == gold
            n_fmt += int(fmt)
            n_exact += int(exact)
            n_done += 1
            records.append(
                {
                    "id": batch_id[i],
                    "exact": exact,
                    "format_ok": fmt,
                    "pred": pred,
                    "gold": gold,
                    "raw": pred_text[:200],
                }
            )
        if n_done % 50 < bs or n_done == len(prompts):
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed else 0
            print(
                f"[{n_done}/{len(prompts)}] exact={n_exact}/{n_done}="
                f"{n_exact/n_done:.4f} format={n_fmt/n_done:.4f} "
                f"{rate:.2f} ex/s",
                flush=True,
            )
            metrics = {
                "dataset": args.dataset,
                "n": n_done,
                "n_total": len(prompts),
                "exact": n_exact,
                "exact_acc": n_exact / n_done,
                "format_ok": n_fmt,
                "format_acc": n_fmt / n_done,
                "seconds": elapsed,
            }
            Path(args.out_json).write_text(json.dumps({"metrics": metrics, "records": records}, ensure_ascii=False))

    elapsed = time.time() - t0
    metrics = {
        "dataset": args.dataset,
        "n": n_done,
        "n_total": len(prompts),
        "exact": n_exact,
        "exact_acc": n_exact / n_done,
        "format_ok": n_fmt,
        "format_acc": n_fmt / n_done,
        "seconds": elapsed,
    }
    Path(args.out_json).write_text(json.dumps({"metrics": metrics, "records": records}, ensure_ascii=False, indent=2))
    print("FINAL", json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
