#!/usr/bin/env python3
"""把 move_to_match 的 challenges/solutions JSON 转成「自带 prompt + 唯一金标网格」的数据集。

每条样本只有一个标准答案（输出网格），没有选项。默认写出 jsonl，模型直接读 prompt 字段。

示例：
  python export_mtm_csv.py data/scat_targeted/test_challenges.json
  python export_mtm_csv.py data/dual_scat/test_new_challenges.json -o data_uniform/dual_scat_test_new.jsonl
  python export_mtm_csv.py data/scat_targeted/test_challenges.json --limit 20
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import re
from pathlib import Path

SEP = 10
DEFAULT_TOKENIZER = "/data0/modelscope/qwen3/models/Qwen/Qwen3-32B"
PROJECT_ROOT = Path(__file__).resolve().parent

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
    n_rows, n_cols = len(arr), (len(arr[0]) if arr else 0)
    seps = [c for c in range(n_cols) if all(arr[r][c] == SEP for r in range(n_rows))]
    blocks, start = [], 0
    for c in seps + [n_cols]:
        if c > start:
            blocks.append([row[start:c] for row in arr])
        start = c + 1
    if len(blocks) >= 4:
        return blocks[:3], _trim_query(blocks[3])
    return None, None


def format_panels(grid) -> str:
    cands, query = split_canvas(grid)
    if cands is None:
        raise ValueError("canvas 无法拆成 3 候选 + query")
    parts = [f"Candidate {n}:\n{grid_to_text(g)}" for n, g in zip("ABC", cands)]
    parts.append(f"Query:\n{grid_to_text(query)}")
    return "\n\n".join(parts)


def format_block(title, inp, output=None) -> str:
    body = f"{title}\n{format_panels(inp)}"
    if output is None:
        return body + "\n\nOutput:"
    return body + f"\n\nOutput:\n{grid_to_text(output)}"


def task_user_content(task: dict) -> str:
    chunks = [format_block(f"Example {i}", ex["input"], ex["output"]) for i, ex in enumerate(task["train"], 1)]
    chunks.append(format_block("Test", task["test"][0]["input"]))
    return "\n\n".join(chunks)


def infer_solutions_path(challenges: Path) -> Path:
    name = challenges.name
    if "_challenges.json" not in name:
        raise ValueError(f"无法从 {challenges} 推断 solutions 路径，请显式传入 --solutions")
    return challenges.with_name(name.replace("_challenges.json", "_solutions.json"))


def chat_kwargs(tokenizer) -> dict:
    kw = {}
    try:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        kw["enable_thinking"] = False
    except TypeError:
        pass
    return kw


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def token_index_at(tokenizer, prompt: str, needle: str, *, last: bool = True) -> int | None:
    pos = prompt.rfind(needle) if last else prompt.find(needle)
    if pos < 0:
        return None
    return token_len(tokenizer, prompt[:pos])


def build_spans(tokenizer, prompt: str) -> dict:
    """token 下标：start/Test 指向 Test 段，gen 是第一个待生成 token 的位置。"""
    test_at = prompt.rfind("\nTest\n")
    start = token_len(tokenizer, prompt[:test_at]) if test_at >= 0 else 0
    spans = {"start": start, "gen": token_len(tokenizer, prompt)}
    for key, needle, last in [
        ("Test", "\nTest\n", True),
        ("Query", "\nQuery:\n", True),
        ("A", "Candidate A:\n", False),
        ("B", "Candidate B:\n", False),
        ("C", "Candidate C:\n", False),
    ]:
        pos = token_index_at(tokenizer, prompt, needle, last=last)
        if pos is not None:
            spans[key] = pos
    return spans


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    tok.pad_token = tok.pad_token or tok.eos_token
    return tok


def make_record(tid: str, dataset: str, task: dict, gold, tokenizer, ck: dict) -> dict:
    user = task_user_content(task)
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
        **ck,
    )
    answer_grid = [[int(v) for v in row] for row in gold]
    task_out = copy.deepcopy(task)
    if not task_out.get("test"):
        raise ValueError("task 缺少 test")
    task_out["test"][0]["output"] = answer_grid
    spans = build_spans(tokenizer, prompt)
    spans.update(locate_attribution_spans(prompt, task_out, tokenizer))
    return {
        "task_id": tid,
        "dataset": dataset,
        "prompt": prompt,
        "answer": grid_to_text(answer_grid),
        "answer_grid": answer_grid,
        "n_tokens": token_len(tokenizer, prompt),
        "spans": spans,
        "task": task_out,
    }


def default_out_path(dataset: str, fmt: str) -> Path:
    ext = "csv" if fmt == "csv" else "jsonl"
    return PROJECT_ROOT / "data_uniform" / f"{dataset}.{ext}"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["task_id", "dataset", "prompt", "answer", "answer_grid", "n_tokens", "spans", "task"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "task_id": row["task_id"],
                    "dataset": row["dataset"],
                    "prompt": row["prompt"],
                    "answer": row["answer"],
                    "answer_grid": json.dumps(row["answer_grid"], ensure_ascii=False),
                    "n_tokens": row["n_tokens"],
                    "spans": json.dumps(row["spans"], ensure_ascii=False),
                    "task": json.dumps(row["task"], ensure_ascii=False),
                }
            )


def _parse_json_field(value, default=None):
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        import ast

        return ast.literal_eval(value)


def _char_to_tok(tokenizer, prompt: str, char_i: int) -> int:
    return len(tokenizer.encode(prompt[:char_i], add_special_tokens=False))


def task_io_shapes(task: dict) -> dict | None:
    """Query / 金标 Output 的网格形状。变长只来自这两块，不来自 A/B/C。"""
    test0 = (task.get("test") or [{}])[0]
    gold = test0.get("output")
    _, query = split_canvas(test0.get("input") or [])
    if query is None or gold is None:
        return None
    qh, qw = len(query), (len(query[0]) if query else 0)
    oh, ow = len(gold), (len(gold[0]) if gold else 0)
    return {
        "q": (qh, qw),
        "o": (oh, ow),
        "q_cells": qh * qw,
        "o_cells": oh * ow,
        "query": query,
        "gold": gold,
        "cohort": (qh, qw, oh, ow),
    }


def _decode_one(tokenizer, tok_id) -> str:
    return tokenizer.decode([int(tok_id)]).strip().strip('"').strip("'")


def verify_span_tokens(tokenizer, prompt: str, spans: dict, gold, query) -> tuple[bool, list[str]]:
    """核对 Final / Copy / QLast / WinDigit 是否落在预期数字 token 上。"""
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    n = len(ids)
    reasons: list[str] = []
    a = spans.get("Final", spans.get("A-"))
    if a != n - 1:
        reasons.append(f"Final={a} vs last={n - 1}")
    v = spans.get("Copy")
    want_v = str(int(gold[0][0])) if gold else None
    if v is None or not (0 <= int(v) < n):
        reasons.append("Copy missing/OOB")
    elif want_v is not None and _decode_one(tokenizer, ids[int(v)]) != want_v:
        reasons.append(f"Copy tok {_decode_one(tokenizer, ids[int(v)])!r} != gold[0,0]={want_v}")
    last_q = None
    if query:
        for row in query:
            for cell in row:
                last_q = str(int(cell))
    q = spans.get("QLast")
    if q is None or not (0 <= int(q) < n):
        reasons.append("QLast missing/OOB")
    elif last_q is not None and _decode_one(tokenizer, ids[int(q)]) != last_q:
        reasons.append(f"QLast tok {_decode_one(tokenizer, ids[int(q)])!r} != query_last={last_q}")
    win = spans.get("WinDigit")
    if win is not None and last_q is not None:
        if not (0 <= int(win) < n):
            reasons.append("WinDigit OOB")
        elif _decode_one(tokenizer, ids[int(win)]) != last_q:
            reasons.append(
                f"WinDigit tok {_decode_one(tokenizer, ids[int(win)])!r} != query_last={last_q}"
            )
    return (len(reasons) == 0), reasons


def find_output_subgrid(cand, gold):
    """在 3×3 候选里找与金标完全重合的子块，返回 [(r, c), ...]。"""
    ch, cw = len(cand), len(cand[0])
    gh, gw = len(gold), len(gold[0])
    hits = []
    for r in range(ch - gh + 1):
        for c in range(cw - gw + 1):
            if all(int(cand[r + i][c + j]) == int(gold[i][j]) for i in range(gh) for j in range(gw)):
                hits.append((r, c))
    return hits


def _grid_cell_char(block: str, row: int, col: int) -> int | None:
    offset = 0
    grid_row = 0
    for line in block.splitlines(keepends=True):
        parts = line.replace(",", " ").split()
        if parts and all(p.lstrip("-").isdigit() for p in parts):
            matches = list(re.finditer(r"-?\d+", line))
            if grid_row == row and col < len(matches):
                return offset + matches[col].start()
            grid_row += 1
        elif grid_row:
            break
        offset += len(line)
    return None


_DIGIT_RE = re.compile(r"(?<!\d)\d(?!\d)")

# Token-index keys that shift when TransformerLens prepends BOS.
# Do not start with A/V/QK (get_ranges_tom Hi-ToM prefixes).
# DemoOut1- is stored as an exact pre-write index; get_ranges_tom must not
# extra-shift exact keys (see attribute.get_ranges_tom).
SCAT_SHIFT_KEYS = (
    "Copy",
    "QLast",
    "WinDigit",
    "WinDigit0",
    "DemoOut1",
    "DemoOut1-",
    "DemoOutN",
    "DemoQLast",
    "DemoCopy",
    "Demo1OutN",
    "Demo2OutN",
    "Demo1QLast",
    "Demo2QLast",
    "Demo1Out1",
    "Demo1Out1-",
    "Demo2Out1",
    "Demo2Out1-",
    "Demo1Beg",
    "Demo2Beg",
    "Demo3Beg",
    "TestBeg",
    "start",
)

# Final is reset to seq-1 after to_tokens; CandDigits is a list.
SCAT_ROLE_KEYS = SCAT_SHIFT_KEYS + ("Final", "A-")


def _section_starts(prompt: str) -> dict:
    starts = {}
    for i in (1, 2, 3):
        needle = f"\nExample {i}\n"
        at = prompt.find(needle)
        if at < 0:
            at = prompt.find(f"Example {i}\n")
        starts[i] = at
    starts["Test"] = prompt.rfind("\nTest\n")
    return starts


def _first_last_digit_tok(tokenizer, prompt: str, start: int, end: int):
    body = prompt[start:end]
    digits = list(_DIGIT_RE.finditer(body))
    if not digits:
        return None, None
    first = _char_to_tok(tokenizer, prompt, start + digits[0].start())
    last = _char_to_tok(tokenizer, prompt, start + digits[-1].start())
    return first, last


def _query_output_char_spans(block: str, abs0: int):
    """Absolute char [start, end) of Query body and Output body inside one Example/Test block."""
    q_rel = block.rfind("\nQuery:\n")
    o_rel = block.rfind("\nOutput:\n")
    if q_rel < 0:
        return None, None
    q_start = abs0 + q_rel + len("\nQuery:\n")
    q_end = abs0 + (o_rel if o_rel > q_rel else len(block))
    out = None
    if o_rel >= 0:
        out = (abs0 + o_rel + len("\nOutput:\n"), abs0 + len(block))
    return (q_start, q_end), out


def _pick_unique_or_first(cands, needle):
    winner, origin = None, None
    for name, cand in zip("ABC", cands):
        hits = find_output_subgrid(cand, needle)
        if len(hits) == 1:
            winner, origin = name, hits[0]
            break
    if winner is None:
        for name, cand in zip("ABC", cands):
            hits = find_output_subgrid(cand, needle)
            if hits:
                winner, origin = name, hits[0]
                break
    return winner, origin


def _cand_cell_tok(tokenizer, prompt: str, section_abs: int, section: str, name: str, row: int, col: int):
    marker = f"Candidate {name}:\n"
    c_rel = section.find(marker)
    if c_rel < 0:
        return None
    block_start = section_abs + c_rel + len(marker)
    cell_rel = _grid_cell_char(prompt[block_start:], row, col)
    if cell_rel is None:
        return None
    return _char_to_tok(tokenizer, prompt, block_start + cell_rel)


def locate_attribution_spans(prompt: str, task: dict, tokenizer) -> dict:
    """散向靶点角色（不用 Hi-ToM 的 QK_I / VK_* 名）。

    - Final：prompt 最后一 token（即将写答案）
    - Copy：Test 赢家抽出块左上角（要拷的第一位）
    - QLast：Test Query 最后一个数字
    - WinDigit：赢家 3×3 里等于 QLast 的最后一格
    - DemoOut1 / DemoOutN：最后一个 example 的 Output 首/末位数字
    - DemoOut1-：DemoOut1 的前一个 token（与 Final 同类：写首位之前）
    - DemoQLast：最后一个 example 的 Query 末位
    - CandDigits：Test A/B/C 里所有等于 QLast 的格
    """
    n_tok = _char_to_tok(tokenizer, prompt, len(prompt))
    last = max(n_tok - 1, 0)
    spans = {"Final": last, "A-": last}
    starts = _section_starts(prompt)
    test_at = starts.get("Test", -1)
    if test_at < 0:
        return spans
    test_part = prompt[test_at:]
    for k, label in ((1, "Demo1Beg"), (2, "Demo2Beg"), (3, "Demo3Beg")):
        if starts.get(k, -1) >= 0:
            spans[label] = _char_to_tok(tokenizer, prompt, starts[k])
    spans["TestBeg"] = _char_to_tok(tokenizer, prompt, test_at)

    bounds = {
        1: (starts.get(1, -1), starts.get(2, test_at)),
        2: (starts.get(2, -1), starts.get(3, test_at)),
        3: (starts.get(3, -1), test_at),
    }
    train = task.get("train") or []
    for k, (a, b) in bounds.items():
        if a is None or a < 0 or b is None or b <= a:
            continue
        block = prompt[a:b]
        qspan, ospan = _query_output_char_spans(block, a)
        if qspan:
            _, ql = _first_last_digit_tok(tokenizer, prompt, qspan[0], qspan[1])
            if ql is not None:
                spans[f"Demo{k}QLast"] = ql
                if k == 3:
                    spans["DemoQLast"] = ql
        if ospan:
            ov, ol = _first_last_digit_tok(tokenizer, prompt, ospan[0], ospan[1])
            if ov is not None:
                spans[f"Demo{k}Out1"] = ov
                if ov >= 1:
                    spans[f"Demo{k}Out1-"] = ov - 1
                if k == 3:
                    spans["DemoOut1"] = ov
                    if ov >= 1:
                        spans["DemoOut1-"] = ov - 1
            if ol is not None:
                spans[f"Demo{k}OutN"] = ol
                if k == 3:
                    spans["DemoOutN"] = ol

    if len(train) >= 3:
        ex3 = train[2]
        cands3, _ = split_canvas(ex3.get("input") or [])
        gold3 = ex3.get("output")
        if cands3 is not None and gold3:
            w3, origin3 = _pick_unique_or_first(cands3, gold3)
            if w3 is not None and starts.get(3, -1) >= 0:
                tok = _cand_cell_tok(
                    tokenizer, prompt, starts[3], prompt[starts[3]:test_at], w3, origin3[0], origin3[1]
                )
                if tok is not None:
                    spans["DemoCopy"] = tok
                    spans["ex3_winner"] = w3

    q_rel = test_part.rfind("\nQuery:\n")
    query = None
    gold = task["test"][0].get("output")
    cands, query = split_canvas(task["test"][0]["input"])
    if q_rel >= 0:
        q_body = test_part[q_rel + len("\nQuery:\n") :]
        digits = list(_DIGIT_RE.finditer(q_body.split("Output:")[0]))
        if digits:
            spans["QLast"] = _char_to_tok(
                tokenizer, prompt, test_at + q_rel + len("\nQuery:\n") + digits[-1].start()
            )

    if gold is None or cands is None:
        return spans
    winner, origin = _pick_unique_or_first(cands, gold)
    if winner is None:
        return spans
    tok_v = _cand_cell_tok(tokenizer, prompt, test_at, test_part, winner, origin[0], origin[1])
    if tok_v is not None:
        spans["Copy"] = tok_v
        spans["winner"] = winner
        spans["value_cell"] = list(origin)

    if query:
        last_q = int(query[-1][-1])
        # 散向靶点：Query 往往不是赢家里的连续子块。WinDigit = 赢家中
        # 与 Query 末位同数字的最后一格（行优先）；CandDigits = Test A/B/C 全部同数字格。
        if winner is not None:
            wcand = cands["ABC".index(winner)]
            same = [
                (r, c)
                for r, row in enumerate(wcand)
                for c, val in enumerate(row)
                if int(val) == last_q
            ]
            if same:
                tok_qf = _cand_cell_tok(tokenizer, prompt, test_at, test_part, winner, *same[0])
                tok_ql = _cand_cell_tok(tokenizer, prompt, test_at, test_part, winner, *same[-1])
                if tok_qf is not None:
                    spans["WinDigit0"] = tok_qf
                if tok_ql is not None:
                    spans["WinDigit"] = tok_ql
                spans["win_q_cell"] = list(same[-1])

        cand_q = []
        for name, cand in zip("ABC", cands):
            for r, row in enumerate(cand):
                for c, val in enumerate(row):
                    if int(val) != last_q:
                        continue
                    tok = _cand_cell_tok(tokenizer, prompt, test_at, test_part, name, r, c)
                    if tok is not None:
                        cand_q.append(tok)
        if cand_q:
            spans["CandDigits"] = cand_q
    return spans


def load_prompt_dataset(path: str | Path) -> list[dict]:
    """读回本脚本导出的 jsonl / csv。CSV 里的 task / answer_grid / spans 会反序列化。"""
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["answer_grid"] = _parse_json_field(row.get("answer_grid"))
            row["n_tokens"] = int(row["n_tokens"])
            row["spans"] = _parse_json_field(row.get("spans"), default={})
            if row.get("task"):
                row["task"] = _parse_json_field(row["task"])
            rows.append(row)
    return rows


def iter_viz_tasks(path: str | Path):
    """画布可视化入口：yield ``(task_id, task)``。

    - ``.jsonl`` / ``.csv``：用 ``load_prompt_dataset``，直接取嵌入的 ``row["task"]``
      （含 train/test canvas，金标已写在 ``task.test[0].output``）。
    - ``*_challenges.json``：走旧逻辑，并把 sibling ``*_solutions.json`` 的金标写进
      ``task.test[0].output``。
    """
    path = Path(path)
    if path.suffix in {".jsonl", ".csv"}:
        for row in load_prompt_dataset(path):
            yield row["task_id"], row["task"]
        return
    challenges = json.loads(path.read_text(encoding="utf-8"))
    solutions = {}
    try:
        sol_path = infer_solutions_path(path)
        if sol_path.exists():
            solutions = json.loads(sol_path.read_text(encoding="utf-8"))
    except ValueError:
        pass
    for tid, task in challenges.items():
        task = copy.deepcopy(task)
        gold = solutions.get(tid)
        if gold is not None and task.get("test"):
            task["test"][0]["output"] = gold[0]
        yield tid, task


def convert(
    challenges_path: Path,
    solutions_path: Path | None = None,
    out_path: Path | None = None,
    tokenizer_path: str = DEFAULT_TOKENIZER,
    limit: int = 0,
    dataset_name: str | None = None,
    fmt: str = "jsonl",
) -> Path:
    challenges_path = challenges_path.resolve()
    solutions_path = (solutions_path or infer_solutions_path(challenges_path)).resolve()
    dataset_name = dataset_name or challenges_path.parent.name
    if out_path is None:
        out_path = default_out_path(dataset_name, fmt)
    else:
        out_path = out_path.resolve()
        if fmt == "auto":
            fmt = "csv" if out_path.suffix == ".csv" else "jsonl"
    if fmt == "auto":
        fmt = "jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    challenges = json.loads(challenges_path.read_text())
    solutions = json.loads(solutions_path.read_text())
    tokenizer = load_tokenizer(tokenizer_path)
    ck = chat_kwargs(tokenizer)

    rows = []
    n_skip = 0
    for tid, task in challenges.items():
        if tid not in solutions:
            n_skip += 1
            continue
        try:
            gold = solutions[tid][0]
            rows.append(make_record(tid, dataset_name, task, gold, tokenizer, ck))
        except (ValueError, IndexError, TypeError) as e:
            print(f"[skip] {tid}: {e}")
            n_skip += 1
            continue
        if limit and len(rows) >= limit:
            break

    if fmt == "csv":
        write_csv(out_path, rows)
    else:
        write_jsonl(out_path, rows)
    print(f"wrote {len(rows)} rows -> {out_path} (skipped {n_skip})")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Export move_to_match JSON to a prompt+answer dataset.")
    ap.add_argument("challenges", type=Path, help="*_challenges.json 路径")
    ap.add_argument("--solutions", type=Path, default=None, help="默认把 _challenges 换成 _solutions")
    ap.add_argument("-o", "--out", type=Path, default=None, help="输出路径；默认 data_uniform/<dataset>.jsonl")
    ap.add_argument("--format", dest="fmt", choices=["jsonl", "csv", "auto"], default="auto")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="用于 chat template 和 token 计数")
    ap.add_argument("--dataset-name", default=None, help="写入 dataset 字段；默认用 challenges 所在目录名")
    ap.add_argument("--limit", type=int, default=0, help="只导出前 N 条；0 = 全部")
    args = ap.parse_args()
    fmt = args.fmt
    if fmt == "auto" and args.out is not None:
        fmt = "csv" if args.out.suffix == ".csv" else "jsonl"
    elif fmt == "auto":
        fmt = "jsonl"
    convert(
        args.challenges,
        solutions_path=args.solutions,
        out_path=args.out,
        tokenizer_path=args.tokenizer,
        limit=args.limit,
        dataset_name=args.dataset_name,
        fmt=fmt,
    )


if __name__ == "__main__":
    main()
