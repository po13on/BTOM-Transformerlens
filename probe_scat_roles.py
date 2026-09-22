#!/usr/bin/env python3
"""Probe cross-span attention roles on scat_targeted (not V->V / A-->A-).

Roles
-----
A-  -> EX3_OL     last demo Output last digit (rule-writeout hypothesis)
A-  -> EX3_V      last demo Output first digit
A-  -> EX3_QL     last demo Query last digit
A-  -> EX3_WIN_V  last demo's V-analogue inside the winning candidate
A-  -> EX2_OL / EX1_OL
A-  -> WIN_QL     Test winner's query-match last cell
A-  -> V          copy first answer digit (existing)
QK_I -> WIN_QL    Query last digit points at that digit in the winner
QK_I -> CAND_Q    same digit anywhere in Test A/B/C
QK_I -> EX3_QL    analogical query-to-query
QK_I -> V
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
os.environ.setdefault("MODELSCOPE_CACHE", "/data0/modelscope/AI-ModelScope")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path("/home/hushengchun/project/BTOM-transformerlens")
sys.path.insert(0, str(ROOT))

REPORT = ROOT / "role_probe_report.md"
JSON_OUT = ROOT / "role_probe.json"
TZ = timezone(timedelta(hours=8))

TARGET_CORRECT = 6
MAX_FORWARD = 24
LAYER_FROM = 24

# (query_role, key_role) — single-index keys only except CAND_Q handled separately
EDGES = [
    ("A-", "EX3_OL"),
    ("A-", "EX3_V"),
    ("A-", "EX3_QL"),
    ("A-", "EX3_WIN_V"),
    ("A-", "EX2_OL"),
    ("A-", "EX1_OL"),
    ("A-", "WIN_QL"),
    ("A-", "V"),
    ("A-", "QK_I"),
    ("QK_I", "WIN_QL"),
    ("QK_I", "WIN_QF"),
    ("QK_I", "WIN_QBLK"),
    ("QK_I", "EX3_QL"),
    ("QK_I", "V"),
    ("QK_I", "EX3_OL"),
]


def now_str():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")


def _q_pos(r, key: str) -> int:
    m = r.index_map[0]
    if key == "A-":
        return int(m["A-"])
    return int(m[key])


def _row_at(aw, qpos):
    """Attention from query token ``qpos``. TL returns [bsz, H, Q, K]; do not use Q=0."""
    if aw.ndim == 4:
        q = max(0, min(int(qpos), aw.shape[2] - 1))
        return aw[0, :, q]
    return aw[0]


def _mass_at(row, positions):
    import torch

    if not positions:
        return torch.zeros(row.shape[0], device=row.device)
    idx = torch.as_tensor(positions, device=row.device, dtype=torch.long)
    idx = idx.clamp(0, row.shape[-1] - 1)
    return row.index_select(-1, idx).sum(-1)


def main():
    import torch
    from model_hooks import get_attn_weights
    from export_mtm_csv import _decode_one
    import run_scat_targeted_attr as driver

    driver.TARGET_CORRECT = TARGET_CORRECT
    driver.MAX_FORWARD = MAX_FORWARD
    driver.write_status = lambda text: print(f"[status] {text}", flush=True)

    print(f"[probe] start {now_str()} GPU={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    model, tokenizer = driver.load_model()
    results, acc_str, cohort, counts = driver.build_results(model, tokenizer)
    print(f"[probe] kept {len(results)} {acc_str} cohort={cohort}", flush=True)

    samples_meta = []
    for r in results:
        m = r.index_map[0]
        ids = model.to_tokens(r.prompt, prepend_bos=False, padding_side="left")[0]
        rec = {
            "n_tokens": r.n_tokens,
            "winner": m.get("winner"),
            "roles": {},
        }
        for k in (
            "A-",
            "V",
            "QK_I",
            "EX3_OL",
            "EX3_V",
            "EX3_QL",
            "EX3_WIN_V",
            "EX2_OL",
            "EX1_OL",
            "WIN_QL",
            "WIN_QF",
        ):
            if k not in m:
                continue
            p = int(m[k])
            rec["roles"][k] = {"pos": p, "tok": _decode_one(tokenizer, ids[p])}
        rec["n_cand_q"] = len(m.get("CAND_Q") or [])
        samples_meta.append(rec)
        print(f"  sample {r.index} winner={m.get('winner')} roles={rec['roles']}", flush=True)

    H = model.cfg.n_heads
    L0, L1 = LAYER_FROM, model.cfg.n_layers
    acc = defaultdict(lambda: torch.zeros(L1 - L0, H))
    n_ok = defaultdict(int)
    region = defaultdict(lambda: torch.zeros(L1 - L0, H))  # A- mass on EX1/EX2/EX3/TEST

    for r in results:
        m = r.index_map[0]
        q_cache = {"A-": _q_pos(r, "A-")}
        if "QK_I" in m:
            q_cache["QK_I"] = _q_pos(r, "QK_I")
        # Pattern Score 的 A- 会再 -1；空 think 后缀时真正写答案的前一格也要看
        q_cache["Am"] = max(int(q_cache["A-"]) - 1, 0)

        for l in range(L0, L1):
            li = l - L0
            aw = get_attn_weights(model, r, l, head=None, pos_ids=None, use_cache=False)
            rows = {qk: _row_at(aw, qpos) for qk, qpos in q_cache.items()}
            for qk, sk in EDGES:
                if qk not in rows or sk not in m:
                    continue
                mass = _mass_at(rows[qk], [int(m[sk])])
                acc[(qk, sk)][li] += mass.detach().cpu()
                n_ok[(qk, sk)] += 1
            if "Am" in rows:
                for sk in ("EX3_OL", "EX3_V", "EX3_QL", "EX3_WIN_V", "WIN_QL", "V", "QK_I"):
                    if sk not in m:
                        continue
                    mass = _mass_at(rows["Am"], [int(m[sk])])
                    acc[("Am", sk)][li] += mass.detach().cpu()
                    n_ok[("Am", sk)] += 1
            if "QK_I" in rows and m.get("CAND_Q"):
                mass = _mass_at(rows["QK_I"], [int(x) for x in m["CAND_Q"]])
                acc[("QK_I", "CAND_Q")][li] += mass.detach().cpu()
                n_ok[("QK_I", "CAND_Q")] += 1
            if "A-" in rows:
                a_pos = int(q_cache["A-"])
                for name, a, b in (
                    ("EX1", m.get("EX1_BEG"), m.get("EX2_BEG")),
                    ("EX2", m.get("EX2_BEG"), m.get("EX3_BEG")),
                    ("EX3", m.get("EX3_BEG"), m.get("TEST_BEG")),
                    ("TEST", m.get("TEST_BEG"), a_pos + 1),
                ):
                    if a is None or b is None:
                        continue
                    lo, hi = int(a), int(b)
                    if hi <= lo:
                        continue
                    region[name][li] += _mass_at(rows["A-"], list(range(lo, hi))).detach().cpu()
            del aw
            driver._free_cuda()
        print(f"[probe] done sample {r.index} {driver.gpu_mem()}", flush=True)

    n_s = max(len(results), 1)
    ranked = {}
    for edge, mat in acc.items():
        n = max(n_ok[edge] // max(L1 - L0, 1), 1)
        mean = (mat / n).tolist()
        flat = []
        for li, row in enumerate(mean):
            for h, s in enumerate(row):
                flat.append(((L0 + li, h), float(s)))
        flat.sort(key=lambda x: -x[1])
        ranked[f"{edge[0]}->{edge[1]}"] = {
            "n": n,
            "top": [(f"({l},{h})", round(s, 4)) for (l, h), s in flat[:8]],
            "best": flat[0][1] if flat else 0.0,
        }

    region_top = {}
    for name, mat in region.items():
        mean = (mat / n_s).tolist()
        flat = []
        for li, row in enumerate(mean):
            for h, s in enumerate(row):
                flat.append(((L0 + li, h), float(s)))
        flat.sort(key=lambda x: -x[1])
        region_top[name] = [(f"({l},{h})", round(s, 4)) for (l, h), s in flat[:5]]

    payload = {
        "time": now_str(),
        "acc_str": acc_str,
        "cohort": list(cohort) if cohort else None,
        "n_samples": len(results),
        "layer_from": L0,
        "samples": samples_meta,
        "edges": ranked,
        "a_region_top": region_top,
    }
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# scat_targeted 跨位置角色探测",
        "",
        f"生成时间：{now_str()} (UTC+8)",
        f"样本：{acc_str}；队列 {cohort}；层 L{L0}..{L1 - 1}",
        "",
        "## 角色定义",
        "",
        "| key | 位置 | 解释 |",
        "|-----|------|------|",
        "| EX3_OL | 最后一个 example 的 Output 最后一位 | A- 读演示规则写完处（优先假设） |",
        "| EX3_V / EX3_QL | 最后 example 的 Output 首位 / Query 末位 | 备选规则位置 |",
        "| EX3_WIN_V | 最后 example 赢家抽出块左上角 | demo 版的 V |",
        "| WIN_QL | Test 赢家中与 Query 同形子块的最后一格 | Query 末位应对齐的那个数字 |",
        "| CAND_Q | Test A/B/C 里所有等于 Query 末位的格 | 未消歧的指针 |",
        "",
        "## 各边最强头（对指定 key 的注意力质量）",
        "",
    ]
    for name, info in sorted(ranked.items(), key=lambda kv: -kv[1]["best"]):
        top = ", ".join(f"{lh}={s}" for lh, s in info["top"][:5])
        lines.append(f"- **{name}** (n={info['n']}, best={info['best']:.4f}): {top}")
    lines += [
        "",
        "## A- 落在哪一段 example / Test（区域质量和，不是单格）",
        "",
    ]
    for name, tops in region_top.items():
        lines.append(f"- **{name}**: " + ", ".join(f"{lh}={s}" for lh, s in tops))
    lines += [
        "",
        "## 样本角色 token 核对",
        "",
    ]
    for i, rec in enumerate(samples_meta):
        bits = [f"{k}={v['tok']}@{v['pos']}" for k, v in rec["roles"].items()]
        lines.append(f"- s{i} winner={rec.get('winner')} " + " ".join(bits))
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"), flush=True)
    print(f"[probe] wrote {REPORT} and {JSON_OUT}", flush=True)


if __name__ == "__main__":
    main()
