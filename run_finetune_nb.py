#!/usr/bin/env python3
"""Execute all code cells of fine-tune.ipynb in order (for tmux jobs)."""
from pathlib import Path

import nbformat

NB = Path(__file__).resolve().parent / "fine-tune.ipynb"
nb = nbformat.read(NB, as_version=4)
chunks = []
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
    if src.lstrip().startswith("# @skip_train"):
        print(f"[run_finetune_nb] skip cell {i} (@skip_train)", flush=True)
        continue
    chunks.append(f"# --- notebook cell {i} ---\n{src.rstrip()}\n")
code = "\n".join(chunks)
print(f"[run_finetune_nb] executing {len(chunks)} code cells from {NB}", flush=True)
exec(compile(code, str(NB), "exec"), {"__name__": "__main__"})
