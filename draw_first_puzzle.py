#!/usr/bin/env python3
"""Draw the first scat_targeted puzzle with reading-order cell indices."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

from export_mtm_csv import load_prompt_dataset, split_canvas

from matplotlib import font_manager

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

COLOR_NAMES = ["black", "blue", "red", "green", "yellow", "grey", "pink", "orange", "teal", "maroon", "white"]
COLOR_RGB = {
    "black": (0, 0, 0),
    "blue": (0.00, 0.46, 0.85),
    "red": (1.00, 0.25, 0.21),
    "green": (0.18, 0.80, 0.25),
    "yellow": (1.00, 0.86, 0.00),
    "grey": (0.67, 0.67, 0.67),
    "pink": (0.94, 0.07, 0.75),
    "orange": (1.00, 0.52, 0.11),
    "teal": (0.50, 0.86, 1.00),
    "maroon": (0.53, 0.05, 0.15),
    "white": (1, 1, 1),
}
CMAP = ListedColormap([COLOR_RGB[n] for n in COLOR_NAMES])
NORM = BoundaryNorm(np.arange(-0.5, 11.5, 1.0), CMAP.N)
ROOT = Path(__file__).resolve().parent


def text_color(val):
    r, g, b = COLOR_RGB[COLOR_NAMES[int(val)]]
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.55 else "black"


def draw(ax, grid, title, mark=None):
    mat = np.array(grid, dtype=int)
    ax.imshow(mat, cmap=CMAP, norm=NORM, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#222")
        spine.set_linewidth(1.2)
    n = 1
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            ax.text(
                c, r, str(n), ha="center", va="center",
                fontsize=11, color=text_color(mat[r, c]), fontweight="bold",
            )
            if mark == n:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="#ffffff", linewidth=2.4))
                ax.add_patch(Rectangle((c - 0.42, r - 0.42), 0.84, 0.84, fill=False, edgecolor="#111111", linewidth=1.1))
            n += 1
    ax.set_title(title, fontsize=11, pad=4)


def main():
    row = load_prompt_dataset(ROOT / "data_uniform" / "scat_targeted.csv")[0]
    task = row["task"]
    blocks = []
    for i, ex in enumerate(task["train"], 1):
        cands, query = split_canvas(ex["input"])
        blocks.append((f"示例 {i}", cands, query, ex["output"]))
    cands, query = split_canvas(task["test"][0]["input"])
    blocks.append(("Test（尚未写出）", cands, query, task["test"][0]["output"]))

    fig, axes = plt.subplots(4, 5, figsize=(12.2, 9.2))
    col_names = ["候选 A", "候选 B", "候选 C", "Query", "Output"]
    for i, (lab, cands, query, out) in enumerate(blocks):
        grids = list(cands) + [query, out]
        for j, grid in enumerate(grids):
            mark = 2 if j >= 3 and i < 3 else None
            draw(axes[i, j], grid, col_names[j] if i == 0 else "", mark=mark)
        axes[i, 0].set_ylabel(lab, fontsize=12, rotation=0, labelpad=36, va="center")
    fig.suptitle(
        "数据集第一条  move_to_match_0000\n格子序号按从左到右、从上到下。白框是各示例 Output / Query 的第 2 格",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.04, 0.02, 1, 0.93))
    out = ROOT / "figures" / "move_to_match_0000.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140)
    print(out)


if __name__ == "__main__":
    main()
