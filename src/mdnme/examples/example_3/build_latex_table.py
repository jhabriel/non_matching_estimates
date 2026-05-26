"""
Build the LaTeX table for the 3D fracture network with small features
(Flow Benchmark 3D Case 3).

Reads results_small_features.csv (produced by main.py) and emits the table
reproducing tab:thinfeatures3dresults from the paper.

Layout
------
Rows   : Matching / Non-matching
Columns: Approximation | η_Ω³ | η_Ω² | η_Ω¹ | η_Γ² | η_Γ¹ | M⁺

Requires LaTeX packages: booktabs

Usage
-----
    python -m mdnme.papers.2025_non_matching.example_3.build_latex_table
    python -m mdnme.papers.2025_non_matching.example_3.build_latex_table --csv path/to/results_small_features.csv
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTFILE = pathlib.Path(__file__).parent / "small_features_table.tex"

ROW_LABELS = {0: "Matching", 1: "Non-matching"}

# (csv_key, latex_column_header)
COLUMNS = [
    ("sd_error_3d",   r"$\eta_{\Omega^3}$"),
    ("sd_error_2d",   r"$\eta_{\Omega^2}$"),
    ("sd_error_1d",   r"$\eta_{\Omega^1}$"),
    ("intf_error_2d", r"$\eta_{\Gamma^2}$"),
    ("intf_error_1d", r"$\eta_{\Gamma^1}$"),
    ("majorant",      r"$\mathcal{M}^{\oplus}$"),
]


# ---------------------------------------------------------------------------
# CSV parsing (no pandas dependency)
# ---------------------------------------------------------------------------
def _load_csv(path: pathlib.Path) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    with open(path) as fh:
        header = next(fh).strip().split(",")
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            rec: Dict[str, float] = {}
            for k, v in zip(header, parts):
                try:
                    rec[k] = float(v)
                except ValueError:
                    rec[k] = float("nan")
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# LaTeX builder
# ---------------------------------------------------------------------------
def _fmt(x: float) -> str:
    return f"{x:.2e}"


def _build_table(records: List[Dict[str, float]]) -> str:
    # Sort: matching (non_matching=0) first
    rows = sorted(records, key=lambda r: int(r.get("non_matching", 0)))

    col_headers = " & ".join(h for _, h in COLUMNS)
    col_spec = "c " * (1 + len(COLUMNS))

    lines: List[str] = []
    lines.append(r"\begin{table}[tbp]")
    lines.append(
        r"    \caption{Majorant and dimension-based aggregated error estimators "
        r"for the 3D fracture network with small features from "
        r"Section~\ref{sec:small_features}.}"
    )
    lines.append(r"    \centering")
    lines.append(rf"    \begin{{tabular}}{{{col_spec.strip()}}}")
    lines.append(r"        \toprule")
    lines.append(f"        Approximation & {col_headers}\\\\ ")
    lines.append(r"        \midrule")

    for i, row in enumerate(rows):
        nm = int(row.get("non_matching", 0))
        label = ROW_LABELS.get(nm, f"case_{nm}")
        cells = " & ".join(_fmt(row.get(key, float("nan"))) for key, _ in COLUMNS)
        lines.append(f"        {label} & {cells} \\\\")
        if i < len(rows) - 1:
            lines.append(r"        \midrule")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    \label{tab:thinfeatures3dresults}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(csv_path: Optional[pathlib.Path] = None) -> None:
    if csv_path is None:
        script_dir = pathlib.Path(__file__).parent
        candidates = [
            script_dir / "results_small_features.csv",
            pathlib.Path("results_small_features.csv"),
        ]
        for c in candidates:
            if c.exists():
                csv_path = c
                break
        if csv_path is None:
            raise FileNotFoundError(
                "results_small_features.csv not found. Run main.py first or pass --csv."
            )

    print(f"Reading: {csv_path}")
    records = _load_csv(csv_path)
    tex = _build_table(records)
    OUTFILE.write_text(tex)
    print(f"Table written to: {OUTFILE.resolve()}")
    print()
    print(tex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build small features LaTeX table")
    parser.add_argument("--csv", type=pathlib.Path, default=None, metavar="FILE")
    args = parser.parse_args()
    main(csv_path=args.csv)
