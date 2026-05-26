"""
Build the consolidated horizontal LaTeX table for the paper.

Reads results_raw.csv (produced by main.py) and emits a single sidewaystable
that replaces the former tab:verification_majorant and tab:verification_local.

Layout
------
Rows   : indicators (η terms), majorant, true errors (p and u), I_eff (p and u)
Columns: one pair (M, NM) per mesh level h ∈ {0.3, 0.15, 0.075, 0.0375}
         M  = single matching value
         NM = mean ± std over 8 translation directions

Requires LaTeX packages: booktabs, rotating (sidewaystable), multirow

Usage
-----
    python -m mdnme.papers.2025_non_matching.example_1.build_latex_table
    python -m mdnme.papers.2025_non_matching.example_1.build_latex_table --csv path/to/results_raw.csv
"""

from __future__ import annotations

import argparse
import math
import pathlib
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
H_LEVELS = [0.3, 0.15, 0.075, 0.0375]
OUTFILE = pathlib.Path(__file__).parent / "results_consolidated.tex"

# Keys that are identically zero for the matching case (GM indicators)
GM_KEYS = {"eta_gm_gamma1", "eta_gm_gamma2", "eta_gm_omega1", "eta_gm_omega2"}

# Row definitions: (csv_key, latex_label, separator_before)
#   separator_before=True  inserts \midrule above this row
ROWS: List[Tuple[str, str, bool]] = [
    ("eta_nm_omega2", r"$\eta_{\mathrm{NM},\Omega_2}$",                             False),
    ("eta_nm_omega1", r"$\eta_{\mathrm{NM},\Omega_1}$",                             False),
    ("eta_nm_gamma1", r"$\eta_{\mathrm{NM},\Gamma_1}$",                             False),
    ("eta_nm_gamma2", r"$\eta_{\mathrm{NM},\Gamma_2}$",                             False),
    ("eta_gm_gamma1", r"$\eta_{\mathrm{GM},\Gamma_1}$",                             False),
    ("eta_gm_gamma2", r"$\eta_{\mathrm{GM},\Gamma_2}$",                             False),
    ("eta_gm_omega1", r"$\eta_{\mathrm{GM},\Omega_1}$",                             False),
    ("majorant",      r"$\mathcal{M}^{\oplus}_p=\mathcal{M}^{\oplus}_{\bm{u}}$",   True),
    ("true_p",        r"$\tnorm{p-p_h}$",                                           False),
    ("true_u",        r"$\tnormstar{\bm{u}-\bm{\sigma}_h}$",                        False),
    ("eff_p",         r"$I^{\mathrm{eff}}_p$",                                      False),
    ("eff_u",         r"$I^{\mathrm{eff}}_{\bm{u}}$",                               False),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class _Stats(NamedTuple):
    mean: float
    std: float


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _fmt_single(x: float) -> str:
    """Scalar value in compact scientific notation, e.g. 2.83e-02."""
    return f"{x:.2e}"


def _fmt_mean_std(mean: float, std: float) -> str:
    """Mean ± std formatted for LaTeX: '2.83e-02\\,$\\pm$\\,8.12e-03'."""
    return rf"{mean:.2e}\,$\pm$\,{std:.2e}"


def _dash() -> str:
    return r"\text{---}"


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


def _extract(
    records: List[Dict[str, float]], h: float
) -> Tuple[Dict[str, float], Dict[str, _Stats]]:
    """Return (matching_dict, nm_stats_dict) for one h level."""
    h_rows = [r for r in records if abs(r["h"] - h) < 1e-10]
    match = next(r for r in h_rows if r["case"] == 0)
    nm_rows = [r for r in h_rows if r["case"] == 1]

    all_keys = [k for k, _, _ in ROWS]
    nm_stats: Dict[str, _Stats] = {}
    for k in all_keys:
        vals = np.array([r[k] for r in nm_rows if k in r], dtype=float)
        if len(vals) > 1:
            nm_stats[k] = _Stats(mean=float(vals.mean()), std=float(vals.std(ddof=1)))
        elif len(vals) == 1:
            nm_stats[k] = _Stats(mean=float(vals[0]), std=float("nan"))
        else:
            nm_stats[k] = _Stats(mean=float("nan"), std=float("nan"))

    return match, nm_stats


# ---------------------------------------------------------------------------
# LaTeX builder
# ---------------------------------------------------------------------------
def _build_table(
    data: List[Tuple[Dict[str, float], Dict[str, _Stats]]],
) -> str:
    n_h = len(H_LEVELS)
    # 1 label column + 2 columns per h level (M, NM)
    col_spec = "l" + "".join(["rr"] * n_h)

    # Cmidrule positions: columns 2-3, 4-5, 6-7, 8-9
    cmidrules = "".join(
        rf"\cmidrule(lr){{{2 + 2*i}-{3 + 2*i}}}" for i in range(n_h)
    )

    lines: List[str] = []

    # --- table environment (sidewaystable requires rotating package) ---
    lines.append(r"\begin{sidewaystable}")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Verification of the 3D/2D manufactured solution. "
        r"Subdomain and interface error indicators, global majorants "
        r"$\mathcal{M}^\oplus_p = \mathcal{M}^\oplus_{\bm{u}}$, "
        r"true errors in pressure and flux, and effectivity indices. "
        r"Column~M: single matching-grid run; "
        r"column~NM: mean\,$\pm$\,std over 8 mortar-translation directions.}"
    )
    lines.append(r"\label{tab:verification_consolidated}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # --- h-level header ---
    h_headers = [
        rf"\multicolumn{{2}}{{c}}{{$h = {h:g}$}}" for h in H_LEVELS
    ]
    lines.append("& " + " & ".join(h_headers) + r" \\")
    lines.append(cmidrules)

    # --- M / NM sub-header ---
    sub_hdrs = []
    for _ in H_LEVELS:
        sub_hdrs.extend([r"\multicolumn{1}{c}{M}", r"\multicolumn{1}{c}{NM}"])
    lines.append("& " + " & ".join(sub_hdrs) + r" \\")
    lines.append(r"\midrule")

    # --- data rows ---
    for key, label, sep_before in ROWS:
        if sep_before:
            lines.append(r"\midrule")

        cells: List[str] = []
        for match, nm_stats in data:
            # Matching column
            if key in GM_KEYS:
                m_str = r"$" + _dash() + r"$"
            else:
                m_val = match.get(key, float("nan"))
                m_str = _fmt_single(m_val) if math.isfinite(m_val) else r"$" + _dash() + r"$"

            # NM column (mean ± std)
            stats = nm_stats.get(key)
            if stats is None or not math.isfinite(stats.mean):
                nm_str = r"$" + _dash() + r"$"
            elif not math.isfinite(stats.std):
                nm_str = _fmt_single(stats.mean)
            else:
                nm_str = _fmt_mean_std(stats.mean, stats.std)

            cells.extend([m_str, nm_str])

        lines.append(label + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{sidewaystable}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(csv_path: Optional[pathlib.Path] = None) -> None:
    if csv_path is None:
        script_dir = pathlib.Path(__file__).parent
        candidates = [
            script_dir / "results_raw.csv",
            pathlib.Path("results_raw.csv"),
        ]
        for c in candidates:
            if c.exists():
                csv_path = c
                break
        if csv_path is None:
            raise FileNotFoundError(
                "results_raw.csv not found. Run main.py first or pass --csv."
            )

    print(f"Reading: {csv_path}")
    records = _load_csv(csv_path)
    data = [_extract(records, h) for h in H_LEVELS]

    tex = _build_table(data)
    OUTFILE.write_text(tex)
    print(f"Table written to: {OUTFILE.resolve()}")
    print()
    print(tex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build consolidated LaTeX table")
    parser.add_argument("--csv", type=pathlib.Path, default=None, metavar="FILE")
    args = parser.parse_args()
    main(csv_path=args.csv)
