"""
Batch experiment runner for matching vs non‑matching grids
for VarelaJNum 3D example (mdnme + PorePy).

What it does
------------
• Sweeps 4 cell sizes: [0.3000, 0.1500, 0.0750, 0.0375].
• For each cell size, runs one matching case and eight non‑matching
  cases using prescribed translation vectors.
• Aggregates non‑matching results by mean and standard deviation.
• Exports a LaTeX table (results_table.tex) with 8 rows: for each h,
  the first row is Matching, the second row is Non‑matching (mean ± std).

Columns in the table
--------------------
[h, eta_matrix, eta_frac, eta_left_intf, eta_right_intf, majorant, eff_idx]

Notes
-----
• LaTeX: requires \\usepackage{rotating} for sidewaystable.
• Make sure mdnme and porepy are installed & importable in your environment.
• This script intentionally suppresses time exports to keep runs quieter.
• If something crashes in one configuration, it raises and stops, to avoid
  mixing partial results.
"""
from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import porepy as pp

# mdnme imports
from mdnme.examples.varela_jnum_3d.model import VarelaJNumSetup3D
from mdnme.examples.varela_jnum_2d.model import manu_incomp_fluid, manu_incomp_solid
from mdnme.examples.varela_jnum_3d.true_errors import VarelaJNumTrueErrors3D
from mdnme.estimates.error_estimation import estimate_errors


# -----------------------------
# Experiment configuration
# -----------------------------
CELL_SIZES: Sequence[float] = (0.3000, 0.1500, 0.0750, 0.0375)
# CELL_SIZES: Sequence[float] = (0.3000, 0.1500)
TRANSLATIONS: Sequence[Tuple[int, int, int]] = (
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
    (0, 1, 1),
    (0, -1, -1),
    (0, 1, -1),
    (0, -1, 1),
)
# Numerical formatting for LaTeX table values
FMT = "{:.2e}"  # change to "{:.4e}" for scientific notation
OUTDIR = pathlib.Path(".")
TABLE_TEX = OUTDIR / "results_table.tex"
CSV_RAW = OUTDIR / "results_raw.csv"


@dataclass
class Metrics:
    h: float
    eta_matrix: float
    eta_frac: float
    eta_left_intf: float
    eta_right_intf: float
    majorant: float
    eff_idx: float

    def as_list(self) -> List[float]:
        return [
            self.h,
            self.eta_matrix,
            self.eta_frac,
            self.eta_left_intf,
            self.eta_right_intf,
            self.majorant,
            self.eff_idx,
        ]


# -----------------------------
# Helpers
# -----------------------------

def _material_constants() -> Dict[str, pp.PhysicalConstants]:
    solid_constants = pp.SolidConstants(**manu_incomp_solid)
    fluid_constants = pp.FluidComponent(**manu_incomp_fluid)
    return {"solid": solid_constants, "fluid": fluid_constants}


def _split_interface_lr(diff_intf: np.ndarray, n_cells: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split interface diffusive error array into left/right halves, like the user's ref code."""
    half = n_cells // 2
    right = diff_intf[:half]
    left = diff_intf[half:]
    return left, right


def _run_single(h: float, *, non_matching: bool, translation: Tuple[int, int, int] | None) -> Metrics:
    """Run one configuration and compute scalar metrics.

    Parameters
    ----------
    h : float
        Target cell size.
    non_matching : bool
        If True, set up non‑matching case with fracture perturbation.
    translation : tuple or None
        (dx, dy, dz) integer translation vector for internal node perturbation.
    """
    material_constants = _material_constants()

    common_params = {
        "grid_type": "simplex",
        "material_constants": material_constants,
        "meshing_arguments": {"cell_size": h},
        "times_to_export": [],  # suppress outputs
    }

    if non_matching:
        assert translation is not None, "translation must be provided in non‑matching runs"
        params = dict(
            common_params,
            non_matching=True,
            perturb_fracture=True,
            perturb_mortar=True,
            refine_fracture=False,
            refine_mortar=False,
            translation_vector=translation,
        )
    else:
        params = common_params

    setup = VarelaJNumSetup3D(params)

    # Run the time‑dependent model and estimate errors
    pp.run_time_dependent_model(setup, {})
    estimate_errors(setup.mdg)

    mdg = setup.mdg

    # Extract data: subdomains and interfaces
    (sd_mat, d_mat), = mdg.subdomains(dim=3, return_data=True)
    (sd_frac, d_frac), = mdg.subdomains(dim=2, return_data=True)
    (intf, d_intf), = mdg.interfaces(dim=2, return_data=True)

    # Diffusive errors (arrays per cell)
    diff_sd_mat = np.asarray(d_mat["estimates"]["diffusive_error"])  # shape: (n_cells,)
    diff_sd_frac = np.asarray(d_frac["estimates"]["diffusive_error"])  # shape: (n_cells,)
    diff_intf = np.asarray(d_intf["estimates"]["diffusive_error"])     # shape: (n_cells,)

    # Interface split into left / right halves
    diff_intf_left, diff_intf_right = _split_interface_lr(diff_intf, intf.num_cells)

    # Residual errors (scalars after summation)
    resi_sd_mat = float(setup.exact_sol.residual_error_matrix(sd_mat, d_mat).sum())
    resi_sd_frac = float(setup.exact_sol.residual_error_fracture(sd_frac, d_frac).sum())

    # Component eta's (scalar contributions)
    eta_matrix = math.sqrt(float(diff_sd_mat.sum()) + resi_sd_mat)
    eta_frac = math.sqrt(float(diff_sd_frac.sum()) + resi_sd_frac)
    eta_left_intf = math.sqrt(float(diff_intf_left.sum()))
    eta_right_intf = math.sqrt(float(diff_intf_right.sum()))

    # Global majorant pieces (scalars)
    diff_error = math.sqrt(
        float(diff_sd_mat.sum())
        + float(diff_sd_frac.sum())
        + float(diff_intf_left.sum())
        + float(diff_intf_right.sum())
    )
    residual_error = math.sqrt(resi_sd_mat + resi_sd_frac)
    majorant = diff_error + residual_error

    # True error and efficiency index
    te = VarelaJNumTrueErrors3D(setup)
    true_error = float(te.true_error())
    eff_idx = majorant / true_error

    return Metrics(
        h=h,
        eta_matrix=eta_matrix,
        eta_frac=eta_frac,
        eta_left_intf=eta_left_intf,
        eta_right_intf=eta_right_intf,
        majorant=majorant,
        eff_idx=eff_idx,
    )


@dataclass
class MeanStd:
    mean: float
    std: float

    def latex(self) -> str:
        return f"{FMT.format(self.mean)}\\,$\\pm$\\,{FMT.format(self.std)}"


def _aggregate(ms: Sequence[Metrics]) -> Dict[str, MeanStd]:
    """Compute mean/std for each Metrics field except h (assumed constant)."""
    assert len(ms) > 0
    keys = [
        "eta_matrix",
        "eta_frac",
        "eta_left_intf",
        "eta_right_intf",
        "majorant",
        "eff_idx",
    ]
    agg: Dict[str, MeanStd] = {}
    for k in keys:
        vals = np.array([getattr(m, k) for m in ms], dtype=float)
        agg[k] = MeanStd(mean=float(vals.mean()), std=float(vals.std(ddof=1)))
    return agg


def _fmt_val(x: float) -> str:
    return FMT.format(x)


def build_latex_table(rows: List[Tuple[str, List[str]]]) -> str:
    """Create a LaTeX table using plain tabular + hline.

    Parameters
    ----------
    rows : list of (h_str, columns) where columns are strings already formatted
           (including $\pm$ where desired).
    """
    header = (
        "\\begin{sidewaystable}\n"
        "\\centering\n"
        "\\caption{Matching vs non-matching (mean +//- std over 8 translations). "
        "First row per h is Matching; second is Non-matching.}\n"
        "\\label{tab:3d_verification}\n"
        "\\begin{tabular}{lrrrrrr}\n"
        "\\hline\n"
        "$h$ & $\\eta_{\\mathrm{mat}}$ & $\\eta_{\\mathrm{frac}}$ & $\\eta_{\\mathrm{"
        "intf,L}}$ & $\\eta_{\\mathrm{intf,R}}$ & Majorant & Eff. idx \\\\ \n"
        "\\hline\n"
    )
    body_lines = []
    for h_str, cols in rows:
        body_lines.append(" ".join([h_str, "&", " & ".join(cols), "\\\\"]))
    body = "\n".join(body_lines) + "\n"
    footer = "\\hline\n\\end{tabular}\n\end{sidewaystable}\n"
    return header + body + footer


def main() -> None:
    material_constants = _material_constants()  # sanity build once
    del material_constants

    all_rows: List[Tuple[str, List[str]]] = []

    # For optional CSV export of raw numbers
    csv_records: List[Dict[str, float]] = []

    for h in CELL_SIZES:
        print(f"\n=== h = {h:.4f} | Matching ===")
        m_match = _run_single(h, non_matching=False, translation=None)

        # Matching row (no ±):
        row_match = [
            _fmt_val(m_match.eta_matrix),
            _fmt_val(m_match.eta_frac),
            _fmt_val(m_match.eta_left_intf),
            _fmt_val(m_match.eta_right_intf),
            _fmt_val(m_match.majorant),
            _fmt_val(m_match.eff_idx),
        ]
        all_rows.append((FMT.format(h), row_match))

        # CSV record for matching
        csv_records.append({
            "h": h,
            "case": 0,  # 0=matching, 1=nonmatching
            "eta_matrix": m_match.eta_matrix,
            "eta_frac": m_match.eta_frac,
            "eta_left_intf": m_match.eta_left_intf,
            "eta_right_intf": m_match.eta_right_intf,
            "majorant": m_match.majorant,
            "eff_idx": m_match.eff_idx,
        })

        # Non‑matching batch
        print(f"=== h = {h:.4f} | Non‑matching (8 translations) ===")
        ms: List[Metrics] = []
        for tr in TRANSLATIONS:
            print(f"  -> translation = {tr}")
            m_nm = _run_single(h, non_matching=True, translation=tr)
            ms.append(m_nm)
            # CSV record for each realization
            csv_records.append({
                "h": h,
                "case": 1,  # 0=matching, 1=nonmatching
                "tr_x": tr[0], "tr_y": tr[1], "tr_z": tr[2],
                "eta_matrix": m_nm.eta_matrix,
                "eta_frac": m_nm.eta_frac,
                "eta_left_intf": m_nm.eta_left_intf,
                "eta_right_intf": m_nm.eta_right_intf,
                "majorant": m_nm.majorant,
                "eff_idx": m_nm.eff_idx,
            })

        agg = _aggregate(ms)
        row_nonmatch = [
            agg["eta_matrix"].latex(),
            agg["eta_frac"].latex(),
            agg["eta_left_intf"].latex(),
            agg["eta_right_intf"].latex(),
            agg["majorant"].latex(),
            agg["eff_idx"].latex(),
        ]
        # duplicate h in the second row as requested; ordering is matching then non‑matching
        all_rows.append((FMT.format(h), row_nonmatch))

    # Build and write LaTeX table
    tex = build_latex_table(all_rows)
    TABLE_TEX.write_text(tex)
    print(f"\nLaTeX table written to: {TABLE_TEX.resolve()}\n")

    # Optional CSV of raw numbers for downstream plotting / checks
    try:
        import pandas as pd
        df = pd.DataFrame.from_records(csv_records)
        df.to_csv(CSV_RAW, index=False)
        print(f"Raw results written to: {CSV_RAW.resolve()}")
    except Exception as e:
        print("(Skipping CSV export; pandas not available or failed)")
        print(str(e))


if __name__ == "__main__":
    main()
