from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import porepy as pp

from mdnme.estimates.error_estimation import estimate_errors
from mdnme.models.varela_jnum_2d.model import manu_incomp_fluid, manu_incomp_solid

# mdnme imports
from mdnme.models.varela_jnum_3d.model import VarelaJNumSetup3D
from mdnme.models.varela_jnum_3d.true_errors import VarelaJNumTrueErrors3D

# -----------------------------
# Experiment configuration
# -----------------------------
CELL_SIZES: Sequence[float] = (0.3000, 0.1500, 0.0750, 0.0375)
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
# Numerical formatting for table values
FMT = "{:.2e}"
OUTDIR = pathlib.Path(".")
TABLE_TEX_MAJORANT = OUTDIR / "results_majorant.tex"
TABLE_TEX_LOCAL = OUTDIR / "results_local.tex"
CSV_RAW = OUTDIR / "results_raw.csv"


@dataclass
class Metrics:
    h: float
    # local components (for Subdomain/Interface table)
    eta_matrix: float
    eta_frac: float
    eta_left_intf: float
    eta_right_intf: float
    # global majorant and true errors
    majorant: float
    true_p: float  # || p - s_h ||
    true_u: float  # || u - sigma_h ||
    eff_p: float  # M^oplus / true_p
    eff_u: float  # M^oplus / true_u

    def as_list(self) -> List[float]:
        return [
            self.h,
            self.eta_matrix,
            self.eta_frac,
            self.eta_left_intf,
            self.eta_right_intf,
            self.majorant,
            self.true_p,
            self.true_u,
            self.eff_p,
            self.eff_u,
        ]


# -----------------------------
# Helpers
# -----------------------------


def _material_constants() -> Dict[str, pp.PhysicalConstants]:  # type:ignore
    solid_constants = pp.SolidConstants(**manu_incomp_solid)  # type:ignore[arg-type]
    fluid_constants = pp.FluidComponent(**manu_incomp_fluid)  # type:ignore[arg-type]
    return {"solid": solid_constants, "fluid": fluid_constants}


def _split_interface_lr(
    diff_intf: np.ndarray, n_cells: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Split interface diffusive error array into left/right halves."""
    half = n_cells // 2
    right = diff_intf[:half]
    left = diff_intf[half:]
    return left, right


def _run_single(
    h: float, *, non_matching: bool, translation: Tuple[int, int, int] | None
) -> Metrics:
    """Run one configuration and compute scalar metrics.

    Parameters
    ----------
    h : float
        Target cell size.
    non_matching : bool
        If True, set up non-matching case with fracture perturbation.
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
        assert (
            translation is not None
        ), "translation must be provided in non-matching runs"
        params: Mapping = dict(
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

    setup = VarelaJNumSetup3D(params)  # type:ignore

    # Run the time-dependent model and estimate errors
    pp.run_time_dependent_model(setup, {})
    estimate_errors(setup.mdg, is_non_matching=non_matching)

    mdg = setup.mdg

    # Extract data: subdomains and interfaces
    ((sd_mat, d_mat),) = mdg.subdomains(dim=3, return_data=True)
    ((sd_frac, d_frac),) = mdg.subdomains(dim=2, return_data=True)
    ((intf, d_intf),) = mdg.interfaces(dim=2, return_data=True)

    # Diffusive errors (arrays per cell)
    diff_sd_mat = np.asarray(d_mat["estimates"]["diffusive_error"])  # (n_matrix_cells,)
    diff_sd_frac = np.asarray(d_frac["estimates"]["diffusive_error"])  # (n_frac_cells,)
    diff_intf = np.asarray(d_intf["estimates"]["diffusive_error"])  # (n_mortar_cells,)

    # Interface split into left / right halves
    diff_intf_left, diff_intf_right = _split_interface_lr(diff_intf, intf.num_cells)

    # Residual errors (scalars after summation)
    # Note: these are the residual parts used in the majorant
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

    # True errors & effectivities
    te = VarelaJNumTrueErrors3D(setup)
    true_p = float(te.true_error_primal())  # assumes you've named it like this
    true_u = float(te.true_error_dual())
    eff_p = majorant / true_p if true_p > 0 else np.nan
    eff_u = majorant / true_u if true_u > 0 else np.nan

    return Metrics(
        h=h,
        eta_matrix=eta_matrix,
        eta_frac=eta_frac,
        eta_left_intf=eta_left_intf,
        eta_right_intf=eta_right_intf,
        majorant=majorant,
        true_p=true_p,
        true_u=true_u,
        eff_p=eff_p,
        eff_u=eff_u,
    )


@dataclass
class MeanStd:
    mean: float
    std: float

    def latex(self) -> str:
        return f"{FMT.format(self.mean)}\\,$\\pm$\\,{FMT.format(self.std)}"


def _aggregate(ms: Sequence[Metrics]) -> Dict[str, MeanStd]:
    """Compute mean/std across Metrics fields except h (assumed constant)."""
    assert len(ms) > 0
    keys = [
        "eta_matrix",
        "eta_frac",
        "eta_left_intf",
        "eta_right_intf",
        "majorant",
        "true_p",
        "true_u",
        "eff_p",
        "eff_u",
    ]
    agg: Dict[str, MeanStd] = {}
    for k in keys:
        vals = np.array([getattr(m, k) for m in ms], dtype=float)
        agg[k] = MeanStd(mean=float(vals.mean()), std=float(vals.std(ddof=1)))
    return agg


def _fmt_val(x: float) -> str:
    return FMT.format(x)


# -----------------------------
# LaTeX builders
# -----------------------------
def build_latex_table_majorant(rows: List[Tuple[str, List[str], List[str]]]) -> str:
    """Create LaTeX table (booktabs + multirow) for majorant/true/errors/eff indices.

    rows: list of (h_str, cols_match, cols_nonmatch)
          where each cols_* = [ M^oplus, ||u-σ_h||, I_u^eff, ||p-s_h||, I_p^eff ]
          matching row uses plain numbers; non-matching uses 'mean ± std' strings.
    """
    header = (
        "\\begin{table}\n"
        "\\centering\n"
        "\\caption{3D/2D verification results: majorants, true errors and effectivity"
        " indices.}\n"
        "\\label{tab:verification_majorant}\n"
        "\\begin{tabular}{lrrrrr}\n"
        "\\toprule\n"
        "$h$ & $\\mathcal{M}^\\oplus_p=\\mathcal{M}^\\oplus_{\\bm{u}}$ & "
        "$\\tnormstar{\\bm{u} - \\bm{\\sigma}_h}$ & $I_{\\bm{u}}^{\\mathrm{eff}}$ & "
        "$\\tnorm{p - s_h}$ & $I_p^{\\mathrm{eff}}$ \\\\\n"
        "\\midrule\n"
    )
    body_lines = []
    n = len(rows)
    for i, (h_str, cols_match, cols_nm) in enumerate(rows):
        body_lines.append(
            f"\\multirow{{2}}{{*}}{{{h_str}}} & " + " & ".join(cols_match) + " \\\\"
        )
        # non-matching line under it
        body_lines.append(" & " + " & ".join(cols_nm) + " \\\\")
        # midrule between blocks, but not after last
        if i != n - 1:
            body_lines.append("\\midrule")
    body = "\n".join(body_lines) + "\n"
    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return header + body + footer


def build_latex_table_local(rows: List[Tuple[str, List[str], List[str]]]) -> str:
    """Create LaTeX table for local subdomain/interface errors with your mixed style."""
    header = (
        "\\begin{table}\n"
        "\\centering\n"
        "\\caption{3D/2D verification results: Subdomain and interface errors.}\n"
        "\\label{tab:verification_local}\n"
        "\\begin{tabular}{lrrrr}\n"
        "\\hline\n"
        "$h$ & $\\eta_{\\Omega_2}$ & $\\eta_{\\Omega_1}$ & $\\eta_{\\Gamma_1}$ & "
        "$\\eta_{\\Gamma_2}$ \\\\\n"
        "\\toprule\n"
    )
    body_lines = []
    n = len(rows)
    for i, (h_str, cols_match, cols_nm) in enumerate(rows):
        body_lines.append(
            f"\\multirow{{2}}{{*}}{{{h_str}}} & " + " & ".join(cols_match) + " \\\\"
        )
        body_lines.append(" & " + " & ".join(cols_nm) + " \\\\")
        if i != n - 1:
            body_lines.append("\\midrule")
    body = "\n".join(body_lines) + "\n"
    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return header + body + footer


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    # touch material constants once (sanity)
    _ = _material_constants()

    # For LaTeX building
    rows_majorant: List[Tuple[str, List[str], List[str]]] = []
    rows_local: List[Tuple[str, List[str], List[str]]] = []

    # For optional CSV export of raw numbers
    csv_records: List[Dict[str, float]] = []

    for h in CELL_SIZES:
        print(f"\n=== h = {h:.4f} | Matching ===")
        m_match = _run_single(h, non_matching=False, translation=None)

        # Matching rows (plain values)
        row_match_majorant = [
            _fmt_val(m_match.majorant),
            _fmt_val(m_match.true_u),
            _fmt_val(m_match.eff_u),
            _fmt_val(m_match.true_p),
            _fmt_val(m_match.eff_p),
        ]
        row_match_local = [
            _fmt_val(m_match.eta_matrix),
            _fmt_val(m_match.eta_frac),
            _fmt_val(m_match.eta_left_intf),
            _fmt_val(m_match.eta_right_intf),
        ]

        # CSV record for matching
        csv_records.append(
            {
                "h": h,
                "case": 0,  # 0=matching, 1=nonmatching
                "eta_matrix": m_match.eta_matrix,
                "eta_frac": m_match.eta_frac,
                "eta_left_intf": m_match.eta_left_intf,
                "eta_right_intf": m_match.eta_right_intf,
                "majorant": m_match.majorant,
                "true_p": m_match.true_p,
                "true_u": m_match.true_u,
                "eff_p": m_match.eff_p,
                "eff_u": m_match.eff_u,
            }
        )

        # Non-matching batch
        print(f"=== h = {h:.4f} | Non-matching (8 translations) ===")
        ms: List[Metrics] = []
        for tr in TRANSLATIONS:
            print(f"  -> translation = {tr}")
            m_nm = _run_single(h, non_matching=True, translation=tr)
            ms.append(m_nm)
            # CSV record for each realization
            csv_records.append(
                {
                    "h": h,
                    "case": 1,  # 0=matching, 1=nonmatching
                    "tr_x": tr[0],
                    "tr_y": tr[1],
                    "tr_z": tr[2],
                    "eta_matrix": m_nm.eta_matrix,
                    "eta_frac": m_nm.eta_frac,
                    "eta_left_intf": m_nm.eta_left_intf,
                    "eta_right_intf": m_nm.eta_right_intf,
                    "majorant": m_nm.majorant,
                    "true_p": m_nm.true_p,
                    "true_u": m_nm.true_u,
                    "eff_p": m_nm.eff_p,
                    "eff_u": m_nm.eff_u,
                }
            )

        agg = _aggregate(ms)

        # Non-matching (mean ± std) rows
        row_nonmatch_majorant = [
            agg["majorant"].latex(),
            agg["true_u"].latex(),
            agg["eff_u"].latex(),
            agg["true_p"].latex(),
            agg["eff_p"].latex(),
        ]
        row_nonmatch_local = [
            agg["eta_matrix"].latex(),
            agg["eta_frac"].latex(),
            agg["eta_left_intf"].latex(),
            agg["eta_right_intf"].latex(),
        ]

        # Accumulate rows (order: matching row, then non-matching row for same h)
        h_str = FMT.format(h)
        rows_majorant.append((h_str, row_match_majorant, row_nonmatch_majorant))
        rows_local.append((h_str, row_match_local, row_nonmatch_local))

    # Build and write LaTeX tables
    export_to_latex = False
    if export_to_latex:
        tex_majorant = build_latex_table_majorant(rows_majorant)
        TABLE_TEX_MAJORANT.write_text(tex_majorant)
        print(
            f"\nLaTeX table (majorant/true/eff) written to:"
            f" {TABLE_TEX_MAJORANT.resolve()}"
        )

        tex_local = build_latex_table_local(rows_local)
        TABLE_TEX_LOCAL.write_text(tex_local)
        print(
            f"LaTeX table (local components) written to: {TABLE_TEX_LOCAL.resolve()}\n"
        )

    # Optional CSV export of raw numbers
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
