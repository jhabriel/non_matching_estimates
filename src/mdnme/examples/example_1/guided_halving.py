"""
Guided subdomain halving for the 3d benchmark.

Experiment
----------
Starting from an initial matching mesh (e.g., h=0.30), at each step the subdomain
with the larger error indicator is uniformly bisected while the others are kept fixed:

  - η_matrix >= η_frac : GMSH bisects the bulk (matrix) via gmsh.model.mesh.refine();
                          the fracture grid is swapped back to its current fixed grid
                          to enforce non-matching from the second iteration.
  - η_frac  >  η_matrix: fracture replaced by StructuredTriangleGrid at h_frac/2;
                          the matrix is unchanged.

Mortar grids are rebuilt geometrically by replace_subdomains_and_interfaces().

Matrix bisection uses PorePy's GridSequenceFactory (mode='nested'), which wraps
gmsh.model.mesh.refine() — the same GMSH bisection used in PorePy's own
convergence studies.  Fracture halving uses StructuredTriangleGrid on the
fracture plane x=0.5, y,z ∈ [0.25, 0.75].

Output
------
  guided_halving_history.csv    per-step metrics (including GM terms)
  guided_halving_table.tex      LaTeX table (9 columns, GM terms not shown)

Usage
-----
    python -m mdnme.papers.2025_non_matching.example_1.guided_halving
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import porepy as pp
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from porepy.grids.refinement import GridSequenceFactory

from mdnme.estimates.error_estimation import estimate_errors
from mdnme.models.varela_jnum_2d.model import manu_incomp_fluid, manu_incomp_solid
from mdnme.models.varela_jnum_3d.geometry import VarelaJNumGeometry3D
from mdnme.models.varela_jnum_3d.model import VarelaJNumSetup3D
from mdnme.models.varela_jnum_3d.true_errors import VarelaJNumTrueErrors3D
from mdnme.utils.grid_rotation import build_canonical_frames

# ── Configuration ─────────────────────────────────────────────────────────────
CELL_SIZE = 0.30    # initial mesh size
N_STEPS = 3         # guided refinement steps after level 0
H_FRAC_INIT = 0.30  # initial fracture mesh size

FMT = "{:.2e}"
OUTDIR = pathlib.Path(".")


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    step: int
    h_mat_approx: float
    h_frac: float
    n_matrix: int
    n_frac: int
    eta_matrix: float    # sqrt(diff_mat + resi_mat)
    eta_frac: float      # sqrt(diff_frac + resi_frac)
    eta_intf: float      # sqrt(diff_intf)
    eta_gm_perp: float   # sqrt(gm_diffusive_error_perp)  — stored in CSV, not in table
    eta_gm_res: float    # sqrt(gm_residual_error_frac)   — stored in CSV, not in table
    majorant: float      # full M^+ (includes GM terms)
    true_p: float        # primal (pressure) true error
    true_u: float        # dual (velocity) true error
    eff_p: float         # M^+ / true_p
    eff_u: float         # M^+ / true_u
    refined: str         # "initial" | "matrix" | "fracture"


# ── Remeshable model ──────────────────────────────────────────────────────────

class _RemeshableSetup(VarelaJNumSetup3D):
    """VarelaJNumSetup3D with optional prebuilt MDG injection."""

    _injected_mdg: Optional[pp.MixedDimensionalGrid] = None
    _injected_fracture_network: Optional[FractureNetwork3d] = None

    def set_geometry(self) -> None:
        if self._injected_mdg is not None:
            self.set_domain()
            self.mdg = self._injected_mdg
            self.nd = self.mdg.dim_max()
            pp.set_local_coordinate_projections(self.mdg)
            build_canonical_frames(self.mdg)
            self.fracture_network = self._injected_fracture_network
        else:
            super().set_geometry()


# ── Fracture grid factory ─────────────────────────────────────────────────────

def _create_fracture_grid(h: float) -> pp.Grid:
    """Structured triangle mesh on the fracture plane x=0.5, y,z ∈ [0.25, 0.75]."""
    n = max(1, int(np.ceil(0.5 / h)))
    grid = pp.StructuredTriangleGrid([n, n], [0.5, 0.5])
    raw = grid.nodes.copy()
    grid.nodes[0, :] = 0.5
    grid.nodes[1, :] = raw[0, :] + 0.25
    grid.nodes[2, :] = raw[1, :] + 0.25
    grid.compute_geometry()
    return grid


# ── Fracture network helper ───────────────────────────────────────────────────

class _FracNetExtractor(VarelaJNumGeometry3D):
    """Lightweight helper: builds the Varela3D fracture network without meshing."""

    def __init__(self, params: dict) -> None:
        self.params = params

    def grid_type(self):   # type: ignore[override]
        return self.params.get("grid_type", "simplex")


def _build_fracture_network(params: dict) -> FractureNetwork3d:
    ext = _FracNetExtractor(params)
    ext.set_domain()
    ext.set_fractures()
    return pp.create_fracture_network(ext._fractures, ext._domain)  # type: ignore[arg-type]


# ── Solve + estimate ──────────────────────────────────────────────────────────

def _solve_and_estimate(
    mdg: pp.MixedDimensionalGrid,
    frac_net: FractureNetwork3d,
    params: dict,
    non_matching: bool,
) -> dict:
    """Solve on mdg and compute error indicators. Returns flat metrics dict."""
    p = dict(params, non_matching=non_matching)
    setup = _RemeshableSetup(p)  # type: ignore[arg-type]
    setup._injected_mdg = mdg
    setup._injected_fracture_network = frac_net
    pp.run_time_dependent_model(setup, {})

    estimate_errors(setup.mdg, is_non_matching=non_matching)

    ((sd_mat, d_mat),) = setup.mdg.subdomains(dim=3, return_data=True)
    ((sd_frac, d_frac),) = setup.mdg.subdomains(dim=2, return_data=True)
    ((_, d_intf),) = setup.mdg.interfaces(dim=2, return_data=True)

    resi_mat  = np.asarray(setup.exact_sol.residual_error_matrix(sd_mat, d_mat))
    resi_frac = np.asarray(setup.exact_sol.residual_error_fracture(sd_frac, d_frac))
    diff_mat  = np.asarray(d_mat["estimates"]["diffusive_error"])
    diff_frac = np.asarray(d_frac["estimates"]["diffusive_error"])
    diff_intf = np.asarray(d_intf["estimates"]["diffusive_error"])

    # GM terms (non-zero only for non-matching; .get fallback for matching case)
    gm_res_frac = np.asarray(d_frac["estimates"].get("gm_residual_error", 0.0))
    gm_perp     = np.asarray(d_intf["estimates"].get("gm_diffusive_error_perp", 0.0))

    eta_matrix  = math.sqrt(float(diff_mat.sum()) + float(resi_mat.sum()))
    eta_frac    = math.sqrt(float(diff_frac.sum()) + float(resi_frac.sum()))
    eta_intf    = math.sqrt(float(diff_intf.sum()))
    eta_gm_perp = math.sqrt(float(np.sum(gm_perp)))
    eta_gm_res  = math.sqrt(float(np.sum(gm_res_frac)))

    diff_sum    = float(diff_mat.sum() + diff_frac.sum() + diff_intf.sum())
    resi_sum    = float(resi_mat.sum() + resi_frac.sum())
    gm_diff_sum = float(np.sum(gm_perp))
    gm_res_sum  = float(np.sum(gm_res_frac))
    M = math.sqrt(diff_sum + gm_diff_sum) + math.sqrt(resi_sum + gm_res_sum)

    te = VarelaJNumTrueErrors3D(setup)
    true_p = float(te.true_error_primal())
    true_u = float(te.true_error_dual())

    return {
        "sd_mat": sd_mat,
        "sd_frac": sd_frac,
        "eta_matrix": eta_matrix,
        "eta_frac": eta_frac,
        "eta_intf": eta_intf,
        "eta_gm_perp": eta_gm_perp,
        "eta_gm_res": eta_gm_res,
        "majorant": M,
        "true_p": true_p,
        "true_u": true_u,
        "eff_p": M / true_p if true_p > 0 else float("nan"),
        "eff_u": M / true_u if true_u > 0 else float("nan"),
    }


# ── Main experiment ───────────────────────────────────────────────────────────

def run_guided_halving(params: dict) -> List[StepRecord]:
    records: List[StepRecord] = []

    frac_net = _build_fracture_network(params)

    factory_params = {
        "mode": "nested",
        "num_refinements": N_STEPS + 2,  # headroom: worst case all steps are matrix
        "mesh_param": {
            "mesh_size_fracture": CELL_SIZE,
            "mesh_size_boundary": CELL_SIZE,
            "mesh_size_min": 0.05 * CELL_SIZE,
            "refinement_size_multiplier": 1.0,
            "refinement_proximity_multiplier": 1.0,
        },
        "grid_param": {"constraints": np.arange(1, 25)},
    }
    factory = GridSequenceFactory(frac_net, factory_params)
    factory_iter = iter(factory)

    # ── Level 0: initial matching solve ───────────────────────────────────────
    print(f"\n  step=0  initial matching  h≈{CELL_SIZE}")
    mdg = pp.fracture_importer.dfm_from_gmsh(factory._out_file, dim=3)
    pp.set_local_coordinate_projections(mdg)

    est = _solve_and_estimate(mdg, frac_net, params, non_matching=False)

    current_frac_sd: pp.Grid = mdg.subdomains(dim=2)[0]
    h_frac = H_FRAC_INIT
    matrix_bisection_count = 0

    records.append(StepRecord(
        step=0,
        h_mat_approx=CELL_SIZE,
        h_frac=h_frac,
        n_matrix=est["sd_mat"].num_cells,
        n_frac=est["sd_frac"].num_cells,
        eta_matrix=est["eta_matrix"],
        eta_frac=est["eta_frac"],
        eta_intf=est["eta_intf"],
        eta_gm_perp=est["eta_gm_perp"],
        eta_gm_res=est["eta_gm_res"],
        majorant=est["majorant"],
        true_p=est["true_p"],
        true_u=est["true_u"],
        eff_p=est["eff_p"],
        eff_u=est["eff_u"],
        refined="initial",
    ))
    _print_step(records[-1])

    # ── Guided loop ───────────────────────────────────────────────────────────
    for k in range(1, N_STEPS + 1):

        if est["eta_matrix"] >= est["eta_frac"]:
            refined = "matrix"
            matrix_bisection_count += 1
            h_mat_approx = CELL_SIZE / 2**matrix_bisection_count
            print(f"\n  step={k}  MATRIX bisection  h_mat≈{h_mat_approx:.4f}")

            new_mdg = next(factory_iter)
            pp.set_local_coordinate_projections(new_mdg)
            gmsh_frac = new_mdg.subdomains(dim=2)[0]
            new_mdg.replace_subdomains_and_interfaces(
                sd_map={gmsh_frac: current_frac_sd}
            )
            mdg = new_mdg

        else:
            refined = "fracture"
            h_frac /= 2.0
            h_mat_approx = CELL_SIZE / 2**matrix_bisection_count
            print(f"\n  step={k}  FRACTURE halved  h_frac→{h_frac:.4f}")

            new_frac_sd = _create_fracture_grid(h_frac)
            mdg.replace_subdomains_and_interfaces(sd_map={current_frac_sd: new_frac_sd})
            current_frac_sd = new_frac_sd

        est = _solve_and_estimate(mdg, frac_net, params, non_matching=True)

        records.append(StepRecord(
            step=k,
            h_mat_approx=h_mat_approx,
            h_frac=h_frac,
            n_matrix=est["sd_mat"].num_cells,
            n_frac=est["sd_frac"].num_cells,
            eta_matrix=est["eta_matrix"],
            eta_frac=est["eta_frac"],
            eta_intf=est["eta_intf"],
            eta_gm_perp=est["eta_gm_perp"],
            eta_gm_res=est["eta_gm_res"],
            majorant=est["majorant"],
            true_p=est["true_p"],
            true_u=est["true_u"],
            eff_p=est["eff_p"],
            eff_u=est["eff_u"],
            refined=refined,
        ))
        _print_step(records[-1])

    factory.close()
    return records


# ── Output helpers ────────────────────────────────────────────────────────────

def _print_step(r: StepRecord) -> None:
    print(
        f"    M={r.majorant:.4e}  I_p={r.eff_p:.3f}  I_u={r.eff_u:.3f}"
        f"  η_Ω³={r.eta_matrix:.4e}  η_Ω²={r.eta_frac:.4e}"
        f"  η_Γ²={r.eta_intf:.4e}  η_GM⊥={r.eta_gm_perp:.2e}  η_GM,R={r.eta_gm_res:.2e}"
        f"  n3d={r.n_matrix}  n2d={r.n_frac}  [{r.refined}]"
    )


def _write_csv(records: List[StepRecord]) -> None:
    path = OUTDIR / "guided_halving_history.csv"
    header = (
        "step,h_mat,h_frac,n_matrix,n_frac,"
        "eta_matrix,eta_frac,eta_intf,eta_gm_perp,eta_gm_res,"
        "majorant,true_p,true_u,eff_p,eff_u,refined"
    )
    lines = [header]
    for r in records:
        lines.append(
            f"{r.step},{r.h_mat_approx:.6f},{r.h_frac:.6f},"
            f"{r.n_matrix},{r.n_frac},"
            f"{r.eta_matrix:.6e},{r.eta_frac:.6e},{r.eta_intf:.6e},"
            f"{r.eta_gm_perp:.6e},{r.eta_gm_res:.6e},"
            f"{r.majorant:.6e},{r.true_p:.6e},{r.true_u:.6e},"
            f"{r.eff_p:.6f},{r.eff_u:.6f},{r.refined}"
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"\nCSV → {path.resolve()}")


def _build_latex_table(records: List[StepRecord]) -> str:
    lines: list[str] = []
    lines.append(r"\begin{table}[tbp]")
    lines.append(
        r"    \caption{Guided subdomain halving for the 3D benchmark."
        r" At each step the subdomain with the larger indicator ($\eta_{\Omega^3}$"
        r" or $\eta_{\Omega^2}$) is bisected independently while the other remains"
        r" fixed; the configuration is non-matching from step~1 onward."
        r" $|\mathcal{T}_{\Omega^3}|$ and $|\mathcal{T}_{\Omega^2}|$ denote the"
        r" cell counts in the bulk and fracture subdomains, respectively."
        r" The grid-mismatch indicators $\eta_{\mathrm{GM},\perp}$ and"
        r" $\eta_{\mathrm{GM},R}$ contribute marginally to $\mathcal{M}^+$"
        r" and are omitted from the table for brevity."
        r" $I_{\mathrm{eff},p} = \mathcal{M}^+ / \|e_p\|$ and"
        r" $I_{\mathrm{eff},u} = \mathcal{M}^+ / \|e_u\|$.}"
    )
    lines.append(r"    \label{tab:guided_halving}")
    lines.append(r"    \centering")
    lines.append(r"    \begin{tabular}{r r r c c c c c c}")
    lines.append(r"        \toprule")
    lines.append(
        r"        $k$ & $|\mathcal{T}_{\Omega^3}|$ & $|\mathcal{T}_{\Omega^2}|$ & "
        r"$\eta_{\Omega^3}$ & $\eta_{\Omega^2}$ & $\eta_{\Gamma^2}$ & "
        r"$\mathcal{M}^+$ & $I_{\mathrm{eff},p}$ & $I_{\mathrm{eff},u}$ \\"
    )
    lines.append(r"        \midrule")
    for r in records:
        lines.append(
            f"        {r.step} & {r.n_matrix} & {r.n_frac} & "
            f"{FMT.format(r.eta_matrix)} & {FMT.format(r.eta_frac)} & "
            f"{FMT.format(r.eta_intf)} & "
            f"{FMT.format(r.majorant)} & "
            f"{r.eff_p:.3f} & {r.eff_u:.3f} \\\\"
        )
    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _print_summary(records: List[StepRecord]) -> None:
    header = (
        f"{'k':>3}  {'n_3D':>7}  {'n_2D':>5}  "
        f"{'η_Ω³':>10}  {'η_Ω²':>10}  {'η_Γ²':>10}  "
        f"{'η_GM⊥':>10}  {'η_GM,R':>10}  "
        f"{'M+':>10}  {'I_p':>6}  {'I_u':>6}  {'Refined':<9}"
    )
    sep = "─" * len(header)
    print("\n" + "═" * len(header))
    print(header)
    print(sep)
    for r in records:
        print(
            f"{r.step:>3}  {r.n_matrix:>7}  {r.n_frac:>5}  "
            f"{FMT.format(r.eta_matrix):>10}  {FMT.format(r.eta_frac):>10}  "
            f"{FMT.format(r.eta_intf):>10}  "
            f"{FMT.format(r.eta_gm_perp):>10}  {FMT.format(r.eta_gm_res):>10}  "
            f"{FMT.format(r.majorant):>10}  "
            f"{r.eff_p:>6.3f}  {r.eff_u:>6.3f}  {r.refined:<9}"
        )
    print("═" * len(header))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    solid = pp.SolidConstants(**manu_incomp_solid)   # type: ignore[arg-type]
    fluid = pp.FluidComponent(**manu_incomp_fluid)   # type: ignore[arg-type]
    params: dict = {
        "grid_type": "simplex",
        "material_constants": {"solid": solid, "fluid": fluid},
        "meshing_arguments": {"cell_size": CELL_SIZE},
        "times_to_export": [],
    }

    print("\n=== Guided subdomain halving ===")
    records = run_guided_halving(params)

    _print_summary(records)
    _write_csv(records)

    tex = _build_latex_table(records)
    tex_path = OUTDIR / "guided_halving_table.tex"
    tex_path.write_text(tex)
    print(f"LaTeX → {tex_path.resolve()}")
    print()
    print(tex)


if __name__ == "__main__":
    main()
