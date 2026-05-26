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
FMT = "{:.2e}"
OUTDIR = pathlib.Path(__file__).parent
CSV_RAW = OUTDIR / "results_raw.csv"


@dataclass
class Metrics:
    h: float
    # Subdomain NM indicators
    eta_nm_omega2: float    # sqrt(diff_mat + resi_mat)   — bulk 3D
    eta_nm_omega1: float    # sqrt(diff_frac + resi_frac) — fracture 2D
    # Interface NM indicators (left = Γ₁, right = Γ₂)
    eta_nm_gamma1: float    # sqrt(diff_intf_left)
    eta_nm_gamma2: float    # sqrt(diff_intf_right)
    # GM indicators (zero for matching)
    eta_gm_gamma1: float    # sqrt(gm_perp_left)
    eta_gm_gamma2: float    # sqrt(gm_perp_right)
    eta_gm_omega1: float    # sqrt(gm_residual_frac)
    # Global majorant (includes GM terms for NM)
    majorant: float
    # True errors and effectivity indices
    true_p: float
    true_u: float
    eff_p: float
    eff_u: float


def _material_constants() -> Dict[str, pp.PhysicalConstants]:  # type:ignore
    solid_constants = pp.SolidConstants(**manu_incomp_solid)  # type:ignore[arg-type]
    fluid_constants = pp.FluidComponent(**manu_incomp_fluid)  # type:ignore[arg-type]
    return {"solid": solid_constants, "fluid": fluid_constants}


def _split_interface_lr(
    diff_intf: np.ndarray, n_cells: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Split interface array into left (Γ₁) and right (Γ₂) halves."""
    half = n_cells // 2
    left = diff_intf[half:]
    right = diff_intf[:half]
    return left, right


def _run_single(
    h: float, *, non_matching: bool, translation: Tuple[int, int, int] | None
) -> Metrics:
    material_constants = _material_constants()

    common_params = {
        "grid_type": "simplex",
        "material_constants": material_constants,
        "meshing_arguments": {"cell_size": h},
        "times_to_export": [],
    }

    if non_matching:
        assert translation is not None, "translation must be provided in non-matching runs"
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
    pp.run_time_dependent_model(setup, {})
    estimate_errors(setup.mdg, is_non_matching=non_matching)

    mdg = setup.mdg

    ((sd_mat, d_mat),) = mdg.subdomains(dim=3, return_data=True)
    ((sd_frac, d_frac),) = mdg.subdomains(dim=2, return_data=True)
    ((intf, d_intf),) = mdg.interfaces(dim=2, return_data=True)

    diff_sd_mat = np.asarray(d_mat["estimates"]["diffusive_error"])
    diff_sd_frac = np.asarray(d_frac["estimates"]["diffusive_error"])
    diff_intf_arr = np.asarray(d_intf["estimates"]["diffusive_error"])

    diff_intf_left, diff_intf_right = _split_interface_lr(diff_intf_arr, intf.num_cells)

    resi_sd_mat = float(setup.exact_sol.residual_error_matrix(sd_mat, d_mat).sum())
    resi_sd_frac = float(setup.exact_sol.residual_error_fracture(sd_frac, d_frac).sum())

    eta_nm_omega2 = math.sqrt(float(diff_sd_mat.sum()) + resi_sd_mat)
    eta_nm_omega1 = math.sqrt(float(diff_sd_frac.sum()) + resi_sd_frac)
    eta_nm_gamma1 = math.sqrt(float(diff_intf_left.sum()))
    eta_nm_gamma2 = math.sqrt(float(diff_intf_right.sum()))

    # GM terms — present only for NM runs; .get() returns None for matching
    gm_perp_raw = d_intf["estimates"].get("gm_diffusive_error_perp", None)
    if gm_perp_raw is not None:
        gm_perp_arr = np.asarray(gm_perp_raw, dtype=float)
        gm_perp_left, gm_perp_right = _split_interface_lr(gm_perp_arr, intf.num_cells)
        eta_gm_gamma1 = math.sqrt(float(gm_perp_left.sum()))
        eta_gm_gamma2 = math.sqrt(float(gm_perp_right.sum()))
        gm_perp_total = float(gm_perp_arr.sum())
    else:
        eta_gm_gamma1 = 0.0
        eta_gm_gamma2 = 0.0
        gm_perp_total = 0.0

    gm_res_raw = d_frac["estimates"].get("gm_residual_error", None)
    if gm_res_raw is not None:
        gm_res_arr = np.asarray(gm_res_raw, dtype=float)
        eta_gm_omega1 = math.sqrt(float(gm_res_arr.sum()))
        gm_res_total = float(gm_res_arr.sum())
    else:
        eta_gm_omega1 = 0.0
        gm_res_total = 0.0

    # Majorant: sqrt(Σ_diff + η²_GM,⊥) + sqrt(Σ_resi + η²_GM,R)
    diff_sum = (
        float(diff_sd_mat.sum())
        + float(diff_sd_frac.sum())
        + float(diff_intf_arr.sum())
    )
    resi_sum = resi_sd_mat + resi_sd_frac
    majorant = math.sqrt(diff_sum + gm_perp_total) + math.sqrt(resi_sum + gm_res_total)

    te = VarelaJNumTrueErrors3D(setup)
    true_p = float(te.true_error_primal())
    true_u = float(te.true_error_dual())
    eff_p = majorant / true_p if true_p > 0 else float("nan")
    eff_u = majorant / true_u if true_u > 0 else float("nan")

    return Metrics(
        h=h,
        eta_nm_omega2=eta_nm_omega2,
        eta_nm_omega1=eta_nm_omega1,
        eta_nm_gamma1=eta_nm_gamma1,
        eta_nm_gamma2=eta_nm_gamma2,
        eta_gm_gamma1=eta_gm_gamma1,
        eta_gm_gamma2=eta_gm_gamma2,
        eta_gm_omega1=eta_gm_omega1,
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
    """Compute mean/std across Metrics fields except h."""
    assert len(ms) > 0
    keys = [
        "eta_nm_omega2",
        "eta_nm_omega1",
        "eta_nm_gamma1",
        "eta_nm_gamma2",
        "eta_gm_gamma1",
        "eta_gm_gamma2",
        "eta_gm_omega1",
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


def _write_csv(records: List[Dict]) -> None:
    header = (
        "h,case,tr_x,tr_y,tr_z,"
        "eta_nm_omega2,eta_nm_omega1,eta_nm_gamma1,eta_nm_gamma2,"
        "eta_gm_gamma1,eta_gm_gamma2,eta_gm_omega1,"
        "majorant,true_p,true_u,eff_p,eff_u"
    )
    lines = [header]
    for r in records:
        lines.append(
            f"{r['h']},{r['case']},"
            f"{r.get('tr_x', '')},"
            f"{r.get('tr_y', '')},"
            f"{r.get('tr_z', '')},"
            f"{r['eta_nm_omega2']:.6e},{r['eta_nm_omega1']:.6e},"
            f"{r['eta_nm_gamma1']:.6e},{r['eta_nm_gamma2']:.6e},"
            f"{r['eta_gm_gamma1']:.6e},{r['eta_gm_gamma2']:.6e},"
            f"{r['eta_gm_omega1']:.6e},"
            f"{r['majorant']:.6e},{r['true_p']:.6e},{r['true_u']:.6e},"
            f"{r['eff_p']:.6f},{r['eff_u']:.6f}"
        )
    CSV_RAW.write_text("\n".join(lines) + "\n")
    print(f"Raw results written to: {CSV_RAW.resolve()}")


def main() -> None:
    _ = _material_constants()

    csv_records: List[Dict] = []

    for h in CELL_SIZES:
        print(f"\n=== h = {h:.4f} | Matching ===")
        m_match = _run_single(h, non_matching=False, translation=None)

        csv_records.append({
            "h": h,
            "case": 0,
            "eta_nm_omega2": m_match.eta_nm_omega2,
            "eta_nm_omega1": m_match.eta_nm_omega1,
            "eta_nm_gamma1": m_match.eta_nm_gamma1,
            "eta_nm_gamma2": m_match.eta_nm_gamma2,
            "eta_gm_gamma1": m_match.eta_gm_gamma1,
            "eta_gm_gamma2": m_match.eta_gm_gamma2,
            "eta_gm_omega1": m_match.eta_gm_omega1,
            "majorant":      m_match.majorant,
            "true_p":        m_match.true_p,
            "true_u":        m_match.true_u,
            "eff_p":         m_match.eff_p,
            "eff_u":         m_match.eff_u,
        })

        print(f"=== h = {h:.4f} | Non-matching (8 translations) ===")
        ms: List[Metrics] = []
        for tr in TRANSLATIONS:
            print(f"  -> translation = {tr}")
            m_nm = _run_single(h, non_matching=True, translation=tr)
            ms.append(m_nm)
            csv_records.append({
                "h": h,
                "case": 1,
                "tr_x": tr[0],
                "tr_y": tr[1],
                "tr_z": tr[2],
                "eta_nm_omega2": m_nm.eta_nm_omega2,
                "eta_nm_omega1": m_nm.eta_nm_omega1,
                "eta_nm_gamma1": m_nm.eta_nm_gamma1,
                "eta_nm_gamma2": m_nm.eta_nm_gamma2,
                "eta_gm_gamma1": m_nm.eta_gm_gamma1,
                "eta_gm_gamma2": m_nm.eta_gm_gamma2,
                "eta_gm_omega1": m_nm.eta_gm_omega1,
                "majorant":      m_nm.majorant,
                "true_p":        m_nm.true_p,
                "true_u":        m_nm.true_u,
                "eff_p":         m_nm.eff_p,
                "eff_u":         m_nm.eff_u,
            })

    _write_csv(csv_records)


if __name__ == "__main__":
    main()
