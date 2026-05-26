"""Convergence study for the Geiger 3D benchmark (Flow Benchmark 3D Case 2).

Parallel version: the three refinement levels are run concurrently using
separate processes (max_workers=3). Within each worker, matching and
non-matching are still sequential so that gmsh state is not shared.

No exact solution is available, so we report majorant and local error
indicators per subdomain/interface dimension only.
"""

from __future__ import annotations

import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import porepy as pp

from mdnme.estimates.error_estimation import aggregate_local_errors, get_majorant
from mdnme.examples.example_2.flow_benchmark_3d_case_2 import solid_constants_conductive
from mdnme.examples.example_2.model import Geiger3dModel

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
REFINEMENT_LEVELS: List[int] = [0, 1, 2]
MAX_WORKERS: int = 3
FMT = "{:.4e}"
OUTDIR = pathlib.Path(".")
CSV_RAW = OUTDIR / "results_geiger3d.csv"


@dataclass
class Metrics:
    refinement_level: int
    non_matching: bool
    majorant: float
    sd_error: Dict[int, float] = field(default_factory=dict)
    intf_error: Dict[int, float] = field(default_factory=dict)


def _run_single(refinement_level: int, *, non_matching: bool) -> Metrics:
    params = {
        "material_constants": {"solid": solid_constants_conductive},
        "refinement_level": refinement_level,
        "non_matching": non_matching,
        "times_to_export": [],
        "export_results": False,
    }
    model = Geiger3dModel(params)  # type: ignore[arg-type]
    pp.run_time_dependent_model(model, params)

    mdg = model.mdg
    local = aggregate_local_errors(mdg)
    majorant = get_majorant(mdg)

    return Metrics(
        refinement_level=refinement_level,
        non_matching=non_matching,
        majorant=majorant,
        sd_error=local["subdomain_error"],
        intf_error=local["interface_error"],
    )


def _run_level(refinement_level: int) -> List[Metrics]:
    """Run matching then non-matching for one refinement level (one worker)."""
    results = []
    print(f"\n=== Refinement level {refinement_level} | Matching ===", flush=True)
    results.append(_run_single(refinement_level, non_matching=False))
    print(f"\n=== Refinement level {refinement_level} | Non-matching ===", flush=True)
    results.append(_run_single(refinement_level, non_matching=True))
    return results


def _print_summary(metrics: List[Metrics]) -> None:
    header = (
        f"{'lvl':>4} {'NM':>3}  {'majorant':>12}"
        f"  {'eta_3D':>12}  {'eta_2D':>12}  {'eta_1D':>12}"
        f"  {'intf_2D':>12}  {'intf_1D':>12}  {'intf_0D':>12}"
    )
    print(header)
    print("-" * len(header))
    for m in metrics:
        print(
            f"{m.refinement_level:>4} {'Y' if m.non_matching else 'N':>3}"
            f"  {FMT.format(m.majorant):>12}"
            f"  {FMT.format(m.sd_error.get(3, np.nan)):>12}"
            f"  {FMT.format(m.sd_error.get(2, np.nan)):>12}"
            f"  {FMT.format(m.sd_error.get(1, np.nan)):>12}"
            f"  {FMT.format(m.intf_error.get(2, np.nan)):>12}"
            f"  {FMT.format(m.intf_error.get(1, np.nan)):>12}"
            f"  {FMT.format(m.intf_error.get(0, np.nan)):>12}"
        )


def _export_csv(metrics: List[Metrics]) -> None:
    header = (
        "refinement_level,non_matching,majorant,"
        "sd_error_3d,sd_error_2d,sd_error_1d,"
        "intf_error_2d,intf_error_1d,intf_error_0d"
    )
    lines = [header]
    for m in metrics:
        lines.append(
            f"{m.refinement_level},{int(m.non_matching)},"
            f"{m.majorant:.6e},"
            f"{m.sd_error.get(3, np.nan):.6e},"
            f"{m.sd_error.get(2, np.nan):.6e},"
            f"{m.sd_error.get(1, np.nan):.6e},"
            f"{m.intf_error.get(2, np.nan):.6e},"
            f"{m.intf_error.get(1, np.nan):.6e},"
            f"{m.intf_error.get(0, np.nan):.6e}"
        )
    CSV_RAW.write_text("\n".join(lines) + "\n")
    print(f"\nRaw results written to: {CSV_RAW.resolve()}")


def main() -> None:
    all_metrics: List[Metrics] = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_run_level, lvl): lvl for lvl in REFINEMENT_LEVELS}
        for future in as_completed(futures):
            lvl = futures[future]
            try:
                all_metrics.extend(future.result())
                print(f"\n[done] refinement level {lvl}", flush=True)
            except Exception as exc:
                print(f"\n[error] refinement level {lvl}: {exc}", flush=True)
                raise

    # Sort by (refinement_level, non_matching) to match sequential output order
    all_metrics.sort(key=lambda m: (m.refinement_level, m.non_matching))

    print("\n\n=== Summary ===")
    _print_summary(all_metrics)
    _export_csv(all_metrics)


if __name__ == "__main__":
    main()
