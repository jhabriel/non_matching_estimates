"""Matching vs non-matching study for the 3D fracture network with small features
(Flow Benchmark 3D Case 3).

No exact solution is available.  Runs one matching and one non-matching solve,
reports majorant and per-dimension local error indicators, and exports a CSV.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import porepy as pp

from mdnme.estimates.error_estimation import aggregate_local_errors, get_majorant
from mdnme.examples.example_3.model import SmallFeaturesModel, solid_constants

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
SOURCE_RATE: float = 0.1
FMT = "{:.4e}"
OUTDIR = pathlib.Path(".")
CSV_RAW = OUTDIR / "results_small_features.csv"


@dataclass
class Metrics:
    non_matching: bool
    majorant: float
    sd_error: Dict[int, float] = field(default_factory=dict)
    intf_error: Dict[int, float] = field(default_factory=dict)


def _run_single(*, non_matching: bool) -> Metrics:
    params = {
        "material_constants": {"solid": solid_constants},
        "non_matching": non_matching,
        "export_to_vtu": False,
        "times_to_export": [],
        "refinement": "nested",
        "matching_from_geo": True,
        "source_rate": SOURCE_RATE,
    }
    model = SmallFeaturesModel(params)  # type: ignore[arg-type]
    pp.run_time_dependent_model(model, params)

    mdg = model.mdg
    local = aggregate_local_errors(mdg)
    majorant = get_majorant(mdg)

    return Metrics(
        non_matching=non_matching,
        majorant=majorant,
        sd_error=local["subdomain_error"],
        intf_error=local["interface_error"],
    )


def _print_summary(metrics: List[Metrics]) -> None:
    header = (
        f"{'NM':>3}  {'majorant':>12}"
        f"  {'eta_3D':>12}  {'eta_2D':>12}  {'eta_1D':>12}"
        f"  {'intf_2D':>12}  {'intf_1D':>12}"
    )
    print(header)
    print("-" * len(header))
    for m in metrics:
        print(
            f"{'Y' if m.non_matching else 'N':>3}"
            f"  {FMT.format(m.majorant):>12}"
            f"  {FMT.format(m.sd_error.get(3, np.nan)):>12}"
            f"  {FMT.format(m.sd_error.get(2, np.nan)):>12}"
            f"  {FMT.format(m.sd_error.get(1, np.nan)):>12}"
            f"  {FMT.format(m.intf_error.get(2, np.nan)):>12}"
            f"  {FMT.format(m.intf_error.get(1, np.nan)):>12}"
        )


def _export_csv(metrics: List[Metrics]) -> None:
    try:
        import pandas as pd

        records = []
        for m in metrics:
            records.append(
                {
                    "non_matching": int(m.non_matching),
                    "majorant": m.majorant,
                    "sd_error_3d": m.sd_error.get(3, np.nan),
                    "sd_error_2d": m.sd_error.get(2, np.nan),
                    "sd_error_1d": m.sd_error.get(1, np.nan),
                    "intf_error_2d": m.intf_error.get(2, np.nan),
                    "intf_error_1d": m.intf_error.get(1, np.nan),
                }
            )
        pd.DataFrame.from_records(records).to_csv(CSV_RAW, index=False)
        print(f"\nRaw results written to: {CSV_RAW.resolve()}")
    except Exception as exc:
        print(f"(Skipping CSV export: {exc})")


def main() -> None:
    all_metrics: List[Metrics] = []

    print("\n=== Matching ===")
    all_metrics.append(_run_single(non_matching=False))

    print("\n=== Non-matching ===")
    all_metrics.append(_run_single(non_matching=True))

    print("\n\n=== Summary ===")
    _print_summary(all_metrics)
    _export_csv(all_metrics)


if __name__ == "__main__":
    main()
