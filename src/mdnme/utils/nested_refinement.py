# geo_refine_factory.py
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Literal, Optional

import gmsh
import porepy as pp


class GeoNestedRefinementFactory:
    """
    Factory to generate a sequence of MDGs by nested global refinement in Gmsh,
    starting from an *existing* geometry (.geo) or mesh (.msh).

    Usage:
        fac = GeoNestedRefinementFactory(
            src_path="case.geo",      # or "case.msh"
            dim=3,                    # network dimension for DFM import (2 or 3)
            num_refinements=3,        # how many 'refine()' steps to perform
            out_stem="nm_conv",       # basename for written .msh files (optional)
            gmsh_opts={...},          # optional dict of gmsh options
        )
        for mdg in fac:
            # mdg_0 is the initial mesh, then mdg_1 after one refine(), etc.
            ...

    Notes:
        - We *do not* finalize GMsh until iteration is done; call .close() or let the
          iterator finish.
        - Each refinement writes a .msh to disk (handy for debugging/reproducibility).
        - You can use your existing replacement routine on the returned MDGs to build
          non-matching cases (e.g., keep 3D from level 0, replace 2D/1D from level k).
    """

    def __init__(
        self,
        src_path: str | Path,
        dim: Literal[2, 3],
        num_refinements: int,
        out_stem: Optional[str] = None,
        gmsh_opts: Optional[dict] = None,
    ) -> None:
        self.src_path = Path(src_path)
        if not self.src_path.exists():
            raise FileNotFoundError(self.src_path)
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3 (DFM import target dimension).")
        self.dim = int(dim)
        self.num_refinements = int(num_refinements)
        self.out_stem = out_stem or self.src_path.stem
        self.gmsh_opts = dict(gmsh_opts or {})
        self._initialized = False
        self._counter = -1  # -1 -> initial mesh (no refine yet)

    # --- context / iterator sugar ---
    def __iter__(self) -> Iterator[pp.MixedDimensionalGrid]:
        return self._generator()

    def close(self) -> None:
        if self._initialized:
            gmsh.finalize()
            self._initialized = False

    # --- internals ---
    def _init_gmsh_and_generate_initial(self) -> None:
        gmsh.initialize()
        self._initialized = True

        # Apply any user options early.
        for k, v in self.gmsh_opts.items():
            gmsh.option.setNumber(k, float(v))

        # Open either a .geo (build + mesh) or a .msh (ready mesh)
        if self.src_path.suffix.lower() == ".geo":
            gmsh.open(str(self.src_path))
            # ensure entities are built and meshable
            try:
                gmsh.model.geo.synchronize()
            except Exception:
                # If geometry is “already CAD”, synchronize may be a no-op.
                pass
            gmsh.model.mesh.generate(self.dim)  # initial mesh from .geo
        elif self.src_path.suffix.lower() == ".msh":
            gmsh.open(str(self.src_path))
        else:
            raise ValueError("src_path must be a .geo or .msh file")

        # Write the level-0 mesh so we can import deterministically
        gmsh.write(f"{self.out_stem}_0.msh")
        self._counter = 0

    def _write_and_import(self) -> pp.MixedDimensionalGrid:
        out_name = f"{self.out_stem}_{self._counter}.msh"
        gmsh.write(out_name)
        # Import to PorePy; this builds the mixed-dimensional grid with mortars, etc.
        mdg = pp.fracture_importer.dfm_from_gmsh(out_name, self.dim)
        # Set local coordinate projections (important for your IBG/TG/SZ pipeline)
        pp.set_local_coordinate_projections(mdg)
        return mdg

    def _generator(self) -> Iterator[pp.MixedDimensionalGrid]:
        try:
            if not self._initialized:
                self._init_gmsh_and_generate_initial()

            # yield level 0 (initial mesh)
            yield self._write_and_import()

            # nested refinements: refine the *current* gmsh model in place
            for k in range(1, self.num_refinements + 1):
                gmsh.model.mesh.refine()      # global uniform refinement
                self._counter = k
                yield self._write_and_import()
        finally:
            self.close()
