# Minimal InternalBoundaryGrid:
# - builds per-side 2D internal-boundary grids from sd_high (3D)
# - uses the mortar's (side-grid) rotation as reference
# - exposes: side grids (pp.TriangleGrid), side projectors (mortar→side),
#            involved high faces per side, and parent face maps
#
# Assumes you already have:
#   - ear_clip_triangulate, ensure_ccw, merge_close_vertices
#   - mdnme.RotatedGrid
#   - numpy as np, scipy.sparse as sps, porepy as pp

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import scipy.sparse as sps
import porepy as pp
import mdnme

from mdnme.utils.grid_utils import (
    ear_clip_triangulate,
    ensure_ccw,
    merge_close_vertices,
    merge_close_vertices_3d,
)


@dataclass
class _SideData:
    mortar_to_side: sps.spmatrix           # selector/projector from mortar → this side
    mortar_side_grid: pp.Grid              # mortar side grid (from MortarGrid)
    ibg_grid: pp.TriangleGrid              # built 2D internal-boundary grid (this side)
    high_faces: np.ndarray                 # indices of 3D faces used for this side
    parent_face_of_cell: np.ndarray        # len == ibg_grid.num_cells (IBG cell → 3D face)


class InternalBoundaryGrid:
    """
    Minimal per-interface Internal Boundary Grid (IBG).

    Input:
        intf:  pp.MortarGrid          (2D interface; no fracture involvement)
        sd_high: pp.Grid              (3D host domain grid)
        tol: float

    Provides, per mortar side:
        - mortar_to_side projector (from MortarGrid)
        - mortar-side grid (pp.Grid)
        - internal-boundary side grid (pp.TriangleGrid) constructed from sd_high faces
        - list of high faces used on that side
        - parent_face_of_cell map for the IBG grid
    """
    def __init__(self, intf: pp.MortarGrid, sd_high: pp.Grid, tol: float = 1e-8, name: str = "ibg"):
        if intf.dim != 2:
            raise NotImplementedError("InternalBoundaryGrid currently implemented for 2D mortar interfaces.")

        self.intf = intf
        self.sd_high = sd_high
        self.tol = tol
        self.name = name

        # Use the FIRST mortar side's rotation as reference so IBG is co-planar with both sides.
        # (Sides coincide geometrically, so any side works as reference.)
        _first_side_grid = next(iter(self.intf.side_grids.values()))
        self._R_ref = mdnme.RotatedGrid(_first_side_grid).rotation_matrix

        # Cache primary→mortar mapping (averaged) to filter high faces per side
        self._Pprim_to_mortar = self.intf.primary_to_mortar_avg()  # (n_mortar, n_primary_faces)

        # Build side data
        self._sides: Dict[object, _SideData] = {}
        for P_side, g_side in self.intf.project_to_side_grids():
            side_enum = self._enum_of_side_grid(g_side)
            faces_side = self._faces_for_side(P_side)                            # high faces for this side
            ibg_grid, parent_map = self._build_ibg_for_faces(faces_side, f"{name}_{side_enum.name.lower()}")
            self._sides[side_enum] = _SideData(
                mortar_to_side=P_side.tocsc(),
                mortar_side_grid=g_side,
                ibg_grid=ibg_grid,
                high_faces=faces_side,
                parent_face_of_cell=parent_map,
            )

    # ----------------- Public API -----------------

    def sides(self):
        """Iterator over the side enums present on the mortar."""
        for s in self._sides.keys():
            yield s

    def mortar_to_side(self, side) -> sps.spmatrix:
        """Selector/projector from global mortar ordering → this side's ordering."""
        return self._sides[side].mortar_to_side

    def mortar_side_grid(self, side) -> pp.Grid:
        """The mortar grid object for this side (from MortarGrid)."""
        return self._sides[side].mortar_side_grid

    def ibg_side_grid(self, side) -> pp.TriangleGrid:
        """The constructed internal-boundary 2D grid for this side."""
        return self._sides[side].ibg_grid

    def high_faces(self, side) -> np.ndarray:
        """3D high-face indices used by this IBG side (feed to your trace routine)."""
        return self._sides[side].high_faces

    def parent_face_of_cell(self, side) -> np.ndarray:
        """For each IBG cell, the originating 3D face index (aligns per-cell data)."""
        return self._sides[side].parent_face_of_cell

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Reference rotation (from mortar side); useful if you need consistent coords."""
        return self._R_ref

    # ----------------- Internals -----------------

    def _enum_of_side_grid(self, g: pp.Grid):
        for k, v in self.intf.side_grids.items():
            if v is g:
                return k
        raise KeyError("Side grid not found among MortarGrid.side_grids")

    def _faces_for_side(self, P_side: sps.spmatrix) -> np.ndarray:
        """
        Determine which 3D faces (primary) contribute to THIS mortar side:
            faces_side = { j | sum_i (P_side * Pprim_to_mortar)[i,j] > 0 }.
        """
        filt = (P_side @ self._Pprim_to_mortar).tocsc()     # (n_side_cells, n_primary_faces)
        mask = np.asarray(filt.sum(axis=0)).ravel() > 0
        faces = np.nonzero(mask)[0]
        return faces.astype(int)

    def _build_ibg_for_faces(self, faces: np.ndarray, name: str) -> Tuple[
        pp.TriangleGrid, np.ndarray]:
        if faces.size == 0:
            g = pp.TriangleGrid(np.zeros((3, 0)), np.zeros((3, 0), dtype=int),
                                name=name);
            g.compute_geometry()
            return g, np.zeros((0,), dtype=int)

        # rotation & plane selection from a mortar side
        some_side_grid = next(iter(self.intf.side_grids.values()))
        rot = mdnme.RotatedGrid(some_side_grid)
        R = rot.rotation_matrix
        dim_bool = rot.dim_bool  # two in-plane axes True

        nodes3d_all = self.sd_high.nodes  # (3, N)
        nodes2d_all = (R @ nodes3d_all)[dim_bool, :]  # (2, N)

        fn = self.sd_high.face_nodes.tocsc()

        tris_local = []
        parent = []
        verts3d = []
        pt_to_idx = {}

        for f in faces:
            start, end = fn.indptr[f], fn.indptr[f + 1]
            f_nodes = fn.indices[start:end]
            if f_nodes.size < 3:
                continue

            pts2d = nodes2d_all[:, f_nodes].T  # (k,2)
            pts3d = nodes3d_all[:, f_nodes].T  # (k,3)

            # dedupe consecutive in 2D
            poly2d = []
            poly3d = []
            for (p2, p3) in zip(map(tuple, pts2d), map(tuple, pts3d)):
                if not poly2d or (abs(p2[0] - poly2d[-1][0]) > self.tol or abs(
                        p2[1] - poly2d[-1][1]) > self.tol):
                    poly2d.append(p2)
                    poly3d.append(p3)
            if len(poly2d) < 3:
                continue

            # ear-clip in 2D to get triangle vertex triplets (indices in poly list)
            if len(poly2d) == 3:
                # ensure CCW by 2D cross
                x0, y0 = poly2d[0]
                x1, y1 = poly2d[1]
                x2, y2 = poly2d[2]
                cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
                local_tris = [[0, 1, 2]] if cross >= 0 else [[0, 2, 1]]
            else:
                local_tris = ear_clip_triangulate(poly2d, tol=self.tol)
                # ear_clip_triangulate returns coords; convert to local indices
                # build map coord->index (stable)
                idx_map = {tuple(poly2d[i]): i for i in range(len(poly2d))}
                local_tris = [[idx_map[tuple(a)], idx_map[tuple(b)], idx_map[tuple(c)]]
                              for (a, b, c) in local_tris]

            # add triangles; register vertices in 3D
            for tri in local_tris:
                gtri = []
                for vid in tri:
                    p3 = tuple(poly3d[vid])
                    if p3 not in pt_to_idx:
                        pt_to_idx[p3] = len(verts3d)
                        verts3d.append(p3)
                    gtri.append(pt_to_idx[p3])
                tris_local.append(gtri)
                parent.append(int(f))

        if not tris_local:
            g = pp.TriangleGrid(np.zeros((3, 0)), np.zeros((3, 0), dtype=int),
                                name=name);
            g.compute_geometry()
            return g, np.zeros((0,), dtype=int)

        # merge close 3D vertices (shared edges across faces)
        coords3d, cells = merge_close_vertices_3d(verts3d, tris_local, tol=self.tol)

        # orientation ensure: run ensure_ccw on 2D shadow (safe since all tris are coplanar)
        shadow2d = (R @ np.vstack(
            (coords3d,)))  # WRONG shape; we already have coords3d (3,N)
        # simpler: project to 2D using the same frame:
        coords2d = (R @ coords3d)  # (3,N) -> (3,N) rotated; pick in-plane axes:
        coords2d = coords2d[dim_bool, :]
        cells = ensure_ccw(cells, coords2d)

        tri_arr = np.array(cells).T
        g2d = pp.TriangleGrid(coords3d, tri_arr, name=name)  # NOTE: 3×N nodes
        g2d.compute_geometry()

        parent = np.asarray(parent, dtype=int)
        if parent.size != g2d.num_cells:
            raise RuntimeError("Parent-face map size mismatch after IBG assembly.")
        return g2d, parent
