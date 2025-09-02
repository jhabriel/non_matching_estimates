from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import scipy.sparse as sps
import porepy as pp

from mdnme.utils.grid_utils import (
    ear_clip_triangulate,
    ensure_ccw,
    merge_close_vertices_3d,
)


@dataclass
class _SideData:
    mortar_to_side: sps.spmatrix
    mortar_side_grid: pp.Grid
    ibg_grid: pp.TriangleGrid
    high_faces: np.ndarray
    parent_face_of_cell: np.ndarray


class InternalBoundaryGrid:
    """Per-interface Internal Boundary Grid built in the interface frame."""

    def __init__(self, intf: pp.MortarGrid, sd_high: pp.Grid,
                 tol: float = 1e-8, name: str = "ibg"):
        if intf.dim != 2:
            raise NotImplementedError("IBG currently implemented for 2D mortar interfaces.")

        # required: assign_canonical_rotations() must have set these
        if not hasattr(intf, "rot_matrix") or intf.rot_matrix is None:
            raise RuntimeError("intf.rot_matrix missing; run assign_canonical_rotations().")
        if not hasattr(intf, "dim_bool") or intf.dim_bool is None:
            raise RuntimeError("intf.dim_bool missing; run assign_canonical_rotations().")

        self.intf = intf
        self.sd_high = sd_high
        self.tol = tol
        self.name = name

        # Interface (canonical) frame
        self.rot_matrix = intf.rot_matrix
        self.dim_bool = np.array(intf.dim_bool, dtype=bool)

        # Cache mortar←primary (avg) to figure out which high faces feed each side
        self._Pprim_to_mortar = self.intf.primary_to_mortar_avg()  # (n_mortar, n_primary_faces)

        # Build side data
        self._sides: Dict[object, _SideData] = {}
        self._side_order = []
        for P_side, g_side in self.intf.project_to_side_grids():
            side_enum = self._enum_of_side_grid(g_side)
            faces_side = self._faces_for_side(P_side)  # high faces contributing to this side
            ibg_grid, parent_map = self._build_ibg_for_faces(faces_side, f"{name}_{side_enum.name.lower()}")
            self._sides[side_enum] = _SideData(
                mortar_to_side=P_side.tocsc(),
                mortar_side_grid=g_side,
                ibg_grid=ibg_grid,
                high_faces=faces_side,
                parent_face_of_cell=parent_map,
            )
            self._side_order.append(side_enum)

        self._finalize_global_ibg_ordering()

    # ----------------- Public API -----------------

    def sides(self):
        for s in self._sides.keys():
            yield s

    def mortar_to_side(self, side) -> sps.spmatrix:
        return self._sides[side].mortar_to_side

    def mortar_side_grid(self, side) -> pp.Grid:
        return self._sides[side].mortar_side_grid

    def ibg_side_grid(self, side) -> pp.TriangleGrid:
        return self._sides[side].ibg_grid

    def high_faces(self, side) -> np.ndarray:
        return self._sides[side].high_faces

    def parent_face_of_cell(self, side) -> np.ndarray:
        return self._sides[side].parent_face_of_cell

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Expose the interface’s canonical frame (used by TransferGrid, etc.)."""
        return self.rot_matrix

    # ----------------- Internals -----------------

    def _enum_of_side_grid(self, g: pp.Grid):
        for k, v in self.intf.side_grids.items():
            if v is g:
                return k
        raise KeyError("Side grid not found among MortarGrid.side_grids")

    def _faces_for_side(self, P_side: sps.spmatrix) -> np.ndarray:
        """
        Faces contributing to THIS side:
            filt = P_side @ (primary->mortar)  -> shape (n_side_cells, n_primary_faces)
            take columns with any contribution.
        """
        filt = (P_side @ self._Pprim_to_mortar).tocsc()
        mask = np.asarray(filt.sum(axis=0)).ravel() > 0
        faces = np.nonzero(mask)[0]
        return faces.astype(int)

    def _build_ibg_for_faces(self, faces: np.ndarray, name: str) -> Tuple[pp.TriangleGrid, np.ndarray]:
        if faces.size == 0:
            g = pp.TriangleGrid(np.zeros((3, 0)), np.zeros((3, 0), dtype=int), name=name)
            g.compute_geometry()
            return g, np.zeros((0,), dtype=int)

        # project high nodes into interface 2D plane for polygon ops
        nodes3d_all = self.sd_high.nodes                      # (3, N)
        nodes2d_all = (self.rot_matrix @ nodes3d_all)[self.dim_bool, :]  # (2, N)

        fn = self.sd_high.face_nodes.tocsc()

        tris_local: list[list[int]] = []
        parent: list[int] = []
        verts3d: list[tuple[float, float, float]] = []
        pt_to_idx: dict[tuple[float, float, float], int] = {}

        for f in faces:
            i0, i1 = fn.indptr[f], fn.indptr[f + 1]
            f_nodes = fn.indices[i0:i1]
            if f_nodes.size < 3:
                continue

            pts2d = nodes2d_all[:, f_nodes].T  # (k,2)
            pts3d = nodes3d_all[:, f_nodes].T  # (k,3)

            # dedupe consecutive in 2D
            poly2d, poly3d = [], []
            for p2, p3 in zip(map(tuple, pts2d), map(tuple, pts3d)):
                if not poly2d or (abs(p2[0]-poly2d[-1][0]) > self.tol or abs(p2[1]-poly2d[-1][1]) > self.tol):
                    poly2d.append(p2); poly3d.append(p3)
            if len(poly2d) < 3:
                continue

            # triangulate in 2D
            if len(poly2d) == 3:
                # ensure CCW by 2D cross
                x0,y0 = poly2d[0]; x1,y1 = poly2d[1]; x2,y2 = poly2d[2]
                tri_idx = [[0,1,2]] if (x1-x0)*(y2-y0) - (y1-y0)*(x2-x0) >= 0 else [[0,2,1]]
            else:
                # ear_clip_triangulate returns list of triangles as coordinate triplets
                tris_coords = ear_clip_triangulate(poly2d, tol=self.tol)
                # map coords back to local indices
                idx_map = {tuple(poly2d[i]): i for i in range(len(poly2d))}
                tri_idx = [[idx_map[tuple(a)], idx_map[tuple(b)], idx_map[tuple(c)]] for (a,b,c) in tris_coords]

            # register triangles and global 3D vertices
            for tri in tri_idx:
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
            g = pp.TriangleGrid(np.zeros((3, 0)), np.zeros((3, 0), dtype=int), name=name)
            g.compute_geometry()
            return g, np.zeros((0,), dtype=int)

        # merge close 3D vertices across faces (to glue triangles)
        coords3d, cells = merge_close_vertices_3d(verts3d, tris_local, tol=self.tol)  # coords3d: (3, N)

        # orient consistently using the 2D shadow in the interface frame
        coords2d = (self.rot_matrix @ coords3d)[self.dim_bool, :]  # (2, N)
        cells = ensure_ccw(cells, coords2d)

        tri_arr = np.array(cells).T  # (3, n_cells)
        g2d = pp.TriangleGrid(coords3d, tri_arr, name=name)  # IBG stored in 3D coords
        g2d.compute_geometry()

        parent_arr = np.asarray(parent, dtype=int)
        if parent_arr.size != g2d.num_cells:
            raise RuntimeError("Parent-face map size mismatch after IBG assembly.")
        return g2d, parent_arr

# --- add to your InternalBoundaryGrid class ---

    def _finalize_global_ibg_ordering(self) -> None:
        """Build global IBG cell ordering and per-side offsets."""

        # compute offsets
        offset = 0
        self._offsets: dict[object, tuple[int,int]] = {}
        for side in self._side_order:
            n_side = self._sides[side].ibg_grid.num_cells
            self._offsets[side] = (offset, offset + n_side)
            offset += n_side
        self._n_total = offset

    def num_cells(self) -> int:
        """Total number of IBG cells across both sides."""
        return self._n_total

    def ibg_to_side(self, side) -> sps.csc_matrix:
        """Selector from global IBG ordering → this side’s IBG cells.

        Shape: (n_side_cells, n_ibg_total). Multiplying with an array of shape
        (n_ibg_total, ndofs) returns (n_side_cells, ndofs).
        """
        start, end = self._offsets[side]
        n_side = end - start
        if n_side == 0:
            return sps.csc_matrix((0, self._n_total))
        rows = np.arange(n_side)
        cols = rows + start
        data = np.ones(n_side, dtype=float)
        return sps.coo_matrix((data, (rows, cols)), shape=(n_side, self._n_total)).tocsc()

    def side_to_ibg(self, side) -> sps.csc_matrix:
        """Scatter from this side’s IBG cells → global IBG ordering.

        Shape: (n_ibg_total, n_side_cells).
        """
        return self.ibg_to_side(side).T.tocsc()

    def project_to_side_ibg(self):
        """Generator like MortarGrid.project_to_side_grids(), but for IBG.

        Yields tuples: (proj, ibg_side_grid) where
          - proj : (n_side_cells, n_ibg_total) selector
          - ibg_side_grid : pp.TriangleGrid for this side
        """
        for side in self._side_order:
            yield self.ibg_to_side(side), self._sides[side].ibg_grid
