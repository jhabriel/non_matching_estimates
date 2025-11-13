from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import scipy.sparse as sps

import mdnme
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


@dataclass
class _SideData1D:
    mortar_to_side: sps.spmatrix
    mortar_side_grid: pp.Grid
    ibg_grid: pp.Grid
    high_edges: np.ndarray
    parent_edge_of_cell: np.ndarray


class InternalBoundaryGrid:
    """Per-interface Internal Boundary Grid built in the interface frame."""

    def __init__(self, intf: pp.MortarGrid, sd_high: pp.Grid,
                 tol: float = 1e-8, name: str = "ibg"):
        if intf.dim != 2:
            raise NotImplementedError("IBG currently implemented for 2D mortar interfaces.")

        self.intf = intf
        self.sd_high = sd_high
        self.tol = tol
        self.name = name

        # Interface (canonical) frame
        self.rot_matrix, self.dim_bool, _ = mdnme.canonical_frame(intf)

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


class InternalBoundaryLineGrid:
    """Per-interface Internal Boundary 1D Grids for traces of 2D subdomains.

    Usage pattern mirrors InternalBoundaryGrid (2D):

        ibg1d = InternalBoundaryLineGrid(intf, sd_high, tol)

        for P_side, mg_side in intf.project_to_side_grids():
            side_enum   = ...   # same as you do for 2D
            ibg_side    = ibg1d.ibg_side_grid(side_enum)
            parent_edge = ibg1d.parent_edge_of_cell(side_enum)
            ...

    Internally:
    - One instance per interface.
    - For each mortar side, we build a side-specific 1D TensorGrid in the
      interface canonical frame, plus a map from IBG cells to parent high-edges.
    - We also maintain a global ordering of all IBG cells across sides
      (ibg_to_side / side_to_ibg) like in the 2D IBG.
    """

    def __init__(
        self,
        intf: pp.MortarGrid,
        sd_high: pp.Grid,
        tol: float = 1e-8,
        name: str = "ibg1d",
    ):
        if intf.dim != 1:
            raise NotImplementedError(
                "InternalBoundaryLineGrid expects a 1D mortar interface."
            )

        self.intf = intf
        self.sd_high = sd_high
        self.tol = tol
        self.name = name

        # Canonical interface frame (same as 2D IBG)
        self.rot_matrix, self.dim_bool, _ = mdnme.canonical_frame(intf)

        # High (2D) → mortar (1D) projector:
        # used only to detect which *edges* contribute to each side.
        self._Pprim_to_mortar = self.intf.primary_to_mortar_avg().tocsc()

        # Per-side containers
        self._sides: Dict[object, _SideData1D] = {}
        self._side_order: List[object] = []

        # Build per-side 1D IBGs
        for P_side, g_side in self.intf.project_to_side_grids():
            side_enum = self._enum_of_side_grid(g_side)
            edges_side = self._edges_for_side(P_side)

            ibg_grid, parent_edge_map = self._build_ibg_for_edges(
                edges_side, f"{name}_{side_enum.name.lower()}"
            )

            self._sides[side_enum] = _SideData1D(
                mortar_to_side=P_side.tocsc(),
                mortar_side_grid=g_side,
                ibg_grid=ibg_grid,
                high_edges=edges_side,
                parent_edge_of_cell=parent_edge_map,
            )
            self._side_order.append(side_enum)

        # Optional: global ordering across sides (like 2D IBG)
        self._finalize_global_ibg_ordering()

    # ---------- public API (parallel to InternalBoundaryGrid) ----------

    def sides(self):
        for s in self._sides.keys():
            yield s

    def mortar_to_side(self, side) -> sps.spmatrix:
        return self._sides[side].mortar_to_side

    def mortar_side_grid(self, side) -> pp.Grid:
        return self._sides[side].mortar_side_grid

    def ibg_side_grid(self, side) -> pp.Grid:
        """Return the 1D IBG grid for THIS side."""
        return self._sides[side].ibg_grid

    def high_edges(self, side) -> np.ndarray:
        """High-edge indices contributing to THIS side."""
        return self._sides[side].high_edges

    def parent_edge_of_cell(self, side) -> np.ndarray:
        """Parent high-edge index for each IBG cell of THIS side."""
        return self._sides[side].parent_edge_of_cell

    @property
    def rotation_matrix(self) -> np.ndarray:
        return self.rot_matrix

    # ---------- global ordering (like 2D IBG) ----------

    def _finalize_global_ibg_ordering(self) -> None:
        """Build global IBG cell ordering and per-side offsets."""
        offset = 0
        self._offsets: Dict[object, Tuple[int, int]] = {}
        for side in self._side_order:
            n_side = self._sides[side].ibg_grid.num_cells
            self._offsets[side] = (offset, offset + n_side)
            offset += n_side
        self._n_total = offset

    def num_cells(self) -> int:
        """Total number of IBG cells across all sides."""
        return self._n_total

    def ibg_to_side(self, side) -> sps.csc_matrix:
        """Selector from global IBG ordering → this side’s IBG cells.

        Shape: (n_side_cells, n_ibg_total).
        Multiplying with an array of shape (n_ibg_total, ndofs) returns
        (n_side_cells, ndofs), in the local IBG ordering of THIS side.
        """
        start, end = self._offsets[side]
        n_side = end - start
        if n_side == 0:
            return sps.csc_matrix((0, self._n_total))
        rows = np.arange(n_side)
        cols = rows + start
        data = np.ones(n_side, dtype=float)
        return sps.coo_matrix(
            (data, (rows, cols)), shape=(n_side, self._n_total)
        ).tocsc()

    def side_to_ibg(self, side) -> sps.csc_matrix:
        """Scatter from this side’s IBG cells → global IBG ordering.

        Shape: (n_ibg_total, n_side_cells).
        """
        return self.ibg_to_side(side).T.tocsc()

    def project_to_side_ibg(self):
        """Generator like MortarGrid.project_to_side_grids(), but for IBG.

        Yields tuples: (proj, ibg_side_grid) where
          - proj : (n_side_cells, n_ibg_total) selector
          - ibg_side_grid : pp.Grid (TensorGrid) for THIS side
        """
        for side in self._side_order:
            yield self.ibg_to_side(side), self._sides[side].ibg_grid

    # ---------- internals ----------

    def _enum_of_side_grid(self, g: pp.Grid):
        for k, v in self.intf.side_grids.items():
            if v is g:
                return k
        raise KeyError("Side grid not found among MortarGrid.side_grids")

    def _edges_for_side(self, P_side: sps.spmatrix) -> np.ndarray:
        """Return indices of high edges that contribute to this side."""
        filt = (P_side @ self._Pprim_to_mortar).tocsc()
        mask = np.asarray(filt.sum(axis=0)).ravel() > 0
        return np.nonzero(mask)[0].astype(int)

    def _build_ibg_for_edges(
        self, edges: np.ndarray, name: str
    ) -> Tuple[pp.Grid, np.ndarray]:
        """Build a 1D TensorGrid for THIS side from its high-edge set.

        Strategy (no merging beyond endpoint dedup):
        1. Project high nodes to interface frame, define scalar coordinate s.
        2. For each selected high edge, build a segment [s_lo, s_hi].
        3. Collect all segment endpoints and deduplicate by 'tol'.
        4. Use sorted unique endpoints as 1D nodes → TensorGrid.
        5. Each IBG cell [s_i, s_{i+1}] gets a parent high-edge such that
           [s_i, s_{i+1}] ⊂ [s_lo, s_hi] within 'tol'.
        """
        if edges.size == 0:
            g = pp.TensorGrid(np.empty((1, 0)), name=name)
            g.compute_geometry()
            return g, np.zeros((0,), dtype=int)

        nodes3d = self.sd_high.nodes  # (3, N_nodes)
        nodes2d = (self.rot_matrix @ nodes3d)[self.dim_bool, :]  # (2, N_nodes)
        s_all = nodes2d[0, :]  # scalar coordinate along interface line

        fn = self.sd_high.face_nodes.tocsc()  # in 2D, "faces" = edges (1D)

        segs: List[Tuple[float, float]] = []
        parents: List[int] = []

        # 1) Build raw segments in s-coordinate
        for e in edges:
            i0, i1 = fn.indptr[e], fn.indptr[e + 1]
            e_nodes = fn.indices[i0:i1]

            # expect exactly 2 nodes for an edge
            if e_nodes.size != 2:
                continue

            s0, s1 = float(s_all[e_nodes[0]]), float(s_all[e_nodes[1]])

            # discard degenerate edges
            if abs(s1 - s0) < self.tol:
                continue

            lo, hi = (s0, s1) if s0 < s1 else (s1, s0)
            segs.append((lo, hi))
            parents.append(int(e))

        if not segs:
            g = pp.TensorGrid(np.empty((1, 0)), name=name)
            g.compute_geometry()
            return g, np.zeros((0,), dtype=int)

        # 2) Collect all endpoints and deduplicate with tolerance
        raw_points = sorted({p for lo, hi in segs for p in (lo, hi)})

        unique_points: List[float] = []
        if raw_points:
            current = raw_points[0]
            unique_points.append(current)
            for p in raw_points[1:]:
                if abs(p - current) > self.tol:
                    unique_points.append(p)
                    current = p

        # Need at least 2 points to form a cell
        if len(unique_points) < 2:
            g = pp.TensorGrid(np.empty((1, 0)), name=name)
            g.compute_geometry()
            return g, np.zeros((0,), dtype=int)

        pts_arr = np.array(unique_points).reshape(1, -1)  # (1, n_nodes)
        g1d = pp.TensorGrid(pts_arr, name=name)
        g1d.compute_geometry()

        # 3) Assign a parent edge to each IBG cell
        parent_edge_of_cell = np.empty(g1d.num_cells, dtype=int)
        for c in range(g1d.num_cells):
            a = unique_points[c]
            b = unique_points[c + 1]
            parent = -1
            # pick the first original segment that fully covers [a, b]
            for k, (lo, hi) in enumerate(segs):
                if lo <= a + self.tol and hi + self.tol >= b:
                    parent = parents[k]
                    break
            parent_edge_of_cell[c] = parent

        return g1d, parent_edge_of_cell
