import numpy as np
from scipy.spatial import cKDTree

import mdnme
import porepy as pp
import matplotlib.pyplot as plt
import scipy.sparse as sps


from mdnme.utils.grid_utils import (
    ensure_ccw,
    ear_clip_triangulate,
    merge_close_vertices,
)

from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
from scipy.sparse import lil_matrix
from shapely.strtree import STRtree
from itertools import combinations
from porepy.grids.refinement import structured_refinement


class TransferGrid:
    """
    Transfer grid between source and target grids.

    Assumes source and target lie on the same geometric plane. All geometric
    intersection and connectivity computations are done in their shared rotated
    2D parameterization (via mdnme.RotatedGrid), which is cached internally.

    """
    def __init__(self,
                 g_source: pp.GridLike,
                 g_target: pp.GridLike,
                 rotation_matrix: np.ndarray = None,
                 tol: float = 1e-8,
                 name: str = "transfer"
                 ):

        self.tol = tol
        """Geometric tolerance. Default is 1e-8."""

        self.name = name
        """Name of the transfer grid. Default is ``transfer''."""

        self.g_source = g_source
        """Source grid."""

        self.g_target = g_target
        """Target grid."""

        self.rot_matrix = rotation_matrix
        """Rotation matrix used to rotate `g_source` and `g_target`. 
        
        If not given, the rotation matrix used will be the rotation matrix of
        `g_source`.        
        """

        # Holders for rotated grids
        self._rot_matrix = rotation_matrix
        self._src_rot = None
        self._tgt_rot = None

        self._build_intersection_polygons()
        self._triangulate_intersections()
        self._assemble_transfer_grid()
        self._build_connectivity_matrices()

    def _get_rotated_grid(self, grid: pp.Grid):
        if grid is self.g_source:
            if self._src_rot is None:
                self._src_rot = mdnme.RotatedGrid(
                    grid, self._rot_matrix
                ) if self._rot_matrix is not None else mdnme.RotatedGrid(grid)
                # If not provided, adopt the source’s matrix so target uses the same
                if self._rot_matrix is None:
                    self._rot_matrix = self._src_rot.rotation_matrix
            return self._src_rot
        if grid is self.g_target:
            if self._tgt_rot is None:
                if self._rot_matrix is None:
                    raise ValueError(
                        "TransferGrid needs a rotation_matrix or"
                        " a rotated source first."
                    )
                self._tgt_rot = mdnme.RotatedGrid(grid, self._rot_matrix)
            return self._tgt_rot
        return mdnme.RotatedGrid(grid)

    def _extract_triangles(self, grid: pp.GridLike):
        """Extract triangles of source and target grids.

        Note:
        -----
            Geometric computations are done using rotated grids. Note that this
            assumes that both source and target grids represent the same surface in 3D
            space. To avoid expensive computations, no check is done to assure that this
            is indeed the case.

        """
        grid_rot = self._get_rotated_grid(grid)  # rotate grid
        nodes = grid_rot.nodes  # retrieve nodes (from rotated grid)
        cn = grid.cell_nodes().tocsc()  # cell-nodes connectivity
        cn_arr = cn.indices.reshape((3, grid.num_cells), order="F")  # make it an array
        tris = []  # prepare list to retrieve triangles
        for i in range(grid.num_cells):
            idx = cn_arr[:, i]
            coords_xy = nodes[:2, idx].T
            tris.append((i, Polygon(coords_xy)))
        return tris

    def _build_intersection_polygons(self):
        tris_source = self._extract_triangles(self.g_source)  # source grid triangles
        tris_target = self._extract_triangles(self.g_target)  # target grid triangles

        intersections = []
        for _, poly_s in tris_source:
            for _, poly_t in tris_target:
                inter = poly_s.intersection(poly_t)
                if inter.area > self.tol:
                    intersections.append(inter)
        self._intersection_polys = intersections

    def _triangulate_intersections(self):
        all_triangles = []
        for poly in self._intersection_polys:
            raw = list(poly.exterior.coords)[:-1]
            coords = []
            for p in raw:
                if not coords or coords[-1] != p:
                    coords.append(p)
            if len(coords) == 3 and poly.area > self.tol:
                all_triangles.append(coords)
            else:
                tris = ear_clip_triangulate(coords, tol=self.tol)
                all_triangles.extend(tris)
        self._all_triangles = all_triangles

    def _assemble_transfer_grid(self):
        # Sanity check
        if not self._all_triangles:
            total_area = sum(p.area for p in self._intersection_polys)
            raise RuntimeError(
                f"No intersection triangles (total intersection area={total_area:.3e}). "
                "Likely the source and target are not coplanar or overlap is degenerate."
            )
        raw_verts = []
        for tri in self._all_triangles:
            for p in tri:
                raw_verts.append((float(p[0]), float(p[1])))
        # initial cells (with duplicates)
        pt_to_idx = {}
        verts = []
        for tri in self._all_triangles:
            for p in tri:
                if p not in pt_to_idx:
                    pt_to_idx[p] = len(verts)
                    verts.append((float(p[0]), float(p[1])))
        cells = [[pt_to_idx[p] for p in tri] for tri in self._all_triangles]

        # merge nearly duplicate vertices
        coords_arr, cells_merged = merge_close_vertices(verts, cells, tol=self.tol)
        # enforce orientation
        cells_ccw = ensure_ccw(cells_merged, coords_arr)
        cells_arr = np.array(cells_ccw).T  # (3, Ncells)

        self.transfer = pp.TriangleGrid(coords_arr, cells_arr, name=self.name)
        self.transfer.compute_geometry()


    # ---- connectivity queries ----
    def _build_connectivity_matrices(self):
        """
        Builds and caches the four connectivity matrices:
            source_to_transfer,
            transfer_to_source,
            transfer_to_target,
            target_to_transfer

        All are binary (0/1) based on centroid-in-polygon containment/touching.
        """
        # --- prepare polygons and spatial indices ---
        # Source polygons
        src_tris = self._extract_triangles(self.g_source)
        src_polys = [poly for _, poly in src_tris]
        prepared_src = [prep(p) for p in src_polys]
        tree_src = STRtree(src_polys)
        # Map geometry -> index (works if query returns geometry)
        src_poly_to_idx = {id(p): i for i, p in enumerate(src_polys)}

        # Transfer polygons
        tr_tris = self._extract_triangles(self.transfer)
        tr_polys = [poly for _, poly in tr_tris]
        prepared_tr = [prep(p) for p in tr_polys]
        tree_tr = STRtree(tr_polys)
        tr_poly_to_idx = {id(p): i for i, p in enumerate(tr_polys)}

        # Target polygons
        tgt_tris = self._extract_triangles(self.g_target)
        tgt_polys = [poly for _, poly in tgt_tris]
        prepared_tgt = [prep(p) for p in tgt_polys]
        tree_tgt = STRtree(tgt_polys)
        tgt_poly_to_idx = {id(p): i for i, p in enumerate(tgt_polys)}

        # --- allocate matrices ---
        n_src = self.g_source.num_cells
        n_tr = self.transfer.num_cells
        n_tgt = self.g_target.num_cells

        s2t = lil_matrix((n_src, n_tr), dtype=int)
        t2tgt = lil_matrix((n_tr, n_tgt), dtype=int)

        # --- source -> transfer ---
        tr_centroids = self.transfer.cell_centers  # shape (>=2, n_tr)
        for j in range(n_tr):
            pt = Point(tr_centroids[0, j], tr_centroids[1, j])
            for hit in tree_src.query(pt):
                # hit might be a geometry or an index depending on Shapely version
                if isinstance(hit, (int, np.integer)):
                    i = int(hit)
                else:
                    i = src_poly_to_idx.get(id(hit), None)
                    if i is None:
                        # fallback to linear search (should be rare)
                        try:
                            i = src_polys.index(hit)
                        except ValueError:
                            continue
                if prepared_src[i].contains(pt) or prepared_src[i].touches(pt):
                    s2t[i, j] = 1

        # --- transfer -> source is transpose ---
        t2s = s2t.transpose().tocsr()

        # --- transfer -> target ---
        for i in range(n_tr):
            pt = Point(tr_centroids[0, i], tr_centroids[1, i])
            for hit in tree_tgt.query(pt):
                if isinstance(hit, (int, np.integer)):
                    j = int(hit)
                else:
                    j = tgt_poly_to_idx.get(id(hit), None)
                    if j is None:
                        try:
                            j = tgt_polys.index(hit)
                        except ValueError:
                            continue
                if prepared_tgt[j].contains(pt) or prepared_tgt[j].touches(pt):
                    t2tgt[i, j] = 1

        # --- target -> transfer is transpose ---
        tgt2t = t2tgt.transpose().tocsr()

        # Cache
        self.source_to_transfer = s2t.tocsr()
        self.transfer_to_source = t2s
        self.transfer_to_target = t2tgt.tocsr()
        self.target_to_transfer = tgt2t

    @classmethod
    def from_nested(
            cls,
            g_source: pp.Grid,
            g_target: pp.Grid,
            coarse_fine: sps.csc_matrix | None = None,  # shape (n_fine, n_coarse)
            rotation_matrix: np.ndarray | None = None,  # kept for API symmetry; unused
            tol: float = 1e-8,
            name: str = "transfer",
    ) -> "TransferGrid":
        """
        Fast path for (assumed) nested refinement: use the *fine* grid as the transfer
        mesh, and assemble the 0/1 incidence matrices algebraically from a
        (fine × coarse) mapping.

        Equal-cell case:
          - Supported only if an explicit (n×n) mapping is provided.
          - Mapping may be identity or a permutation-like 0/1 matrix (one 1 per row).
          - In this case we treat g_source as "fine" and g_target as "coarse" by convention.
        """
        n_src, n_tgt = g_source.num_cells, g_target.num_cells

        def _is_valid_row_stochastic(M: sps.spmatrix) -> bool:
            # one 1 per row, 0/1 entries; columns can be >=1 for nested; for equal cells,
            # permutation would also have one 1 per column.
            row_sums = np.asarray(M.sum(axis=1)).ravel()
            return np.allclose(row_sums, 1.0)  # tolerate float format

        # ---- decide fine/coarse role and pick mapping M (fine × coarse) ----
        if n_src == n_tgt:
            # Equal-cell special: require an explicit mapping
            if coarse_fine is None:
                raise ValueError(
                    "from_nested: source and target have the same number of cells. "
                    "Please provide an explicit (n×n) coarse_fine mapping (e.g., identity "
                    "or permutation). Otherwise, build a geometric TransferGrid instead."
                )
            M = coarse_fine.tocsc()
            if M.shape != (n_src, n_tgt):
                raise ValueError(
                    f"from_nested: provided mapping has shape {M.shape}, expected {(n_src, n_tgt)}."
                )
            if not _is_valid_row_stochastic(M):
                raise ValueError(
                    "from_nested: mapping for equal-cell case must have exactly one 1 per row."
                )
            # Convention: treat source as fine, target as coarse
            g_fine, g_coarse = g_source, g_target
            src_is_coarse = False
        else:
            # Strict nested by cell counts
            if n_src < n_tgt:
                g_coarse, g_fine = g_source, g_target
                src_is_coarse = True
            else:
                g_coarse, g_fine = g_target, g_source
                src_is_coarse = False

            # pick/find mapping if absent
            if coarse_fine is None:
                # try typical storage on coarse grid
                d = getattr(g_coarse, "data", {})
                M0 = d.get("coarse_fine_cell_mapping", None)
                if isinstance(M0, sps.spmatrix) and M0.shape == (
                g_fine.num_cells, g_coarse.num_cells):
                    M = M0.tocsc()
                else:
                    raise ValueError(
                        "from_nested: coarse_fine (fine×coarse) not provided and not found in "
                        "g_coarse.data['coarse_fine_cell_mapping']."
                    )
            else:
                M = coarse_fine.tocsc()
                if M.shape != (g_fine.num_cells, g_coarse.num_cells):
                    raise ValueError(
                        f"from_nested: provided mapping has shape {M.shape}, expected "
                        f"{(g_fine.num_cells, g_coarse.num_cells)}."
                    )

        # ---- construct a lightweight instance ----
        obj = cls.__new__(cls)
        obj.tol = tol
        obj.name = name
        obj.g_source = g_source
        obj.g_target = g_target

        # use the *actual* fine mesh as transfer
        obj.transfer = g_fine.copy()
        R_eff = mdnme.RotatedGrid(g_source).rotation_matrix
        obj._rot_matrix = R_eff
        obj._src_rot = None
        obj._tgt_rot = None

        # ---- assemble the four incidence matrices ----
        n_fine = g_fine.num_cells
        I_fine = sps.identity(n_fine, format="csc")

        if src_is_coarse:
            # source == coarse, target == fine
            s2t = M.T  # (n_coarse × n_fine)
            t2s = M  # (n_fine  × n_coarse)
            t2tgt = I_fine  # (n_fine  × n_fine)
            tgt2t = I_fine
        else:
            # source == fine, target == coarse
            s2t = I_fine
            t2s = I_fine
            t2tgt = M  # (n_fine × n_coarse)
            tgt2t = M.T

        obj.source_to_transfer = s2t.tocsr()
        obj.transfer_to_source = t2s.tocsr()
        obj.transfer_to_target = t2tgt.tocsr()
        obj.target_to_transfer = tgt2t.tocsr()

        return obj

    def summary(self):
        return {
            "n_source_cells": self.g_source.num_cells,
            "n_target_cells": self.g_target.num_cells,
            "n_transfer_cells": self.transfer.num_cells,
            "n_transfer_nodes": self.transfer.num_nodes,
        }

    def plot(self, ax=None, base_cmap="rainbow", alpha=1.0):
        """
        Plot the transfer mesh with a proper 4-coloring (no two neighbors share a color).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If None, a new figure+axes is created.
        base_cmap : str or Colormap, optional
            A Matplotlib colormap to draw from (should have >=4 distinct colors).
        alpha : float, optional
            Face alpha for the polygons.
        Returns
        -------
        fig, ax : tuple
            The figure and axes containing the plot.
        """
        # 1) prepare axes
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        # 2) get the 2D nodes and cells
        nodes2d = self.transfer.nodes[:2, :]
        cn = self.transfer.cell_nodes().tocsc()
        cells = cn.indices.reshape((3, self.transfer.num_cells), order="F").T  # (n_tri, 3)

        # 3) build edge→triangles lookup
        edge_to_tris: dict[tuple[int,int], list[int]] = {}
        for t_idx, tri in enumerate(cells):
            for edge in combinations(tri, 2):
                e = tuple(sorted(edge))
                edge_to_tris.setdefault(e, []).append(t_idx)

        # 4) build adjacency list
        n_tri = len(cells)
        neighbors = [set() for _ in range(n_tri)]
        for tris in edge_to_tris.values():
            if len(tris) == 2:
                i, j = tris
                neighbors[i].add(j)
                neighbors[j].add(i)

        # 5) greedy graph-coloring
        colors = [-1] * n_tri
        for t in range(n_tri):
            used = {colors[nbr] for nbr in neighbors[t] if colors[nbr] >= 0}
            # assign smallest non-negative integer not in used
            c = 0
            while c in used:
                c += 1
            colors[t] = c
        n_colors = max(colors) + 1

        # 6) sample RGBA’s from colormap
        cmap = plt.get_cmap(base_cmap)
        # for categorical colors, take indices 0, 1/(n_colors-1),...,1
        color_vals = cmap(np.linspace(0, 1, n_colors))

        # 7) build the polygons
        verts = [nodes2d[:, tri].T for tri in cells]

        # 8) build collection with facecolors by triangle-color
        facecolors = [color_vals[c] for c in colors]
        coll = PolyCollection(
            verts,
            facecolors=facecolors,
            edgecolors="none",
            alpha=alpha,
        )
        ax.add_collection(coll)

        # 9) finalize
        ax.autoscale()
        ax.set_aspect("equal", "box")
        ax.set_xticks([])
        ax.set_yticks([])

        # 10) save figure
        fig = ax.get_figure()
        fig.savefig(f'{self.name}.pdf')


class TransferLine:
    """Transfer 'grid' for 1D->1D mappings (segments on a common line)."""

    def __init__(
        self,
        g_source: pp.Grid,
        g_target: pp.Grid,
        tol: float = 1e-10,
        rotation_matrix: np.ndarray | None = None,
        name: str = "transfer1d",
    ):
        if g_source.dim != 1 or g_target.dim != 1:
            raise ValueError("TransferLine expects 1D source and 1D target grids.")
        self.tol = float(tol)
        self.name = name
        self.g_source = g_source
        self.g_target = g_target
        self.rot_matrix = rotation_matrix  # anchor frame if provided

        self._build_transfer_segments()
        self._build_connectivity_matrices()

    # same as you had
    def _x_in_common_frame(self, g: pp.Grid) -> np.ndarray:
        import mdnme
        if self.rot_matrix is None:
            rot = mdnme.RotatedGrid(g)  # let the first call set the frame
            self.rot_matrix = rot.rotation_matrix
            return rot.nodes[0, :]
        else:
            rot = mdnme.RotatedGrid(g, self.rot_matrix)
            return rot.nodes[0, :]

    def _breaks(self, g: pp.Grid) -> np.ndarray:
        x = self._x_in_common_frame(g)
        # uniq & sort; also snap tiny negatives to 0 with tol for stability
        x = np.unique(np.asarray(x, dtype=float))
        return x


    def _build_transfer_segments(self):
        xs = self._breaks(self.g_source)  # in common frame
        xt = self._breaks(self.g_target)

        i, j = 0, 0
        segs: list[tuple[float, float]] = []

        while i < len(xs) - 1 and j < len(xt) - 1:
            a1, b1 = xs[i], xs[i + 1]
            a2, b2 = xt[j], xt[j + 1]

            a = max(a1, a2)
            b = min(b1, b2)

            # keep only positive-length overlaps
            if b - a > self.tol:
                segs.append((float(a), float(b)))

            # advance pointer(s) with tolerance
            if b1 <= b2 + self.tol:
                i += 1
            if b2 <= b1 + self.tol:
                j += 1

        # If we found nothing, it’s a true miss or pure point-touch
        if not segs:
            raise RuntimeError("No 1D intersections between source and target.")

        # --- tolerant dedupe of endpoints after segment filtering ---
        pts = np.fromiter((p for ab in segs for p in ab), dtype=float)
        pts.sort()

        # merge neighbors closer than tol
        merged = [pts[0]]
        for p in pts[1:]:
            if abs(p - merged[-1]) > self.tol:
                merged.append(p)

        # Require at least one *positive* segment
        if len(merged) < 2:
            raise RuntimeError("Only point-touch between source and target.")
        # Also guard against accidental equal neighbors
        diffs = np.diff(merged)
        if not np.any(diffs > self.tol):
            raise RuntimeError("Only point-touch between source and target.")

        self.transfer_nodes = np.array(merged, dtype=float).reshape(1, -1)
        self.transfer = pp.TensorGrid(self.transfer_nodes)
        self.transfer.compute_geometry()

    def _locate_owner_cells(self, midpoints: np.ndarray, breaks: np.ndarray) -> np.ndarray:
        ids = np.searchsorted(breaks, midpoints, side="right") - 1
        return np.clip(ids, 0, len(breaks) - 2)

    def _build_connectivity_matrices(self):
        # handle the 0-cell short-circuit cleanly
        if self.transfer.num_cells == 0:
            n_src = self.g_source.num_cells
            n_tgt = self.g_target.num_cells
            zst = sps.csr_matrix((n_src, 0), dtype=int)
            ztt = sps.csr_matrix((0, n_tgt), dtype=int)
            self.source_to_transfer = zst
            self.transfer_to_source = zst.T
            self.transfer_to_target = ztt
            self.target_to_transfer = ztt.T
            return

        xs = self._breaks(self.g_source)
        xt = self._breaks(self.g_target)
        xtf = self._breaks(self.transfer)

        n_src = self.g_source.num_cells
        n_tr  = self.transfer.num_cells
        n_tgt = self.g_target.num_cells

        s2t = sps.lil_matrix((n_src, n_tr), dtype=int)
        t2tg = sps.lil_matrix((n_tr, n_tgt), dtype=int)

        mid = 0.5 * (xtf[:-1] + xtf[1:])
        src_owner = self._locate_owner_cells(mid, xs)
        tgt_owner = self._locate_owner_cells(mid, xt)

        for j in range(n_tr):
            s2t[src_owner[j], j] = 1
            t2tg[j, tgt_owner[j]] = 1

        self.source_to_transfer = s2t.tocsr()
        self.transfer_to_source = self.source_to_transfer.T.tocsr()
        self.transfer_to_target = t2tg.tocsr()
        self.target_to_transfer = self.transfer_to_target.T.tocsr()


    def summary(self):
        return {
            "n_source_cells": self.g_source.num_cells,
            "n_target_cells": self.g_target.num_cells,
            "n_transfer_cells": self.transfer.num_cells,
            "n_transfer_nodes": self.transfer.num_nodes,
        }


# ---- Utility functions ---
def build_transfer_grid_nested(
    gA: pp.Grid,
    gB: pp.Grid,
    mapping: sps.csc_matrix | None = None
) -> TransferGrid:
    """Return a TransferGrid using the fast nested path."""
    return TransferGrid.from_nested(gA, gB, coarse_fine=mapping, name="transfer_fast")


def coarse_fine_or_build(
    gA: pp.Grid,
    gB: pp.Grid,
    *,
    tol: float = 1e-9
) -> sps.csc_matrix:
    """
    Return coarse_fine mapping with shape (n_fine x n_coarse).

    - If g_coarse.data['coarse_fine_cell_mapping'] exists and matches shape, use it.
    - If n_fine == n_coarse, return identity (no call to structured_refinement).
    - Otherwise build via structured_refinement(g_coarse, g_fine).
    """
    # decide who is coarse/fine by num_cells
    if gA.num_cells <= gB.num_cells:
        g_coarse, g_fine = gA, gB
    else:
        g_coarse, g_fine = gB, gA

    n_coarse, n_fine = g_coarse.num_cells, g_fine.num_cells

    # 1) use cached if present and correct shape
    d = getattr(g_coarse, "data", None)
    if isinstance(d, dict) and "coarse_fine_cell_mapping" in d:
        M0 = d["coarse_fine_cell_mapping"]
        if isinstance(M0, sps.spmatrix) and M0.shape == (n_fine, n_coarse):
            return M0.tocsc()

    # 2) equal-size case: identity (fine x coarse) == (n x n)
    if n_fine == n_coarse:
        return sps.identity(n_fine, format="csc")

    # 3) strictly nested: build
    M = structured_refinement(g_coarse, g_fine, point_in_poly_tol=tol).tocsc()
    return M


def permute_transfer_columns(
        A: sps.spmatrix,
        perm: np.ndarray
    ) -> sps.spmatrix:
    """Return A with its columns permuted so that new[:, j] = A[:, perm[j]]."""
    P = sps.coo_matrix((np.ones(len(perm)), (perm, np.arange(len(perm)))),
                       shape=(len(perm), len(perm))).tocsr()
    return A @ P


def transfer_permutation_by_centroids(
        tg_ref,
        tg_to_perm,
        *,
        rtol=0,
        atol=1e-12
    ) -> np.ndarray:
    """
    Compute permutation that reorders tg_to_perm.transfer cells to match tg_ref.transfer
    by nearest neighbor matching of 2D centroids.
    """
    C_ref  = tg_ref.transfer.cell_centers[:2, :].T
    C_perm = tg_to_perm.transfer.cell_centers[:2, :].T
    tree = cKDTree(C_perm)
    d, idx = tree.query(C_ref, k=1)
    if not np.all(d <= atol + rtol*np.abs(C_ref).max()):
        raise AssertionError(f"Transfer centroids mismatch; max diff {d.max():.3e}")
    return idx


# def mapping_fine_x_coarse(
#         coarse: pp.Grid,
#         fine: pp.Grid,
#         tol=1e-8
# ) -> sps.csc_matrix:
#     # reuse cached mapping if present
#     M = getattr(coarse, "data", {}).get("coarse_fine_cell_mapping", None)
#     if isinstance(M, sps.spmatrix) and M.shape == (fine.num_cells, coarse.num_cells):
#         return M.tocsc()
#     M = structured_refinement(coarse, fine, point_in_poly_tol=tol)
#     if not hasattr(coarse, "data") or not isinstance(coarse.data, dict):
#         coarse.data = {}
#     coarse.data["coarse_fine_cell_mapping"] = M
#     return M
