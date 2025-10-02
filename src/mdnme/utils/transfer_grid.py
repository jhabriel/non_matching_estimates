import numpy as np

import mdnme
import porepy as pp
import matplotlib.pyplot as plt
import scipy.sparse as sps


from mdnme.utils.grid_utils import (
    ensure_ccw,
    ear_clip_triangulate,
    is_ccw,
    merge_close_vertices,
)

from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
from scipy.sparse import lil_matrix
from shapely.strtree import STRtree
from itertools import combinations, chain


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

        # Dummy holders for rotated grids
        self._src_rot = None
        self._tgt_rot = None
        self._rot_matrix = None

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


# TODO: DEPRECATE FUNCTIONALITY WHEN ERROR ESTIMATORS ARE IN PLACE
# NOW, EVERYTHING IS MEDIATE IT VIA THE TRANSFER GRID
def build_high_internal_surface_grid(
    sd_high: pp.Grid,
    sd_low: pp.Grid,
    intf: pp.MortarGrid,
    tol: float = 1e-8,
    name: str = "high_internal_surface",
) -> tuple[pp.Grid, np.ndarray, np.ndarray]:
    """
    Construct a 2D TriangleGrid that is the union of the 3D high-side faces
    participating in the interface. The mesh is built in the 2D parameterization
    defined by the *lower* grid's rotation (so it is co-planar with the mortar).

    Returns
    -------
    g2d : pp.TriangleGrid
        The 2D surface grid (triangulated).
    frac_faces : np.ndarray (nf,)
        Indices of high-side faces used (same order as PorePy mappings).
    parent_face_of_cell : np.ndarray (g2d.num_cells,)
        For each triangle cell in g2d, the index of the originating high-side face.
    """
    # --- 1) Get the set of high-side faces that belong to the interface
    # primary_to_mortar_avg maps high faces -> mortar cells
    frac_faces = sps.find(intf.primary_to_mortar_avg())[1]

    # --- 2) Build a consistent 2D parameterization using the *low* grid
    low_rot = mdnme.RotatedGrid(sd_low)
    R = low_rot.rotation_matrix
    dim_bool = low_rot.dim_bool  # pick the two active in-plane axes

    # Rotate high-side nodes and discard inactive axis -> 2 x N
    nodes2d = (R @ sd_high.nodes)[dim_bool, :]

    # Face->nodes (ordered list per face)
    fn = sd_high.face_nodes.tocsc()

    # --- 3) Triangulate each face polygon in 2D and collect triangles
    all_tris_xy = []          # list of [(x,y), (x,y), (x,y)]
    parent_face_of_cell = []  # parallel list: face index per triangle

    for f in frac_faces:
        start, end = fn.indptr[f], fn.indptr[f + 1]
        face_nodes = fn.indices[start:end]
        pts = nodes2d[:, face_nodes].T  # (k, 2)

        if pts.shape[0] < 3:
            # degenerate / tiny face -> skip
            continue

        # Order polygon roughly around its centroid to get a simple loop
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        order = np.argsort(ang)
        poly = [tuple(pts[i]) for i in order]

        # Remove consecutive duplicates (robustness on coincident nodes)
        clean = []
        for p in poly:
            if (not clean) or (abs(p[0]-clean[-1][0]) > tol or abs(p[1]-clean[-1][1]) > tol):
                clean.append(p)
        if len(clean) < 3:
            continue

        # CCW orientation for ear clipping
        if not is_ccw(clean):
            clean.reverse()

        if len(clean) == 3:
            all_tris_xy.append(clean)
            parent_face_of_cell.append(f)
        else:
            tris = ear_clip_triangulate(clean, tol=tol)
            for tri in tris:
                all_tris_xy.append(tri)
                parent_face_of_cell.append(f)

    if not all_tris_xy:
        raise RuntimeError(
            "No triangles could be built from the high-side internal boundary."
        )

    # --- 4) Deduplicate vertices, enforce CCW, assemble TriangleGrid
    raw_verts = []
    raw_cells = []
    pt_to_idx = {}

    for tri in all_tris_xy:
        cell = []
        for p in tri:
            if p not in pt_to_idx:
                pt_to_idx[p] = len(raw_verts)
                raw_verts.append((float(p[0]), float(p[1])))
            cell.append(pt_to_idx[p])
        raw_cells.append(cell)

    coords_arr, cells_merged = merge_close_vertices(raw_verts, raw_cells, tol=tol)
    cells_ccw = ensure_ccw(cells_merged, coords_arr)
    tri_arr = np.array(cells_ccw).T  # (3, n_cells)

    g2d = pp.TriangleGrid(coords_arr, tri_arr, name=name)
    g2d.compute_geometry()

    parent_face_of_cell = np.asarray(parent_face_of_cell, dtype=int)
    if parent_face_of_cell.size != g2d.num_cells:
        # This should not happen (vertex merging does not change cell count),
        # but keep a guard.
        raise RuntimeError("Parent-face map size mismatch after assembling 2D grid.")

    return g2d, frac_faces, parent_face_of_cell


class TransferLine:
    """Transfer 'grid' for 1D->1D mappings (segments on a common line)."""

    def __init__(self,
                 g_source: pp.Grid,
                 g_target: pp.Grid,
                 tol: float = 1e-10,
                 name: str = "transfer1d"
        ):
        if g_source.dim != 1 or g_target.dim != 1:
            raise ValueError("TransferLine expects 1D source and 1D target grids.")
        self.tol = tol
        self.name = name
        self.g_source = g_source
        self.g_target = g_target

        self._build_transfer_segments()
        self._build_connectivity_matrices()

    def _breaks(self, g: pp.Grid) -> np.ndarray:
        # pp.TensorGrid stores nodes sorted; each cell is [x_i, x_{i+1}]
        x = g.nodes[0, :]
        x = np.unique(x)  # robust
        return x

    def _build_transfer_segments(self):
        xs = self._breaks(self.g_source)
        xt = self._breaks(self.g_target)
        # union of breakpoints
        xu = np.unique(np.concatenate([xs, xt]))
        # form segments; keep only those that overlap *both* a source and a target cell
        segs = []
        for a, b in zip(xu[:-1], xu[1:]):
            mid = 0.5 * (a + b)
            # check containment
            # source: any cell interval [xs[i], xs[i+1]] containing mid?
            ok_s = np.any((xs[:-1] - self.tol <= mid) & (mid <= xs[1:] + self.tol))
            ok_t = np.any((xt[:-1] - self.tol <= mid) & (mid <= xt[1:] + self.tol))
            if ok_s and ok_t and b - a > self.tol:
                segs.append((float(a), float(b)))

        if not segs:
            raise RuntimeError("No 1D intersections between source and target.")

        self.transfer_nodes = np.array(sorted({p for ab in segs for p in ab})).reshape(1, -1)
        self.transfer = pp.TensorGrid(self.transfer_nodes)
        self.transfer.compute_geometry()

    def _locate_owner_cells(self, midpoints: np.ndarray, breaks: np.ndarray):
        # return the index of the cell interval that contains midpoint
        # cells are 0..len(breaks)-2
        ids = np.searchsorted(breaks, midpoints, side="right") - 1
        ids = np.clip(ids, 0, len(breaks) - 2)
        return ids

    def _build_connectivity_matrices(self):
        xs = self._breaks(self.g_source)
        xt = self._breaks(self.g_target)
        xtf = self._breaks(self.transfer)

        n_src = self.g_source.num_cells
        n_tr  = self.transfer.num_cells
        n_tgt = self.g_target.num_cells

        s2t = sps.lil_matrix((n_src, n_tr), dtype=int)
        t2tg = sps.lil_matrix((n_tr, n_tgt), dtype=int)

        # midpoints of transfer cells
        mid = 0.5 * (xtf[:-1] + xtf[1:])
        # owner cells
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

