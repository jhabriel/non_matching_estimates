import numpy as np

import mdnme
import porepy as pp
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
from shapely.strtree import STRtree
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix


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

        # Dummy holders for rotated grids
        self._src_rot = None
        self._tgt_rot = None

        self._build_intersection_polygons()
        self._triangulate_intersections()
        self._assemble_transfer_grid()
        self._build_connectivity_matrices()

    # ---- internal extraction ----
    def _get_rotated_grid(self, grid: pp.Grid):
        """Caching of rotated grids"""
        if grid is self.g_source:
            if self._src_rot is None:
                self._src_rot = mdnme.RotatedGrid(grid)
            return self._src_rot
        if grid is self.g_target:
            if self._tgt_rot is None:
                self._tgt_rot = mdnme.RotatedGrid(grid)
            return self._tgt_rot
        # fallback
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


# ---------- Utility triangulation helpers ----------
def is_ccw(coords):
    x = np.array([p[0] for p in coords])
    y = np.array([p[1] for p in coords])
    return np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)) > 0


def point_in_triangle(pt, tri):
    (x, y), (x1, y1), (x2, y2) = pt, tri[0], tri[1]
    (x3, y3) = tri[2]
    denom = ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    if denom == 0:
        return False
    a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denom
    b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denom
    c = 1 - a - b
    return (0 < a < 1) and (0 < b < 1) and (0 < c < 1)


def merge_close_vertices(verts, cells, tol=1e-8):
    """
    verts: list of (x,y) tuples or array shape (N,2)
    cells: list of [i0,i1,i2] (connectivity using indices into verts)
    Returns:
        new_verts_arr: array shape (2, N_new)
        new_cells: list of [i0,i1,i2] with reindexed vertices
    """
    arr = np.array(verts)  # shape (N,2)
    tree = cKDTree(arr)
    groups = tree.query_ball_tree(tree, r=tol)

    # Find representative for each original index (smallest in its cluster)
    rep = {}
    for i, neigh in enumerate(groups):
        rep[i] = min(neigh)

    # Build mapping from representatives to consecutive new indices
    uniq_reps = sorted(set(rep.values()))
    new_index = {r: idx for idx, r in enumerate(uniq_reps)}

    # Final old->new map
    old_to_new = {i: new_index[rep[i]] for i in range(len(arr))}

    # Build new vertex list (take reps)
    new_verts = arr[uniq_reps, :]  # shape (N_new,2)

    # Remap cells
    remapped = []
    for tri in cells:
        remapped.append([old_to_new[i] for i in tri])

    return new_verts.T, remapped  # return in (2, N_new) form and updated cells


def ear_clip_triangulate(coords, tol=1e-12):
    verts = coords.copy()
    if not is_ccw(verts):
        verts.reverse()
    tris = []
    max_iter = len(verts) ** 2
    it = 0
    while len(verts) > 3 and it < max_iter:
        it += 1
        n = len(verts)
        ear_found = False
        for i in range(n):
            prev = verts[(i - 1) % n]
            curr = verts[i]
            nxt = verts[(i + 1) % n]
            ax, ay = prev; bx, by = curr; cx, cy = nxt
            # convexity test
            if (bx - ax)*(cy - ay) - (by - ay)*(cx - ax) <= 0:
                continue
            tri = [prev, curr, nxt]
            # no other point inside
            if any(point_in_triangle(p, tri)
                   for j, p in enumerate(verts)
                   if j not in {(i - 1) % n, i, (i + 1) % n}):
                continue
            # clip ear
            tris.append(tri)
            del verts[i]
            ear_found = True
            break
        if not ear_found:
            break  # degeneracy; bail
    if len(verts) == 3:
        tris.append(verts)
    # filter tiny
    filtered = []
    for t in tris:
        area = 0.5 * abs(
            np.cross(np.subtract(t[1], t[0]), np.subtract(t[2], t[0]))
        )
        if area > tol:
            filtered.append(t)
    return filtered


def ensure_ccw(cells, coords_arr):
    new_cells = []
    for tri in cells:
        i0, i1, i2 = tri
        p0 = coords_arr[:, i0]
        p1 = coords_arr[:, i1]
        p2 = coords_arr[:, i2]
        cross = (p1[0] - p0[0])*(p2[1] - p0[1]) - (p1[1] - p0[1])*(p2[0] - p0[0])
        if cross < 0:
            new_cells.append([i0, i2, i1])
        else:
            new_cells.append([i0, i1, i2])
    return new_cells


def refine_grid(g: pp.TriangleGrid) -> tuple[pp.TriangleGrid, np.ndarray]:
    if not hasattr(g, "face_centers"):
        g.compute_geometry()
    nd = g.dim  # expected 2 for triangle grid

    # Dense face-node and cell-face maps
    fn = g.face_nodes.indices.reshape((nd, g.num_faces), order="F")
    cf = g.cell_faces.indices.reshape((nd + 1, g.num_cells), order="F")

    new_nodes = np.hstack((g.nodes, g.face_centers))
    offset = g.num_nodes

    # Red refinement combinations in 2D: each original triangle produces 4 children
    binom = ((1, 0), (2, 1), (0, 2))  # 3 of the subtriangles; 4 is face-centers

    # Holder: shape (nd+1, n_cells, nd+2) -> (3, n_cells, 4)
    new_tri = np.empty(shape=(nd + 1, g.num_cells, nd + 2), dtype=int)

    for ti, b in enumerate(binom):
        # Stack face-node indices for the two faces per cell
        loc_n = np.vstack((fn[:, cf[b[0]]], fn[:, cf[b[1]]]))  # shape (4, n_cells)

        # Sort so duplicates are adjacent
        loc_n.sort(axis=0)
        diffs = np.diff(loc_n, axis=0)  # shape (3, n_cells)

        # Extract duplicated vertex per cell (should be exactly one per column)
        dup_node = np.empty(g.num_cells, dtype=int)
        for cell in range(g.num_cells):
            # find first place where diff is zero
            row_hits = np.where(diffs[:, cell] == 0)[0]
            if len(row_hits) == 0:
                msg = (f"Could not find duplicated vertex for cell {cell} during"
                       f" refinement.")
                raise RuntimeError(msg)
            row = row_hits[0]
            dup_node[cell] = loc_n[row, cell]

        new_tri[0, :, ti] = dup_node
        new_tri[1, :, ti] = offset + cf[b[0]]
        new_tri[2, :, ti] = offset + cf[b[1]]

    # Fourth child: triangle of the three face centers
    new_tri[:, :, -1] = offset + cf  # shape (3, n_cells)

    # Flatten to (nd+1, (nd+2)*n_cells)
    new_tri = new_tri.reshape((nd + 1, (nd + 2) * g.num_cells))

    # Enforce consistent CCW orientation using existing helper
    # ensure_ccw expects list of [i0,i1,i2] so transpose appropriately
    cells_list = new_tri.T.tolist()  # list of [i0,i1,i2]
    # returns list of corrected triples
    corrected = ensure_ccw(cells_list, new_nodes[:2, :])
    corrected_arr = np.array(corrected).T  # back to shape (3, Nnew_cells)

    # Parent mapping: each original cell gives (nd+2) children
    parent = np.tile(np.arange(g.num_cells), g.dim + 2)

    # Build new grid, preserving history
    history = g.history.copy()
    history.append("Refinement")
    new_grid = pp.TriangleGrid(new_nodes, tri=corrected_arr, name=g.name)
    new_grid.compute_geometry()
    new_grid.history = history

    return new_grid, parent
