"""Point cloud -> 2D occupancy grid: local floor, height band, unknown cells, alignment, despeckle."""
import os

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .pcdio import read_pcd, read_positions

OCC, FREE, UNKNOWN = 0, 254, 205


def local_floor(P, positions=None, cell=0.2, near=0.3):
    """per-point floor height.

    With trajectory positions (ground robot, ground-level body frame): 10th percentile of the points within `near` of
    the nearest position's height, per `cell`, then a 3x3 minimum.
    Without: 10th percentile of each 0.4 m column, median-smoothed and 3x3 minimum (works when the floor is visible).
    """
    if positions is not None and len(positions):
        _, nk = cKDTree(positions[:, :2]).query(P[:, :2], workers=-1)
        ref = positions[nk, 2]
        use = np.abs(P[:, 2] - ref) < near
    else:
        cell = 0.4
        ref = np.full(len(P), np.percentile(P[:, 2], 5))
        use = np.ones(len(P), bool)
    ij = np.floor(P[:, :2] / cell).astype(np.int64)
    ij -= ij.min(0)
    shp = tuple(ij.max(0) + 1)
    grid = np.full(shp, np.inf)
    idx = np.nonzero(use)[0]
    idx = idx[np.lexsort((P[idx, 2], ij[idx, 1], ij[idx, 0]))]
    key = ij[idx, 0] * shp[1] + ij[idx, 1]
    uk, st, cnt = np.unique(key, return_index=True, return_counts=True)
    grid.flat[uk] = P[idx[st + (cnt * 0.1).astype(int)], 2]
    if positions is None or not len(positions):
        fin = np.isfinite(grid)
        med = ndimage.median_filter(np.where(fin, grid, np.nanmedian(grid[fin])), size=5)
        grid = np.where(fin, np.minimum(grid, med + 0.05), np.inf)
    grid = ndimage.minimum_filter(grid, size=3)
    f = grid[ij[:, 0], ij[:, 1]]
    return np.where(np.isfinite(f), f, ref)


def read_grid(d):
    """grid.yaml (+ image) in folder d -> (img row 0 = top, resolution, (x0, y0)) or None"""
    try:
        import yaml
        from PIL import Image
        with open(os.path.join(d, 'grid.yaml')) as f:
            y = yaml.safe_load(f)
        img = np.array(Image.open(os.path.join(d, y['image'])).convert('L'))
        return img, float(y['resolution']), (float(y['origin'][0]), float(y['origin'][1]))
    except (OSError, KeyError, ValueError, TypeError, ImportError):
        return None


class Source:
    """a map folder (map.pcd [+ poses.txt] [+ grid.yaml]) or a single .pcd file"""

    def __init__(self, path, pcd_name='map.pcd', poses=None):
        path = os.path.abspath(os.path.expanduser(path))
        self.dir = path if os.path.isdir(path) else os.path.dirname(path)
        self.pcd = os.path.join(path, pcd_name) if os.path.isdir(path) else path
        if not os.path.isfile(self.pcd):
            raise FileNotFoundError(f'{self.pcd} not found')
        self.P = read_pcd(self.pcd)
        if not len(self.P):
            raise ValueError(f'{self.pcd} has no points')
        pp = poses or os.path.join(self.dir, 'poses.txt')
        self.positions = read_positions(pp) if os.path.isfile(pp) else None
        self.floor_source = 'trajectory' if self.positions is not None else 'point cloud'
        self.floor = local_floor(self.P, self.positions)
        self.grid = read_grid(self.dir)

    def heights(self, relative=True):
        return self.P[:, 2] - self.floor if relative else self.P[:, 2]


def make_map(src, zmin=0.15, zmax=0.8, res=0.05, min_points=2, relative=True, align=True, unknown=True,
             despeckle=0):
    """-> (image uint8 row 0 = +y top, resolution, (x0, y0), stats)

    A cell is occupied when at least `min_points` points lie in [zmin, zmax) (above the local floor if `relative`).
    With `unknown`, cells that the folder's existing grid.pgm marks unknown stay unknown; otherwise every cell without
    such points is free (what a plain "slice and count" converter does).
    """
    h = src.heights(relative)
    sel = (h >= zmin) & (h < zmax)
    Q = src.P[sel]
    g = src.grid
    if align and g is not None and abs(g[1] - res) < 1e-9:
        H, W = g[0].shape
        x0, y0 = g[2]
    else:
        lo, hi = src.P[:, :2].min(0), src.P[:, :2].max(0)
        x0, y0 = np.floor(lo / res) * res - res
        W, H = int(np.ceil((hi[0] - x0) / res)) + 2, int(np.ceil((hi[1] - y0) / res)) + 2
    col = np.floor((Q[:, 0] - x0) / res).astype(np.int64)
    row = H - 1 - np.floor((Q[:, 1] - y0) / res).astype(np.int64)
    ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    cnt = np.zeros((H, W), np.int32)
    np.add.at(cnt, (row[ok], col[ok]), 1)
    occ = cnt >= max(1, int(min_points))
    removed = 0
    if despeckle > 0 and occ.any():
        lab, _ = ndimage.label(occ, np.ones((3, 3)))
        small = np.bincount(lab.ravel()) <= despeckle
        small[0] = False
        drop = small[lab]
        removed = int(drop.sum())
        occ &= ~drop
    out = np.full((H, W), FREE, np.uint8)
    if unknown and g is not None:
        gi, gr, (gx, gy) = g
        cy = y0 + (H - 1 - np.arange(H) + 0.5) * res
        cx = x0 + (np.arange(W) + 0.5) * res
        gc = np.floor((cx - gx) / gr).astype(np.int64)
        grow = gi.shape[0] - 1 - np.floor((cy - gy) / gr).astype(np.int64)
        okc, okr = (gc >= 0) & (gc < gi.shape[1]), (grow >= 0) & (grow < gi.shape[0])
        sub = gi[np.ix_(np.clip(grow, 0, gi.shape[0] - 1), np.clip(gc, 0, gi.shape[1] - 1))]
        known = np.zeros((H, W), bool)
        known[np.ix_(okr, okc)] = sub[np.ix_(okr, okc)] != UNKNOWN
        out[~known] = UNKNOWN
    out[occ] = OCC
    stats = dict(points=int(sel.sum()), occupied=int(occ.sum()), free=int((out == FREE).sum()),
                 unknown=int((out == UNKNOWN).sum()), removed=removed, size=(W, H),
                 height=f'above local floor ({src.floor_source})' if relative else 'absolute z')
    return out, res, (float(x0), float(y0)), stats


def save_map(path, img, res, origin):
    """write <path>.pgm + <path>.yaml (Nav2 map_server, trinary)"""
    from PIL import Image
    if not path.endswith('.pgm'):
        path += '.pgm'
    Image.fromarray(img).save(path)
    yp = path[:-4] + '.yaml'
    with open(yp, 'w') as f:
        f.write(f'image: {os.path.basename(path)}\nmode: trinary\nresolution: {res}\n'
                f'origin: [{origin[0]}, {origin[1]}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')
    return path, yp
