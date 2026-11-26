"""Edit a Nav2 map by hand in GIMP (or any layer-based editor).

export: the point cloud as height-band layers (PNG, same pixels as grid.pgm) next to the current grid
import: the edited image -> grid.pgm (dark = occupied, very light = free, rest / transparent = unknown)
"""
import os
import shutil

import numpy as np
from PIL import Image

from .core import FREE, OCC, UNKNOWN, Source, read_grid

LAYERS = [
    ('10_walls_pillars', (230, 30, 30), 'points from near the floor (0.1-0.6 m) up to 1.5 m+: walls, pillars, tall shelves'),
    ('20_robot_height', (30, 90, 230), 'points 0.15-0.8 m above the floor: desks, chairs, boxes'),
    ('30_low', (20, 170, 60), 'only 0.06-0.15 m: curbs, thresholds, low platforms'),
    ('40_overhead', (245, 150, 20), 'only above 0.8 m: things the robot passes under'),
]


def _turbo(x):
    # compact Turbo approximation (Google, Apache-2.0 polynomial)
    x = np.clip(x, 0, 1)[..., None]
    c = np.array([[0.13572138, 0.09140261, 0.10667330], [4.61539260, 2.19418839, 12.64194608],
                  [-42.66032258, 4.84296658, -60.58204836], [132.13108234, -14.18503333, 110.36276771],
                  [-152.94239396, 4.27729857, -89.90310912], [59.28637943, 2.82956604, 27.34824973]])
    v = sum(c[i] * x ** i for i in range(6))
    return (np.clip(v, 0, 1) * 255).astype(np.uint8)


def export(map_dir, out=None, min_points=2):
    src = Source(map_dir)
    g = read_grid(src.dir)
    if g is None:
        raise SystemExit(f'{src.dir}/grid.yaml not found (make one first, e.g. `pcd2nav2map {map_dir} --no-gui '
                         f'--out {src.dir}/grid.pgm`)')
    grid, res, (x0, y0) = g
    H, W = grid.shape
    out = out or os.path.join(src.dir, 'grid_edit')
    os.makedirs(out, exist_ok=True)
    rel = src.heights(True)
    col = np.floor((src.P[:, 0] - x0) / res).astype(np.int64)
    row = H - 1 - np.floor((src.P[:, 1] - y0) / res).astype(np.int64)
    ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)

    def band(lo, hi):
        s = ok & (rel >= lo) & (rel < hi)
        c = np.zeros((H, W), np.int32)
        np.add.at(c, (row[s], col[s]), 1)
        return c >= min_points

    low, mid, base, top, over = band(0.06, 0.15), band(0.15, 0.8), band(0.1, 0.6), band(1.5, 2.5), band(0.8, 2.5)
    tall = base & top
    masks = [tall, mid & ~tall, low & ~mid & ~tall, over & ~mid & ~low & ~tall]
    Image.fromarray(grid).convert('RGBA').save(os.path.join(out, '00_grid.png'))
    for (name, color, _), m in zip(LAYERS, masks):
        a = np.zeros((H, W, 4), np.uint8)
        a[m, :3], a[m, 3] = color, 255
        Image.fromarray(a, 'RGBA').save(os.path.join(out, name + '.png'))
    s = ok & (rel > 0.06) & (rel < 2.0)
    hmax = np.full((H, W), -1.0)
    np.maximum.at(hmax, (row[s], col[s]), rel[s])
    el = np.zeros((H, W, 4), np.uint8)
    m = hmax > 0
    el[m, :3], el[m, 3] = _turbo(hmax[m] / 2.0), 255
    Image.fromarray(el, 'RGBA').save(os.path.join(out, '50_height.png'))
    print(f'{out}/: 6 layers, {W} x {H} px, 1 px = {res:g} m')
    for (name, _, desc), mm in zip(LAYERS, masks):
        print(f'  {name}: {int(mm.sum())} cells ({desc})')
    print(f'GIMP: File > Open as Layers... (select all PNGs), edit 00_grid, hide the colour layers, '
          f'File > Export As {out}/grid_static.png, then `pcd2nav2map-layers import {map_dir} {out}/grid_static.png`')
    return out


def import_(map_dir, image, dark=100, light=230):
    d = os.path.abspath(os.path.expanduser(map_dir if os.path.isdir(map_dir) else os.path.dirname(map_dir)))
    g = read_grid(d)
    if g is None:
        raise SystemExit(f'{d}/grid.yaml not found')
    grid = g[0]
    im = Image.open(image)
    if im.size != (grid.shape[1], grid.shape[0]):
        raise SystemExit(f'size {im.size} != grid {grid.shape[1]}x{grid.shape[0]} - do not resize the canvas')
    a = np.array(im.convert('LA'))
    lum, alpha = a[..., 0], a[..., 1]
    out = np.full(grid.shape, UNKNOWN, np.uint8)
    out[(lum < dark) & (alpha > 0)] = OCC
    out[(lum > light) & (alpha > 0)] = FREE
    import yaml
    with open(os.path.join(d, 'grid.yaml')) as f:
        img_name = yaml.safe_load(f)['image']
    target = os.path.join(d, img_name)
    backup = target[:-4] + '_auto.pgm' if target.endswith('.pgm') else target + '.auto'
    if not os.path.exists(backup):
        shutil.copy2(target, backup)
    Image.fromarray(out).save(target)
    auto = np.array(Image.open(backup).convert('L'))
    print(f'{target} <- {image} (previous map kept as {os.path.basename(backup)})')
    print(f'  occupied {int((auto == OCC).sum())} -> {int((out == OCC).sum())}, free {int((auto == FREE).sum())} -> '
          f'{int((out == FREE).sum())}, unknown {int((auto == UNKNOWN).sum())} -> {int((out == UNKNOWN).sum())}')
    return target
