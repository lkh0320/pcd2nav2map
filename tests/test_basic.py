import os
import struct
import tempfile

import numpy as np
from PIL import Image

from pcd2nav2map.core import FREE, OCC, UNKNOWN, Source, make_map, save_map
from pcd2nav2map.pcdio import read_pcd, write_pcd


def room(tilt_deg=2.0, seed=0):
    """8 x 6 m room, floor tilted along x, walls, a 0.4 m box, a 0.1 m step, a shelf at 1.2-1.3 m"""
    rng = np.random.default_rng(seed)
    t = np.tan(np.radians(tilt_deg))

    def floor_z(x):
        return x * t

    pts = []
    xy = rng.uniform([0, 0], [8, 6], (60000, 2))
    pts.append(np.c_[xy, floor_z(xy[:, 0]) + rng.normal(0, 0.01, len(xy))])
    for x0, x1, y0, y1 in ((0, 8, 0, 0.05), (0, 8, 5.95, 6), (0, 0.05, 0, 6), (7.95, 8, 0, 6)):
        w = np.c_[rng.uniform(x0, x1, 20000), rng.uniform(y0, y1, 20000), rng.uniform(0, 2.5, 20000)]
        w[:, 2] += floor_z(w[:, 0])
        pts.append(w)
    b = np.c_[rng.uniform(3, 3.5, 4000), rng.uniform(3, 3.5, 4000), rng.uniform(0, 0.4, 4000)]
    b[:, 2] += floor_z(b[:, 0])
    s = np.c_[rng.uniform(5, 6, 3000), rng.uniform(1, 1.5, 3000), rng.uniform(0.0, 0.1, 3000)]
    s[:, 2] += floor_z(s[:, 0])
    sh = np.c_[rng.uniform(1, 2, 3000), rng.uniform(4, 4.5, 3000), rng.uniform(1.2, 1.3, 3000)]
    sh[:, 2] += floor_z(sh[:, 0])
    pts += [b, s, sh]
    poses = np.c_[np.linspace(1, 7, 30), np.full(30, 2.0), floor_z(np.linspace(1, 7, 30))]
    return np.vstack(pts), poses


def cell(img, res, origin, x, y):
    return img[img.shape[0] - 1 - int((y - origin[1]) / res), int((x - origin[0]) / res)]


def write_dir(d, P, poses=None):
    write_pcd(os.path.join(d, 'map.pcd'), P)
    if poses is not None:
        with open(os.path.join(d, 'poses.txt'), 'w') as f:
            for i, p in enumerate(poses):
                f.write(f'{i}.pcd {p[0]} {p[1]} {p[2]} 1 0 0 0\n')


def test_floor_relative_handles_tilt():
    P, poses = room(tilt_deg=2.0)
    with tempfile.TemporaryDirectory() as d:
        write_dir(d, P, poses)
        src = Source(d)
        img, res, o, _ = make_map(src, 0.15, 0.8, 0.05, 2)
        assert cell(img, res, o, 3.25, 3.25) == OCC          # box
        assert cell(img, res, o, 7.0, 2.5) == FREE           # tilted floor (~0.24 m high) is still floor
        assert cell(img, res, o, 1.5, 4.25) == FREE          # shelf at 1.2 m is above the band
        assert cell(img, res, o, 5.5, 1.25) == FREE          # 0.1 m step below the band
        assert cell(img, res, o, 0.02, 3.0) == OCC           # wall
        img2, res, o, _ = make_map(src, 0.06, 0.8, 0.05, 2)
        assert cell(img2, res, o, 5.5, 1.25) == OCC          # step included with a lower band
        ab, res, o, _ = make_map(src, 0.15, 0.8, 0.05, 2, relative=False)
        assert cell(ab, res, o, 7.0, 2.5) == OCC             # absolute z: the tilted floor turns into an obstacle


def test_floor_without_trajectory():
    P, _ = room(tilt_deg=1.0)
    with tempfile.TemporaryDirectory() as d:
        write_dir(d, P)
        src = Source(d)
        assert src.floor_source == 'point cloud'
        img, res, o, _ = make_map(src, 0.15, 0.8, 0.05, 2)
        assert cell(img, res, o, 3.25, 3.25) == OCC
        assert cell(img, res, o, 6.5, 2.5) == FREE


def test_unknown_and_alignment_from_existing_grid():
    P, poses = room(tilt_deg=0.0)
    with tempfile.TemporaryDirectory() as d:
        write_dir(d, P, poses)
        g = np.full((140, 180), UNKNOWN, np.uint8)
        g[20:120, 20:160] = FREE
        save_map(os.path.join(d, 'grid.pgm'), g, 0.05, (-0.5, -0.5))
        img, res, o, _ = make_map(Source(d), 0.15, 0.8, 0.05, 2)
        assert img.shape == g.shape and o == (-0.5, -0.5)
        assert img[5, 5] == UNKNOWN and img[70, 90] in (FREE, OCC)


def test_layers_roundtrip():
    from pcd2nav2map import layers
    P, poses = room(tilt_deg=0.0)
    with tempfile.TemporaryDirectory() as d:
        write_dir(d, P, poses)
        img, res, o, _ = make_map(Source(d), 0.15, 0.8, 0.05, 2, align=False, unknown=False)
        save_map(os.path.join(d, 'grid.pgm'), img, res, o)
        out = layers.export(d)
        assert len([f for f in os.listdir(out) if f.endswith('.png')]) == 6
        layers.import_(d, os.path.join(out, '00_grid.png'))     # unchanged image -> same map
        assert (np.array(Image.open(os.path.join(d, 'grid.pgm'))) == img).all()
        assert os.path.exists(os.path.join(d, 'grid_auto.pgm'))


def test_pcd_formats():
    xyz = np.random.default_rng(1).uniform(-5, 5, (100, 3)).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        write_pcd(os.path.join(d, 'b.pcd'), xyz)
        assert np.allclose(read_pcd(os.path.join(d, 'b.pcd')), xyz)
        with open(os.path.join(d, 'a.pcd'), 'w') as f:
            f.write('VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n'
                    f'WIDTH {len(xyz)}\nHEIGHT 1\nPOINTS {len(xyz)}\nDATA ascii\n')
            for p in xyz:
                f.write(f'{p[0]} {p[1]} {p[2]} 7\n')
        assert np.allclose(read_pcd(os.path.join(d, 'a.pcd')), xyz, atol=1e-5)
        # binary_compressed: field blocks, LZF with literal runs only
        raw = b''.join(xyz[:, i].tobytes() for i in range(3))
        comp = b''.join(bytes([len(raw[i:i + 32]) - 1]) + raw[i:i + 32] for i in range(0, len(raw), 32))
        with open(os.path.join(d, 'c.pcd'), 'wb') as f:
            f.write((f'VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {len(xyz)}\nHEIGHT 1\n'
                     f'POINTS {len(xyz)}\nDATA binary_compressed\n').encode())
            f.write(struct.pack('<II', len(comp), len(raw)) + comp)
        assert np.allclose(read_pcd(os.path.join(d, 'c.pcd')), xyz)
