"""Command line entry points: `pcd2nav2map` (GUI / batch) and `pcd2nav2map-layers` (GIMP export / import)."""
import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog='pcd2nav2map',
                                 description='3D point cloud map (.pcd) -> Nav2 2D occupancy map (.pgm + .yaml)')
    ap.add_argument('path', nargs='?', default=None, help='map folder (map.pcd [+ poses.txt] [+ grid.yaml]) or a .pcd')
    ap.add_argument('--no-gui', action='store_true', help='convert and save without opening a window')
    ap.add_argument('--zmin', type=float, default=0.15, help='lower end of the height band (m)')
    ap.add_argument('--zmax', type=float, default=0.8, help='upper end of the height band (m)')
    ap.add_argument('--res', type=float, default=0.05, help='cell size (m)')
    ap.add_argument('--min-points', type=int, default=2, help='points needed in a cell to mark it occupied')
    ap.add_argument('--absolute', action='store_true', help='use map z instead of height above the local floor')
    ap.add_argument('--poses', default=None, help='trajectory file for the local floor (default: <folder>/poses.txt)')
    ap.add_argument('--no-align', action='store_true', help="don't reuse grid.yaml's origin and size")
    ap.add_argument('--no-unknown', action='store_true', help='every cell without points is free')
    ap.add_argument('--despeckle', type=int, default=0, help='drop occupied blobs of at most this many cells')
    ap.add_argument('--out', default=None, help='output .pgm (default <folder>/grid_points.pgm)')
    a = ap.parse_args(argv)
    if not a.no_gui:
        from .gui import run
        run(a.path)
        return 0
    if not a.path:
        ap.error('path is required with --no-gui')
    from .core import Source, make_map, save_map
    src = Source(a.path, poses=a.poses)
    img, res, origin, st = make_map(src, a.zmin, a.zmax, a.res, a.min_points, not a.absolute, not a.no_align,
                                    not a.no_unknown, a.despeckle)
    p, y = save_map(a.out or os.path.join(src.dir, 'grid_points.pgm'), img, res, origin)
    print(f'{st["height"]} {a.zmin}..{a.zmax} m, {st["size"][0]} x {st["size"][1]} @ {res:g} m: '
          f'occupied {st["occupied"]}, free {st["free"]}, unknown {st["unknown"]} -> {p}, {os.path.basename(y)}')
    return 0


def layers_main(argv=None):
    ap = argparse.ArgumentParser(prog='pcd2nav2map-layers', description='edit a Nav2 map by hand in GIMP')
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('export', help='point cloud -> height-band PNG layers aligned with grid.pgm')
    e.add_argument('map_dir')
    e.add_argument('--out', default=None, help='output folder (default <map_dir>/grid_edit)')
    e.add_argument('--min-points', type=int, default=2)
    i = sub.add_parser('import', help='edited image -> grid.pgm (old one kept as *_auto.pgm)')
    i.add_argument('map_dir')
    i.add_argument('image')
    a = ap.parse_args(argv)
    from . import layers
    if a.cmd == 'export':
        layers.export(a.map_dir, a.out, a.min_points)
    else:
        layers.import_(a.map_dir, a.image)
    return 0


if __name__ == '__main__':
    sys.exit(main())
