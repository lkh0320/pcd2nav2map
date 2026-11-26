"""Minimal PCD reader (ascii / binary / binary_compressed) -> (N, 3) float64 xyz. No PCL / Open3D needed."""
import struct

import numpy as np

_TYPES = {('F', 4): 'f4', ('F', 8): 'f8', ('U', 1): 'u1', ('U', 2): 'u2', ('U', 4): 'u4', ('U', 8): 'u8',
          ('I', 1): 'i1', ('I', 2): 'i2', ('I', 4): 'i4', ('I', 8): 'i8'}


def _lzf_decompress(data, out_len):
    """LZF (as used by PCL binary_compressed), pure Python"""
    out = bytearray(out_len)
    ip = op = 0
    n = len(data)
    while ip < n:
        ctrl = data[ip]
        ip += 1
        if ctrl < 32:
            ctrl += 1
            out[op:op + ctrl] = data[ip:ip + ctrl]
            ip += ctrl
            op += ctrl
        else:
            length = ctrl >> 5
            ref = op - ((ctrl & 0x1f) << 8) - 1
            if length == 7:
                length += data[ip]
                ip += 1
            ref -= data[ip]
            ip += 1
            length += 2
            for _ in range(length):
                out[op] = out[ref]
                op += 1
                ref += 1
    return bytes(out)


def read_pcd(path):
    """-> xyz (N, 3) float64, finite points only"""
    with open(path, 'rb') as f:
        h = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f'{path}: no DATA line in header')
            line = line.decode('ascii', 'ignore').strip()
            if not line or line.startswith('#'):
                continue
            k, *v = line.split()
            h[k.upper()] = v
            if k.upper() == 'DATA':
                break
        fields = h['FIELDS']
        sizes = [int(s) for s in h['SIZE']]
        types = [t.upper() for t in h['TYPE']]
        counts = [int(c) for c in h.get('COUNT', ['1'] * len(fields))]
        n = int(h['POINTS'][0]) if 'POINTS' in h else int(h['WIDTH'][0]) * int(h['HEIGHT'][0])
        kind = h['DATA'][0].lower()
        for k in 'xyz':
            if k not in fields:
                raise ValueError(f'{path}: field {k} missing')
        if kind == 'ascii':
            cols = np.cumsum([0] + counts)
            arr = np.loadtxt(f, dtype=np.float64, ndmin=2)
            xyz = np.stack([arr[:, cols[fields.index(k)]] for k in 'xyz'], 1)
        else:
            dt = []
            for i, (name, s, t, c) in enumerate(zip(fields, sizes, types, counts)):
                base = _TYPES[(t, s)]
                nm = name if name not in ('_', '') else f'_pad{i}'
                dt.append((nm, base) if c == 1 else (nm, base, (c,)))
            if kind == 'binary':
                a = np.frombuffer(f.read(n * sum(s * c for s, c in zip(sizes, counts))), dtype=np.dtype(dt), count=n)
                xyz = np.stack([a[k] for k in 'xyz'], 1).astype(np.float64)
            elif kind == 'binary_compressed':
                csize, usize = struct.unpack('<II', f.read(8))
                raw = _lzf_decompress(f.read(csize), usize)
                # binary_compressed stores each field as a contiguous block
                off, cols = 0, {}
                for name, s, t, c in zip(fields, sizes, types, counts):
                    nb = n * s * c
                    if name in 'xyz' and len(name) == 1:
                        cols[name] = np.frombuffer(raw[off:off + nb], dtype=_TYPES[(t, s)])[::c].astype(np.float64)
                    off += nb
                xyz = np.stack([cols[k] for k in 'xyz'], 1)
            else:
                raise ValueError(f'{path}: unsupported DATA {kind}')
    return xyz[np.isfinite(xyz).all(1)]


def write_pcd(path, xyz):
    """binary xyz PCD (used by the tests and examples)"""
    xyz = np.asarray(xyz, np.float32)
    with open(path, 'wb') as f:
        f.write((f'# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {len(xyz)}\n'
                 f'HEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(xyz)}\nDATA binary\n').encode())
        f.write(xyz.tobytes())


def read_positions(path):
    """sensor / robot positions from a trajectory file -> (M, 3), or None.

    Accepts FAST-LIO2 PGO poses.txt (`name x y z qw qx qy qz`), TUM (`t x y z qx qy qz qw`) and plain `x y z ...`.
    """
    out = []
    with open(path) as f:
        for line in f:
            s = line.split()
            if not s or s[0].startswith('#'):
                continue
            try:
                vals = [float(v) for v in s]
                xyz = vals[1:4] if len(vals) == 8 else vals[:3]
            except ValueError:
                xyz = [float(v) for v in s[1:4]]
            out.append(xyz)
    return np.array(out) if out else None
