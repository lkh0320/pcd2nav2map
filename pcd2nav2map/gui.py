"""Tkinter GUI: pick a height band with sliders, preview the grid, save pgm/yaml."""
import os
import shutil

import numpy as np

from .core import Source, make_map, save_map


def run(start=None):
    import tkinter as tk
    from tkinter import filedialog, messagebox

    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title('pcd2nav2map - point cloud to Nav2 map')
    st = {'src': None, 'img': None, 'res': None, 'origin': None, 'tk': None}
    VIEW = 640

    left = tk.Frame(root, padx=8, pady=8)
    left.grid(row=0, column=0, sticky='n')
    canvas = tk.Canvas(root, width=VIEW, height=VIEW, bg='#808080', highlightthickness=0)
    canvas.grid(row=0, column=1, padx=4, pady=4)

    def row(r, text, widget):
        tk.Label(left, text=text, anchor='w').grid(row=r, column=0, sticky='w', pady=2)
        widget.grid(row=r, column=1, sticky='we', pady=2)
        return widget

    path_var = tk.StringVar(value=start or '')
    tk.Label(left, text='Map folder or .pcd file').grid(row=0, column=0, columnspan=2, sticky='w')
    tk.Entry(left, textvariable=path_var, width=36).grid(row=1, column=0, columnspan=2, sticky='we')
    bf = tk.Frame(left)
    bf.grid(row=2, column=0, columnspan=2, sticky='we', pady=(2, 6))

    mode_var = tk.StringVar(value='rel')
    zmin_var, zmax_var = tk.DoubleVar(value=0.15), tk.DoubleVar(value=0.8)
    res_var, thr_var, desp_var = tk.StringVar(value='0.05'), tk.StringVar(value='2'), tk.StringVar(value='0')
    align_var, unk_var = tk.BooleanVar(value=True), tk.BooleanVar(value=True)

    mf = tk.Frame(left)
    tk.Radiobutton(mf, text='above local floor', variable=mode_var, value='rel',
                   command=lambda: on_mode()).pack(side='left')
    tk.Radiobutton(mf, text='absolute z', variable=mode_var, value='abs', command=lambda: on_mode()).pack(side='left')
    row(3, 'Height', mf)
    zmin_s = row(4, 'Z min (m)', tk.Scale(left, variable=zmin_var, from_=-1.0, to=2.0, resolution=0.01,
                                        orient='horizontal', length=210))
    zmax_s = row(5, 'Z max (m)', tk.Scale(left, variable=zmax_var, from_=-1.0, to=2.5, resolution=0.01,
                                        orient='horizontal', length=210))
    row(6, 'Resolution (m)', tk.Entry(left, textvariable=res_var, width=8))
    row(7, 'Min points / cell', tk.Entry(left, textvariable=thr_var, width=8))
    row(8, 'Drop blobs <= (cells)', tk.Entry(left, textvariable=desp_var, width=8))
    row(9, '', tk.Checkbutton(left, text='Same origin/size as grid.yaml', variable=align_var))
    row(10, '', tk.Checkbutton(left, text='Keep unobserved cells unknown', variable=unk_var))
    af = tk.Frame(left)
    af.grid(row=11, column=0, columnspan=2, sticky='we', pady=(8, 0))
    info = tk.Label(left, text='-', anchor='w', justify='left', fg='#333')
    info.grid(row=12, column=0, columnspan=2, sticky='w', pady=(8, 0))
    tk.Label(left, text='black = occupied, white = free, grey = unknown\n'
                        'ground robot: ~0.15 .. robot height above the floor\n'
                        'lower Z min (e.g. 0.06) to include curbs / thresholds',
             anchor='w', justify='left', fg='#666').grid(row=13, column=0, columnspan=2, sticky='w', pady=(8, 0))

    def on_mode():
        if st['src'] is None:
            return
        h = st['src'].heights(mode_var.get() == 'rel')
        lo, hi = float(np.percentile(h, 0.5)), float(np.percentile(h, 99.5))
        for s in (zmin_s, zmax_s):
            s.config(from_=round(lo - 0.2, 1), to=round(hi + 0.2, 1))
        if mode_var.get() == 'rel':
            zmin_var.set(0.15)
            zmax_var.set(0.8)

    def load(path=None):
        p = path or path_var.get()
        if not p:
            return
        try:
            root.config(cursor='watch')
            root.update()
            st['src'] = Source(p)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror('Load failed', str(e))
            return
        finally:
            root.config(cursor='')
        s = st['src']
        path_var.set(p)
        if s.grid is not None:
            res_var.set(f'{s.grid[1]:g}')
        on_mode()
        info.config(text=f'{len(s.P):,} points, floor from {s.floor_source}\n'
                         f'grid.yaml: {"found (" + format(s.grid[1], "g") + " m)" if s.grid else "none"}')
        preview()

    def pick_dir():
        d = filedialog.askdirectory(initialdir=path_var.get() or os.getcwd(), title='Map folder')
        if d:
            load(d)

    def pick_pcd():
        f = filedialog.askopenfilename(initialdir=path_var.get() or os.getcwd(), filetypes=[('PCD', '*.pcd')])
        if f:
            load(f)

    def compute():
        s = st['src']
        if s is None:
            messagebox.showerror('Error', 'Load a map first')
            return None
        try:
            r, n, dsp = float(res_var.get()), int(thr_var.get()), int(desp_var.get() or 0)
        except ValueError:
            messagebox.showerror('Error', 'Resolution / min points / blob size must be numbers')
            return None
        if zmax_var.get() <= zmin_var.get():
            messagebox.showerror('Error', 'Z max must be larger than Z min')
            return None
        return make_map(s, zmin_var.get(), zmax_var.get(), r, n, mode_var.get() == 'rel', align_var.get(),
                        unk_var.get(), dsp)

    def preview():
        out = compute()
        if out is None:
            return
        img, r, o, stt = out
        st.update(img=img, res=r, origin=o)
        im = Image.fromarray(img)
        sc = min(VIEW / im.width, VIEW / im.height)
        im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))), Image.NEAREST)
        st['tk'] = ImageTk.PhotoImage(im)
        canvas.delete('all')
        canvas.create_image(VIEW // 2, VIEW // 2, image=st['tk'])
        info.config(text=f'{stt["height"]}\n{stt["size"][0]} x {stt["size"][1]} cells @ {r:g} m\n'
                         f'occupied {stt["occupied"]:,} / free {stt["free"]:,} / unknown {stt["unknown"]:,}'
                         + (f'\n{stt["removed"]} cells in small blobs dropped' if stt['removed'] else ''))

    def save():
        if st['img'] is None:
            preview()
        if st['img'] is None:
            return
        d = st['src'].dir if st['src'] else os.getcwd()
        f = filedialog.asksaveasfilename(initialdir=d, initialfile='grid_points.pgm', defaultextension='.pgm',
                                         filetypes=[('PGM', '*.pgm')])
        if not f:
            return
        target = os.path.abspath(f if f.endswith('.pgm') else f + '.pgm')
        if os.path.exists(target):
            if not messagebox.askyesno('Overwrite', f'{target} exists. A copy is kept as *_backup.pgm. Continue?'):
                return
            shutil.copy2(target, target[:-4] + '_backup.pgm')
        p, y = save_map(target, st['img'], st['res'], st['origin'])
        messagebox.showinfo('Saved', f'{p}\n{y}')

    tk.Button(bf, text='Open folder', command=pick_dir).pack(side='left')
    tk.Button(bf, text='Open .pcd', command=pick_pcd).pack(side='left', padx=4)
    tk.Button(bf, text='Reload', command=lambda: load()).pack(side='left')
    tk.Button(af, text='Preview', command=preview, width=10).pack(side='left')
    tk.Button(af, text='Save', command=save, width=10, bg='#2f7d4f', fg='white').pack(side='left', padx=6)

    if start:
        root.after(100, lambda: load(start))
    root.mainloop()
