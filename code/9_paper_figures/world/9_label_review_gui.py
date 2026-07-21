"""Tkinter GUI for manually relabelling the regenerated sample plots and
comparing against the algorithm's `plume_label`.

Features:
  - Picks one of the three sample folders
    (tune / val_random / val_stratified) on launch.
  - Shows the 6-panel PNG with plant metadata and the algorithm's label.
  - Four big buttons: Plume / No Plume / Unsure / Skip.
  - Auto-saves to `manual_annotations.csv` after every keypress.
  - On reopen, resumes from the first row whose `human_label` is empty.
  - Live agreement rate + 2x2 confusion-matrix counts.

Keyboard shortcuts:
  1 = plume   2 = no plume   3 = unsure   4 = skip
  Left / Right arrows         = previous / next sample
  Home                        = jump to first unlabeled

Usage:
    python 9_label_review_gui.py
"""
import os
import sys
import csv
import datetime as dt
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import pandas as pd
from PIL import Image, ImageTk


SAMPLES_ROOT = Path('/net/fs06/d3/rzhuang/TROPOMI/data/world/paper_figures/labeling_samples')
ANNOTATIONS_FILE = 'manual_annotations.csv'
LABEL_OPTIONS = [
    ('plume',     '✓ Plume',     '#28a745', '1'),
    ('no_plume',  '✗ No Plume',  '#dc3545', '2'),
    ('unsure',    '? Unsure',    '#fd7e14', '3'),
    ('skip',      '⏭ Skip',      '#6c757d', '4'),
]


class LabelReviewApp:
    def __init__(self, root, samples_dir: Path):
        self.root = root
        self.samples_dir = samples_dir
        self.csv_path = samples_dir / 'sampled_emission_snapshots.csv'
        self.ann_path = samples_dir / ANNOTATIONS_FILE

        self.df = self._load_samples()
        self.annotations = self._load_annotations()
        self.idx = self._first_unlabeled_idx()

        self._build_ui()
        self._bind_keys()
        self._refresh()

    # ─── Data ────────────────────────────────────────────────────────────
    def _load_samples(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        # Build the PNG file name as the regenerator wrote it.
        def png_for(row):
            loc = row.get('location', 'UnknownLocation')
            iso = row.get('country', row.get('ISO3', 'UnknownISO'))
            t   = row.get('utc_time', 'UnknownTime')
            return self.samples_dir / f"sampled_location_{loc}_{iso}_{t}.png"
        df['_png'] = df.apply(png_for, axis=1)
        df['_png_exists'] = df['_png'].apply(lambda p: p.exists())
        df['_sample_id']  = df['_png'].apply(lambda p: p.stem)

        missing = (~df['_png_exists']).sum()
        if missing:
            print(f"Warning: {missing} of {len(df)} CSV rows have no PNG on disk.")
        df = df[df['_png_exists']].reset_index(drop=True)
        if df.empty:
            raise RuntimeError(f"No PNGs found under {self.samples_dir}")
        return df

    def _load_annotations(self) -> dict:
        """Read existing manual_annotations.csv (if any) into {sample_id: row}."""
        if not self.ann_path.exists():
            return {}
        a = pd.read_csv(self.ann_path)
        return {r['sample_id']: dict(r) for _, r in a.iterrows()}

    def _save_annotation(self, sample_id, label, row):
        self.annotations[sample_id] = {
            'sample_id':     sample_id,
            'location':      row.get('location'),
            'country':       row.get('country', row.get('ISO3', '')),
            'utc_time':      row.get('utc_time'),
            'algo_label':    bool(row.get('plume_label', False)),
            'human_label':   label,
            'timestamp':     dt.datetime.now().isoformat(timespec='seconds'),
        }
        out = pd.DataFrame(list(self.annotations.values()))
        out.to_csv(self.ann_path, index=False)

    def _first_unlabeled_idx(self) -> int:
        for i, row in self.df.iterrows():
            sid = row['_sample_id']
            if sid not in self.annotations or not self.annotations[sid].get('human_label'):
                return i
        return 0  # all labeled — start from the beginning so the user can review

    # ─── UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.root.title(f"Manual labeling — {self.samples_dir.name}")
        self.root.geometry('1500x1000')
        self.root.configure(bg='#f6f7f9')

        # Top bar: progress + folder name
        top = tk.Frame(self.root, bg='#1f2933', height=48)
        top.pack(fill='x')
        self.title_lbl = tk.Label(top, text='', bg='#1f2933', fg='white',
                                  font=('Helvetica', 14, 'bold'))
        self.title_lbl.pack(side='left', padx=16, pady=12)
        self.progress_lbl = tk.Label(top, text='', bg='#1f2933', fg='#9aa5b1',
                                     font=('Helvetica', 12))
        self.progress_lbl.pack(side='right', padx=16, pady=12)

        # Main split: image (left) | meta + buttons + stats (right)
        body = tk.Frame(self.root, bg='#f6f7f9')
        body.pack(fill='both', expand=True, padx=12, pady=12)

        self.img_canvas = tk.Label(body, bg='white', relief='solid', bd=1)
        self.img_canvas.pack(side='left', fill='both', expand=True, padx=(0, 12))

        side = tk.Frame(body, bg='#f6f7f9', width=420)
        side.pack(side='right', fill='y')
        side.pack_propagate(False)

        # Plant metadata
        meta_box = tk.LabelFrame(side, text='Sample', bg='#f6f7f9',
                                 font=('Helvetica', 11, 'bold'),
                                 padx=12, pady=8)
        meta_box.pack(fill='x', pady=(0, 12))
        self.meta_lbl = tk.Label(meta_box, text='', bg='#f6f7f9',
                                 font=('Courier', 11), justify='left',
                                 anchor='w')
        self.meta_lbl.pack(fill='x')

        # Algorithm label highlight
        algo_box = tk.LabelFrame(side, text="Algorithm's label", bg='#f6f7f9',
                                 font=('Helvetica', 11, 'bold'), padx=12, pady=8)
        algo_box.pack(fill='x', pady=(0, 12))
        self.algo_lbl = tk.Label(algo_box, text='', bg='#f6f7f9',
                                 font=('Helvetica', 14, 'bold'),
                                 anchor='center')
        self.algo_lbl.pack(fill='x', pady=4)

        # Label buttons
        btn_box = tk.LabelFrame(side, text='Your label  (1/2/3/4)',
                                bg='#f6f7f9', font=('Helvetica', 11, 'bold'),
                                padx=12, pady=8)
        btn_box.pack(fill='x', pady=(0, 12))
        for label, text, color, key in LABEL_OPTIONS:
            b = tk.Button(btn_box, text=f"{text}  [{key}]", bg=color, fg='white',
                          font=('Helvetica', 13, 'bold'), relief='flat',
                          activebackground=color, padx=10, pady=8,
                          command=lambda lbl=label: self._record(lbl))
            b.pack(fill='x', pady=3)

        # Navigation
        nav = tk.Frame(side, bg='#f6f7f9')
        nav.pack(fill='x', pady=(0, 12))
        tk.Button(nav, text='← Prev',     command=self._prev,
                  font=('Helvetica', 11)).pack(side='left', expand=True, fill='x', padx=2)
        tk.Button(nav, text='Next →',     command=self._next,
                  font=('Helvetica', 11)).pack(side='left', expand=True, fill='x', padx=2)
        tk.Button(nav, text='Resume ⏵',   command=self._resume,
                  font=('Helvetica', 11)).pack(side='left', expand=True, fill='x', padx=2)

        # Stats panel
        stats_box = tk.LabelFrame(side, text='Agreement so far',
                                  bg='#f6f7f9', font=('Helvetica', 11, 'bold'),
                                  padx=12, pady=8)
        stats_box.pack(fill='x', pady=(0, 12))
        self.stats_lbl = tk.Label(stats_box, text='', bg='#f6f7f9',
                                  font=('Courier', 10), justify='left',
                                  anchor='w')
        self.stats_lbl.pack(fill='x')

        # Open folder
        tk.Button(side, text=f"📂  Open  {self.ann_path.name}",
                  command=lambda: self._open_in_filemanager(self.ann_path),
                  font=('Helvetica', 9)).pack(fill='x')

    def _bind_keys(self):
        for label, _, _, key in LABEL_OPTIONS:
            self.root.bind(f'<Key-{key}>', lambda e, lbl=label: self._record(lbl))
        self.root.bind('<Right>',   lambda e: self._next())
        self.root.bind('<Left>',    lambda e: self._prev())
        self.root.bind('<Home>',    lambda e: self._resume())
        self.root.bind('<Escape>',  lambda e: self.root.destroy())

    # ─── Actions ─────────────────────────────────────────────────────────
    def _record(self, label):
        row = self.df.iloc[self.idx]
        self._save_annotation(row['_sample_id'], label, row)
        self._next()

    def _next(self):
        if self.idx < len(self.df) - 1:
            self.idx += 1
        self._refresh()

    def _prev(self):
        if self.idx > 0:
            self.idx -= 1
        self._refresh()

    def _resume(self):
        self.idx = self._first_unlabeled_idx()
        self._refresh()

    def _open_in_filemanager(self, path):
        if not path.exists():
            messagebox.showinfo('Not yet', 'No annotations file written yet.')
            return
        os.system(f'xdg-open "{path.parent}" 2>/dev/null &')

    # ─── Render ──────────────────────────────────────────────────────────
    def _refresh(self):
        row = self.df.iloc[self.idx]
        sid = row['_sample_id']
        ann = self.annotations.get(sid, {})

        # Image
        try:
            img = Image.open(row['_png'])
            cw = max(800, self.img_canvas.winfo_width()  or 800)
            ch = max(600, self.img_canvas.winfo_height() or 600)
            img.thumbnail((cw, ch), Image.LANCZOS)
            self._tk_img = ImageTk.PhotoImage(img)
            self.img_canvas.configure(image=self._tk_img)
        except Exception as e:
            self.img_canvas.configure(text=f"Could not load image:\n{e}",
                                      image='', font=('Helvetica', 12))

        # Title + progress
        labelled = sum(1 for v in self.annotations.values() if v.get('human_label'))
        self.title_lbl.configure(
            text=f"{self.samples_dir.name}    sample {self.idx + 1} / {len(self.df)}")
        you = ann.get('human_label', '— unlabeled —')
        self.progress_lbl.configure(text=f"labeled so far: {labelled}/{len(self.df)}    "
                                          f"you: {you}")

        # Metadata
        emis_kg = None
        if 'NOx Mass (lbs)' in row and pd.notna(row['NOx Mass (lbs)']):
            emis_kg = row['NOx Mass (lbs)'] * 0.453592
        elif 'annual_nox_emission' in row and pd.notna(row['annual_nox_emission']):
            emis_kg = row['annual_nox_emission']
        meta = [
            f"location  : {row.get('location', '?')}",
            f"country   : {row.get('country', row.get('ISO3', '?'))}",
            f"utc_time  : {row.get('utc_time', '?')}",
            f"lat, lon  : {row.get('latitude', '?'):.3f}, {row.get('longitude', '?'):.3f}"
                if pd.notna(row.get('latitude')) else 'lat, lon  : ?',
            f"emission  : {emis_kg:.1f}" if emis_kg is not None else 'emission  : ?',
            f"continent : {row.get('continent', '—')}",
            f"_dataset  : {row.get('_dataset', '—')}",
        ]
        self.meta_lbl.configure(text='\n'.join(meta))

        # Algorithm label highlight
        algo = bool(row.get('plume_label', False))
        if algo:
            self.algo_lbl.configure(text='PLUME', bg='#d4edda', fg='#155724')
        else:
            self.algo_lbl.configure(text='NO PLUME', bg='#f8d7da', fg='#721c24')

        # Stats: confusion matrix where applicable
        labeled = [v for v in self.annotations.values()
                   if v.get('human_label') in ('plume', 'no_plume')]
        if labeled:
            tp = sum(1 for v in labeled if v['algo_label'] and v['human_label'] == 'plume')
            tn = sum(1 for v in labeled if not v['algo_label'] and v['human_label'] == 'no_plume')
            fp = sum(1 for v in labeled if v['algo_label'] and v['human_label'] == 'no_plume')
            fn = sum(1 for v in labeled if not v['algo_label'] and v['human_label'] == 'plume')
            n  = tp + tn + fp + fn
            agree = (tp + tn) / n * 100 if n else 0.0
            stats = (
                f"  n labeled (plume/no): {n}\n"
                f"  agreement rate     : {agree:5.1f}%\n"
                f"\n"
                f"  algorithm \\ you      plume   no plume\n"
                f"  plume                {tp:5d}    {fp:7d}\n"
                f"  no plume             {fn:5d}    {tn:7d}\n"
            )
        else:
            stats = '  (label some samples to see agreement)'
        self.stats_lbl.configure(text=stats)


def pick_samples_dir():
    """Prompt the user to choose one of the three sample folders."""
    candidates = [p for p in SAMPLES_ROOT.iterdir()
                  if p.is_dir() and (p / 'sampled_emission_snapshots.csv').exists()]
    if not candidates:
        raise RuntimeError(f"No sample folders under {SAMPLES_ROOT}")
    if len(candidates) == 1:
        return candidates[0]

    pick_root = tk.Tk()
    pick_root.title('Pick a sample folder')
    pick_root.geometry('400x260')
    pick_root.configure(bg='#f6f7f9')

    chosen = {'path': None}
    tk.Label(pick_root, text='Which sample set?',
             bg='#f6f7f9', font=('Helvetica', 13, 'bold')).pack(pady=(16, 8))
    for c in candidates:
        tk.Button(pick_root, text=c.name, font=('Helvetica', 12),
                  width=24, pady=6,
                  command=lambda p=c: (chosen.update(path=p), pick_root.destroy())
                  ).pack(pady=4)
    pick_root.mainloop()
    return chosen['path']


if __name__ == '__main__':
    if len(sys.argv) > 1:
        chosen = Path(sys.argv[1])
    else:
        chosen = pick_samples_dir()
    if chosen is None:
        sys.exit(0)

    root = tk.Tk()
    LabelReviewApp(root, chosen)
    root.mainloop()
