"""The main display window with image list, peak-finding & filtering details"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import numpy as np

from image_store import ImageStore
from image_window import ImageWindow
from peakfinders import discover_peakfinders
from peakfinders.base import BasePeakFinder
from powder import LatticeParameters, powder_lines
from peak_eval import evaluate_image


class MainWindow:

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title('Systematic – Control Panel')
        self.root.geometry('560x680')
        self.root.resizable(True, True)

        # Central data store
        self._store = ImageStore()
        self._store.add_listener(self._on_store_event)

        # filepath - ImageWindow dictionary
        self._image_windows: dict[Path, ImageWindow] = {}

        # Discover peakfinder plugins
        self._finders: list[BasePeakFinder] = discover_peakfinders()
        if not self._finders:
            messagebox.showerror(
                "No peakfinders",
                "No peakfinder plugins found in the peakfinders/ directory.\n"
                "Add at least one file implementing BasePeakFinder.",
            )

        self._build_ui()

    def _build_ui(self) -> None:
        root = self.root

        # ── File list ────────────────────────────────────────────────
        lf = tk.LabelFrame(root, text='Loaded Files', padx=6, pady=6)
        lf.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        sb = tk.Scrollbar(lf, orient=tk.VERTICAL)
        self._listbox = tk.Listbox(lf, yscrollcommand=sb.set, selectmode=tk.EXTENDED, activestyle='dotbox', height=5)
        sb.config(command=self._listbox.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._listbox.bind("<Double-Button-1>", self._focus_image_window)

        btn_row = tk.Frame(root)
        btn_row.pack(fill=tk.X, padx=10, pady=(0, 4))
        b = tk.Button(btn_row, text='Open TIFF(s)...', command=self._open_files, width=14)
        b.pack(side=tk.LEFT, padx=2)
        b = tk.Button(btn_row, text="Remove Selected", command=self._remove_selected, width=16)
        b.pack(side=tk.LEFT, padx=2)
        b = tk.Button(btn_row, text="Clear All", command=self._clear_all, width=10)
        b.pack(side=tk.LEFT, padx=2)

        ttk.Separator(root, orient='horizontal').pack(fill=tk.X, padx=8, pady=2)

        # ── Peakfinder notebook ──────────────────────────────────────
        pf_outer = tk.LabelFrame(root, text="Peak Finder", padx=6, pady=6)
        pf_outer.pack(fill=tk.BOTH, expand=False, padx=10, pady=4)

        self._notebook = ttk.Notebook(pf_outer)
        self._notebook.pack(fill=tk.BOTH, expand=True)

        for finder in self._finders:
            tab = ttk.Frame(self._notebook, padding=6)
            self._notebook.add(tab, text=finder.NAME)
            ff = finder.frame(tab)
            ff.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # ── Filtering tools ───────────────────────────────────────────

        ttk.Separator(root, orient='horizontal').pack(fill=tk.X, padx=8, pady=4)

        uc_frame = tk.Frame(root)
        uc_frame.pack(fill=tk.X, padx=10, pady=4)

        self.uc_a = tk.DoubleVar(value=5.0)
        self.uc_b = tk.DoubleVar(value=5.0)
        self.uc_c = tk.DoubleVar(value=5.0)
        self.uc_al = tk.DoubleVar(value=90.0)
        self.uc_be = tk.DoubleVar(value=90.0)
        self.uc_ga = tk.DoubleVar(value=90.0)
        self.uc_ct = tk.StringVar(value='P')
        self.a_per_pixel = tk.DoubleVar(value=0.01)
        # self.hca_threshold = tk.DoubleVar(value=0.01)

        for i in range(8):
            uc_frame.grid_columnconfigure(i, weight=1)

        ttk.Label(uc_frame, text='a [Å]:', anchor='w').grid(row=0, column=0, sticky='w')
        ttk.Label(uc_frame, text='b [Å]:', anchor='w').grid(row=0, column=2, sticky='w')
        ttk.Label(uc_frame, text='c [Å]:', anchor='w').grid(row=0, column=4, sticky='w')
        ttk.Label(uc_frame, text='α [°]:', anchor='w').grid(row=1, column=0, sticky='w')
        ttk.Label(uc_frame, text='β [°]:', anchor='w').grid(row=1, column=2, sticky='w')
        ttk.Label(uc_frame, text='γ [°]:', anchor='w').grid(row=1, column=4, sticky='w')
        ttk.Label(uc_frame, text='Centering:', anchor='w').grid(row=2, column=0, sticky='w')
        ttk.Label(uc_frame, text='Å-1 per pixel:', anchor='w').grid(row=2, column=2, sticky='w')
        # ttk.Label(uc_frame, text='HCA threshold:', anchor='w').grid(row=2, column=4, sticky='w')

        ttk.Entry(uc_frame, textvariable=self.uc_a, width=5).grid(row=0, column=1)
        ttk.Entry(uc_frame, textvariable=self.uc_b, width=5).grid(row=0, column=3)
        ttk.Entry(uc_frame, textvariable=self.uc_c, width=5).grid(row=0, column=5)
        ttk.Entry(uc_frame, textvariable=self.uc_al, width=5).grid(row=1, column=1)
        ttk.Entry(uc_frame, textvariable=self.uc_be, width=5).grid(row=1, column=3)
        ttk.Entry(uc_frame, textvariable=self.uc_ga, width=5).grid(row=1, column=5)
        ttk.Entry(uc_frame, textvariable=self.uc_ct, width=5).grid(row=2, column=1)
        ttk.Entry(uc_frame, textvariable=self.a_per_pixel, width=5).grid(row=2, column=3)
        # ttk.Entry(uc_frame, textvariable=self.hca_threshold, width=5).grid(row=2, column=5)

        b = tk.Button(uc_frame, text='Load CIF...', command=self._load_cif, width=10)
        b.grid(row=2, column=4, columnspan=2, sticky='ew', padx=2, pady=2)

        # ── Action buttons ───────────────────────────────────────────
        ttk.Separator(root, orient='horizontal').pack(fill=tk.X, padx=8, pady=4)

        action_frame = tk.Frame(root)
        action_frame.pack(fill=tk.X, padx=10, pady=4)

        self._run_all_btn = tk.Button(
            action_frame,
            text="▶  Run All Images",
            font=("Helvetica", 11, "bold"),
            bg="#2d7dd2",
            fg="white",
            activebackground="#1a5da0",
            relief=tk.FLAT,
            padx=10,
            pady=6,
            command=self._run_all,
        )
        self._run_all_btn.pack(side=tk.LEFT, padx=2)

        tk.Button(
            action_frame,
            text="Run Selected",
            command=self._run_selected,
            padx=6,
            pady=4,
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            action_frame,
            text='Evaluate diffraction',
            command=self._eval_diffraction,
            padx=6,
            pady=4,
        ).pack(side=tk.LEFT, padx=6)

        # ── Status bar ───────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Ready. Open TIFF files to begin.")
        tk.Label(
            root,
            textvariable=self._status_var,
            anchor="w",
            relief=tk.SUNKEN,
            bd=1,
            pady=3,
        ).pack(side=tk.BOTTOM, fill=tk.X)



    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------

    def _open_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select TIFF images",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
        )
        added = 0
        errors = []
        for p in paths:
            path = Path(p)
            if path in self._store:
                continue
            try:
                self._store.load(path)   # store fires "loaded" → _on_store_event
                added += 1
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")

        if errors:
            messagebox.showerror("Load errors", "\n".join(errors))
        if added:
            self._status_var.set(
                f"Loaded {added} file(s). {len(self._store)} total."
            )

    def _remove_selected(self) -> None:
        indices = list(self._listbox.curselection())
        if not indices:
            return
        paths = self._store.paths()
        for i in reversed(indices):
            if i < len(paths):
                self._store.remove(paths[i])   # fires "removed" → _on_store_event

    def _clear_all(self) -> None:
        self._store.clear()   # fires "cleared" → _on_store_event

    def _on_store_event(self, path: Path | None, event: str) -> None:
        """Sync listbox with the store; open/close image windows as needed."""
        if event == 'loaded' and path is not None:
            self._listbox.insert(tk.END, path.name)
            win = ImageWindow(
                parent=self.root,
                store=self._store,
                filepath=path,
                on_close=self._on_image_window_closed,
            )
            self._image_windows[path] = win

        elif event == 'removed' and path is not None:
            self._close_image_window(path)
            self._sync_listbox()

        elif event == 'cleared':
            for p in list(self._image_windows.keys()):
                self._close_image_window(p)
            self._listbox.delete(0, tk.END)
            self._status_var.set('All files cleared.')

    def _sync_listbox(self) -> None:
        self._listbox.delete(0, tk.END)
        for p in self._store.paths():
            self._listbox.insert(tk.END, p.name)

    def _close_image_window(self, path: Path) -> None:
        win = self._image_windows.pop(path, None)
        if win:
            try:
                win._store.remove_listener(win._on_store_event)
                import matplotlib.pyplot as plt
                plt.close(win._fig)
                win.win.destroy()
            except Exception:
                pass

    def _on_image_window_closed(self, path: Path) -> None:
        """Callback from ImageWindow when user clicks its X button."""
        self._image_windows.pop(path, None)
        if path in self._store:
            self._store.remove(path)
        self._sync_listbox()
        self._status_var.set(
            f"Closed {path.name}. {len(self._store)} file(s) remain."
        )

    def _focus_image_window(self, _event=None) -> None:
        sel = self._listbox.curselection()
        if not sel:
            return
        paths = self._store.paths()
        idx = sel[0]
        if idx < len(paths):
            win = self._image_windows.get(paths[idx])
            if win:
                win.lift_to_front()

    # ------------------------------------------------------------------
    # Tab / peakfinder
    # ------------------------------------------------------------------

    def _active_finder(self) -> BasePeakFinder | None:
        idx = self._notebook.index('current')
        if 0 <= idx < len(self._finders):
            return self._finders[idx]
        return None

    # ------------------------------------------------------------------
    # Peak finding
    # ------------------------------------------------------------------

    def _run_on_paths(self, paths: list[Path]) -> None:
        finder = self._active_finder()
        if finder is None:
            messagebox.showinfo('No finder', 'No peakfinder tab is active.')
            return
        if not paths:
            messagebox.showinfo('No images', 'No images to process.')
            return

        self._status_var.set(f"Running [{finder.NAME}] on {len(paths)} image(s)...")
        self.root.update_idletasks()

        total = 0
        errors = []
        for path in paths:
            entry = self._store[path]
            try:
                result = finder.run(entry.raw)
                self._store.set_result(path, result)   # fires "result" → ImageWindow
                total += len(result)
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")

        if errors:
            messagebox.showerror("Run errors", "\n".join(errors))

        self._status_var.set(
            f"[{finder.NAME}] done — {total} peak(s) across "
            f"{len(paths)} image(s)."
        )

    def _run_all(self) -> None:
        self._run_on_paths(self._store.paths())

    def _run_selected(self) -> None:
        indices = self._listbox.curselection()
        if not indices:
            messagebox.showinfo("Nothing selected",
                                "Select one or more files in the list first.")
            return
        paths = self._store.paths()
        selected = [paths[i] for i in indices if i < len(paths)]
        self._run_on_paths(selected)

    # ------------------------------------------------------------------
    # Diffraction evaluation
    # ------------------------------------------------------------------

    def _load_cif(self) -> None:
        """Open a CIF file and populate the unit-cell entry fields from it."""
        path = filedialog.askopenfilename(
            title='Open CIF file',
            filetypes=[('CIF files', '*.cif'), ('All files', '*.*')],
        )
        if not path:
            return
        try:
            lp = LatticeParameters.from_cif(path)
        except Exception as exc:
            messagebox.showerror('CIF load error', f'Could not read {Path(path).name}:\n{exc}')
            return

        self.uc_a.set(round(lp.a, 5))
        self.uc_b.set(round(lp.b, 5))
        self.uc_c.set(round(lp.c, 5))
        self.uc_al.set(round(lp.alpha, 4))
        self.uc_be.set(round(lp.beta,  4))
        self.uc_ga.set(round(lp.gamma, 4))
        self.uc_ct.set(lp.centering)

        self._status_var.set(
            f'CIF loaded: a={lp.a:.4f} b={lp.b:.4f} c={lp.c:.4f}  '
            f'α={lp.alpha:.2f} β={lp.beta:.2f} γ={lp.gamma:.2f}  '
            f'centering={lp.centering}'
        )

    def _read_lattice_params(self) -> LatticeParameters | None:
        """Read cell parameters and centering from the GUI; return None on error."""
        try:
            lp = LatticeParameters(
                a=self.uc_a.get(),
                b=self.uc_b.get(),
                c=self.uc_c.get(),
                alpha=self.uc_al.get(),
                beta=self.uc_be.get(),
                gamma=self.uc_ga.get(),
                centering=self.uc_ct.get().strip(),
            )
        except Exception as exc:
            messagebox.showerror('Invalid unit cell', str(exc))
            return None
        return lp

    def _eval_diffraction(self) -> None:
        """
        For every open image that has a peak result:

        1. Compute the powder pattern from the GUI unit-cell fields.
        2. Compute pairwise peak distances and match them to powder lines.
        3. Push a PairwiseEval to each ImageWindow so it can draw the
           colour-coded distance overlay and the score annotation.
        4. Report a summary in the status bar.
        """
        if not self._store.paths():
            messagebox.showinfo('No images', 'Open at least one TIFF first.')
            return

        # ── 1. Build lattice params from GUI ─────────────────────────
        lp = self._read_lattice_params()
        if lp is None:
            return

        app = self.a_per_pixel.get()
        if app <= 0:
            messagebox.showerror('Invalid calibration', 'Å-1/pixel must be positive.')
            return

        # ── 2. Compute powder lines ───────────────────────────────────
        # s_max covers the largest inter-peak distance observable on the
        # biggest loaded image; 5% headroom avoids clipping edge cases.
        try:
            max_diag_px = max(
                (float(np.hypot(*self._store[p].raw.shape)) for p in self._store.paths()),
                default=1024.0,
            )
        except Exception:
            max_diag_px = 1024.0

        s_max = max_diag_px * app * 1.05

        try:
            lines = powder_lines(lp, s_max=s_max)
        except Exception as exc:
            messagebox.showerror('Powder pattern error', str(exc))
            return

        if len(lines) == 0:
            messagebox.showwarning(
                'No powder lines',
                f'No reflections found for the given cell up to s_max={s_max:.3f} Å-1.\n'
                'Check your unit-cell parameters and Å-1/px calibration.',
            )
            return

        # ── 3. Evaluate each image ────────────────────────────────────
        self._status_var.set('Evaluating...')
        self.root.update_idletasks()

        scores = []
        n_evaluated = 0
        for path in self._store.paths():
            result = self._store.get_result(path)
            win    = self._image_windows.get(path)

            if result is None or len(result) < 2:
                # No peaks yet — clear any previous overlay
                if win:
                    win.clear_eval()
                continue

            ev = evaluate_image(
                rows=result.rows,
                cols=result.cols,
                powder_inv_d=lines,
                angstrom_per_pixel=app,
            )

            if win:
                win.update_eval(ev)

            scores.append(ev.powder_deviation)
            n_evaluated += 1

        # ── 4. Status summary ─────────────────────────────────────────
        if n_evaluated == 0:
            self._status_var.set(
                'Evaluation done — no images had peak results yet. '
                'Run peak finding first.'
            )
            return

        mean_score = float(np.mean(scores))
        scores_str = '  '.join(f'{s:.3f}' for s in scores)
        self._status_var.set(
            f'Evaluated {n_evaluated} image(s) — '
            f'scores: [{scores_str}]  mean: {mean_score:.3f}  '
            f'(cell: {lp.a:.3f}x{lp.b:.3f}x{lp.c:.3f} A  {lp.centering})'
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.root.mainloop()