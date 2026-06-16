"""
main_window.py
==============
Window 1 – control panel.

Layout
------
┌──────────────────────────────────────┐
│  Loaded Files                        │
│  [listbox]           [Open] [Remove] │
├──────────────────────────────────────┤
│  Peak Finder  ┌──────┬──────┬──────┐ │
│               │ Tab1 │ Tab2 │ ...  │ │
│               └──────┴──────┴──────┘ │
│               [parameter widgets]    │
├──────────────────────────────────────┤
│  [▶ Run All]  [Run Selected]         │
├──────────────────────────────────────┤
│  status bar                          │
└──────────────────────────────────────┘
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

from image_store import ImageStore
from image_window import ImageWindow
from peakfinders import discover_finders
from peakfinders.base import BasePeakFinder


class MainWindow:

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Systematic – Control Panel")
        self.root.geometry("560x680")
        self.root.resizable(True, True)

        # Central data store
        self._store = ImageStore()
        self._store.add_listener(self._on_store_event)

        # filepath - ImageWindow dictionary
        self._image_windows: dict[Path, ImageWindow] = {}

        # Discover peakfinder plugins
        self._finders: list[BasePeakFinder] = discover_finders()
        if not self._finders:
            messagebox.showerror(
                "No peakfinders",
                "No peakfinder plugins found in the peakfinders/ directory.\n"
                "Add at least one file implementing BasePeakFinder.",
            )

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = self.root

        # ── File list ────────────────────────────────────────────────
        list_frame = tk.LabelFrame(root, text="Loaded Files", padx=6, pady=6)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        sb = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self._listbox = tk.Listbox(
            list_frame,
            yscrollcommand=sb.set,
            selectmode=tk.EXTENDED,
            activestyle="dotbox",
            height=7,
        )
        sb.config(command=self._listbox.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._listbox.bind("<Double-Button-1>", self._focus_image_window)

        btn_row = tk.Frame(root)
        btn_row.pack(fill=tk.X, padx=10, pady=(0, 4))
        tk.Button(btn_row, text="Open TIFF(s)…",
                  command=self._open_files, width=14).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Remove Selected",
                  command=self._remove_selected, width=16).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Clear All",
                  command=self._clear_all, width=10).pack(side=tk.LEFT, padx=2)

        ttk.Separator(root, orient="horizontal").pack(fill=tk.X, padx=8, pady=2)

        # ── Peakfinder notebook ──────────────────────────────────────
        pf_outer = tk.LabelFrame(root, text="Peak Finder", padx=6, pady=6)
        pf_outer.pack(fill=tk.BOTH, expand=False, padx=10, pady=4)

        self._notebook = ttk.Notebook(pf_outer)
        self._notebook.pack(fill=tk.BOTH, expand=True)
        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_change)

        for finder in self._finders:
            tab = ttk.Frame(self._notebook, padding=6)
            self._notebook.add(tab, text=finder.NAME)

            # Each finder gets its own LabelFrame inside the tab
            inner = tk.LabelFrame(
                tab,
                text=finder.NAME,
                padx=6,
                pady=6,
            )
            inner.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            finder.build_ui(inner)

        # ── Action buttons ───────────────────────────────────────────
        ttk.Separator(root, orient="horizontal").pack(fill=tk.X, padx=8, pady=4)

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
        if event == "loaded" and path is not None:
            self._listbox.insert(tk.END, path.name)
            win = ImageWindow(
                parent=self.root,
                store=self._store,
                filepath=path,
                on_close=self._on_image_window_closed,
            )
            self._image_windows[path] = win

        elif event == "removed" and path is not None:
            self._close_image_window(path)
            self._sync_listbox()

        elif event == "cleared":
            for p in list(self._image_windows.keys()):
                self._close_image_window(p)
            self._listbox.delete(0, tk.END)
            self._status_var.set("All files cleared.")

        # "result" events are handled by ImageWindow directly

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

    def _on_tab_change(self, _event=None) -> None:
        finder = self._active_finder()
        if finder:
            finder.on_tab_selected()

    def _active_finder(self) -> BasePeakFinder | None:
        idx = self._notebook.index("current")
        if 0 <= idx < len(self._finders):
            return self._finders[idx]
        return None

    # ------------------------------------------------------------------
    # Peak finding
    # ------------------------------------------------------------------

    def _run_on_paths(self, paths: list[Path]) -> None:
        finder = self._active_finder()
        if finder is None:
            messagebox.showinfo("No finder", "No peakfinder tab is active.")
            return
        if not paths:
            messagebox.showinfo("No images", "No images to process.")
            return

        self._status_var.set(f"Running [{finder.NAME}] on {len(paths)} image(s)…")
        self.root.update_idletasks()

        total = 0
        errors = []
        for path in paths:
            entry = self._store[path]
            try:
                result = finder.run(entry.raw, path)
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
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.root.mainloop()
