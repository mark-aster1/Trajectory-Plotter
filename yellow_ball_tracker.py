"""
Yellow Ball Tracker
-------------------
Frame-by-frame UI to click on yellow balls in a video, collect coordinates
for a configurable number of balls, auto-fit parabolas, view an in-app
graph, and export the fitted functions.
"""

import csv
import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


# ---------------------------------------------------------------------------
# Colour palette — cycles if more balls than colours
# ---------------------------------------------------------------------------
_PALETTE_BGR = [
    (0, 0, 220),     # red
    (0, 180, 0),     # green
    (220, 140, 0),   # blue
    (200, 0, 200),   # purple
    (0, 180, 220),   # orange
    (200, 200, 0),   # cyan
    (60, 60, 220),   # dark red
    (0, 220, 120),   # lime
]
_PALETTE_HEX = [
    "#dc0000", "#00b400", "#008cdc", "#c800c8",
    "#dcb400", "#00c8c8", "#dc3c3c", "#78dc00",
]


def ball_color_bgr(idx):
    return _PALETTE_BGR[idx % len(_PALETTE_BGR)]


def ball_color_hex(idx):
    return _PALETTE_HEX[idx % len(_PALETTE_HEX)]


def ball_name(idx):
    return f"Ball {idx + 1}"


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def build_yellow_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([18, 55, 150])
    upper = np.array([38, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def detect_balls(mask: np.ndarray, min_r: int, max_r: int, min_circ: float):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    balls = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < math.pi * min_r * min_r:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < min_circ:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if radius < min_r or radius > max_r:
            continue
        balls.append((int(cx), int(cy), int(radius)))
    return balls


# ---------------------------------------------------------------------------
# Parabola fitting
# ---------------------------------------------------------------------------

def fit_parabola(points):
    """Fit y = ax^2 + bx + c.  Returns (a,b,c) or None."""
    if len(points) < 3:
        return None
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    return tuple(np.polyfit(xs, ys, 2))


# ---------------------------------------------------------------------------
# In-app graph — pure tkinter Canvas, no matplotlib needed
# ---------------------------------------------------------------------------

class GraphCanvas(tk.Canvas):
    """Draws parabolas + sample points on a tkinter Canvas."""

    MARGIN_L = 50
    MARGIN_R = 20
    MARGIN_T = 20
    MARGIN_B = 40

    def __init__(self, master, **kw):
        kw.setdefault("bg", "white")
        kw.setdefault("highlightthickness", 0)
        super().__init__(master, **kw)
        self.bind("<Configure>", lambda e: self._last_data and self.plot(*self._last_data))
        self._last_data = None

    def plot(self, ball_points, fitted, num_balls):
        """
        ball_points: list[list[(rx, ry, frame)]]
        fitted:      list[(a,b,c) | None]
        """
        self._last_data = (ball_points, fitted, num_balls)
        self.delete("all")

        W = self.winfo_width()
        H = self.winfo_height()
        if W < 80 or H < 80:
            return

        ml, mr, mt, mb = self.MARGIN_L, self.MARGIN_R, self.MARGIN_T, self.MARGIN_B
        pw = W - ml - mr   # plot width
        ph = H - mt - mb   # plot height

        # Determine data range
        all_x, all_y = [], []
        for bi in range(num_balls):
            for (rx, ry, _) in ball_points[bi]:
                all_x.append(rx)
                all_y.append(ry)
        if not all_x:
            self.create_text(W // 2, H // 2, text="No data yet", fill="#888")
            return

        x_min, x_max = min(all_x) - 0.02, max(all_x) + 0.02
        y_min, y_max = min(all_y) - 0.02, max(all_y) + 0.02

        # Extend range for fitted curves
        for bi in range(num_balls):
            if fitted[bi] is None:
                continue
            a, b, c = fitted[bi]
            for t in np.linspace(x_min, x_max, 200):
                yv = a * t ** 2 + b * t + c
                if yv > y_max:
                    y_max = yv + 0.02
                if yv < y_min:
                    y_min = yv - 0.02

        def to_px(rx, ry):
            sx = ml + (rx - x_min) / max(x_max - x_min, 1e-9) * pw
            sy = mt + (1.0 - (ry - y_min) / max(y_max - y_min, 1e-9)) * ph
            return sx, sy

        # Grid + axes
        self.create_rectangle(ml, mt, ml + pw, mt + ph, outline="#ccc")
        # Tick marks
        for i in range(6):
            frac = i / 5
            xv = x_min + frac * (x_max - x_min)
            yv = y_min + frac * (y_max - y_min)
            sx, _ = to_px(xv, 0)
            _, sy = to_px(0, yv)
            # X ticks
            self.create_line(sx, mt + ph, sx, mt + ph + 5, fill="#888")
            self.create_text(sx, mt + ph + 15, text=f"{xv:.2f}", fill="#666",
                             font=("Helvetica", 8))
            # Y ticks
            self.create_line(ml - 5, sy, ml, sy, fill="#888")
            self.create_text(ml - 8, sy, text=f"{yv:.2f}", fill="#666",
                             font=("Helvetica", 8), anchor="e")

        # Axis labels
        self.create_text(ml + pw // 2, H - 4, text="X (relative)", fill="#444",
                         font=("Helvetica", 9))
        self.create_text(8, mt + ph // 2, text="Y", fill="#444",
                         font=("Helvetica", 9), angle=90)

        # Draw fitted curves
        for bi in range(num_balls):
            if fitted[bi] is None:
                continue
            a, b, c = fitted[bi]
            color = ball_color_hex(bi)
            pts = []
            for t in np.linspace(x_min, x_max, 300):
                yv = a * t ** 2 + b * t + c
                sx, sy = to_px(t, yv)
                pts.extend([sx, sy])
            if len(pts) >= 4:
                self.create_line(*pts, fill=color, width=2, smooth=True)

        # Draw sample points
        for bi in range(num_balls):
            color = ball_color_hex(bi)
            for (rx, ry, _) in ball_points[bi]:
                sx, sy = to_px(rx, ry)
                r = 4
                self.create_oval(sx - r, sy - r, sx + r, sy + r,
                                 fill=color, outline="black", width=1)

        # Legend
        lx, ly = ml + 10, mt + 10
        for bi in range(num_balls):
            if not ball_points[bi] and fitted[bi] is None:
                continue
            color = ball_color_hex(bi)
            self.create_rectangle(lx, ly, lx + 12, ly + 12, fill=color, outline="black")
            self.create_text(lx + 18, ly + 6, text=ball_name(bi), anchor="w",
                             fill="#333", font=("Helvetica", 9))
            ly += 18


# ---------------------------------------------------------------------------
# Main GUI — three tabs: Selector, Graph, Export
# ---------------------------------------------------------------------------

CANVAS_MAX_W = 900
CANVAS_MAX_H = 560


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Yellow Ball Tracker")
        self.resizable(True, True)

        # Video state
        self.cap = None
        self.total_frames = 0
        self.frame_w = 0
        self.frame_h = 0
        self.current_frame_bgr = None
        self.detected_blobs = []
        self.display_scale = 1.0
        self._canvas_offset = (0, 0)

        # Ball count (configurable)
        self.num_balls = 4

        # Ball selection state
        self.ball_points = [[] for _ in range(self.num_balls)]
        self.current_ball_idx = 0

        # Fitted parabolas
        self.fitted = [None] * self.num_balls

        self._build_ui()
        self._bind_keys()

    # ---- rebuild ball UI when count changes --------------------------------

    def _set_num_balls(self, n):
        old = self.num_balls
        self.num_balls = n
        # Grow / shrink point lists
        while len(self.ball_points) < n:
            self.ball_points.append([])
        while len(self.fitted) < n:
            self.fitted.append(None)
        self.ball_points = self.ball_points[:n]
        self.fitted = self.fitted[:n]
        if self.current_ball_idx >= n:
            self.current_ball_idx = n - 1
        self._rebuild_ball_buttons()
        self._update_ball_indicator()
        self._highlight_active_btn()
        self._refresh_tree()
        self._rebind_number_keys()
        if self.cap:
            self._detect_and_draw(int(self.frame_slider.get()))
        self.status_var.set(f"Ball count changed to {n}.")

    def _rebuild_ball_buttons(self):
        for btn in self.ball_btns:
            btn.destroy()
        self.ball_btns.clear()
        for i in range(self.num_balls):
            btn = tk.Button(
                self.ball_btn_frame, text=f"{i+1}", width=3,
                bg=ball_color_hex(i), fg="white",
                font=("Helvetica", 10, "bold"),
                command=lambda idx=i: self._switch_ball(idx),
            )
            btn.pack(side="left", padx=2)
            self.ball_btns.append(btn)

    def _rebind_number_keys(self):
        # Unbind old
        for i in range(10):
            self.unbind(str(i))
        for i in range(self.num_balls):
            self.bind(str(i + 1), lambda e, idx=i: self._switch_ball(idx))

    # ---- build UI ---------------------------------------------------------

    def _build_ui(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        # --- Tab 1: Selector -----------------------------------------------
        self.sel_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.sel_tab, text="  Selector  ")

        top = ttk.Frame(self.sel_tab)
        top.pack(fill="x", padx=8, pady=4)

        tk.Button(top, text="Open Video...", command=self._open_video).pack(side="left", padx=4)

        tk.Label(top, text="  Start frame:").pack(side="left")
        self.start_frame_var = tk.IntVar(value=0)
        tk.Spinbox(top, from_=0, to=9_999_999,
                   textvariable=self.start_frame_var, width=8).pack(side="left", padx=2)
        tk.Button(top, text="Go", command=self._goto_start_frame).pack(side="left", padx=2)

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)

        tk.Label(top, text="Balls:").pack(side="left")
        self.num_balls_var = tk.IntVar(value=self.num_balls)
        num_spin = tk.Spinbox(top, from_=1, to=20, textvariable=self.num_balls_var, width=4,
                              command=self._on_num_balls_change)
        num_spin.pack(side="left", padx=2)
        num_spin.bind("<Return>", lambda e: self._on_num_balls_change())

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)

        # Detection params
        tk.Label(top, text="MinR:").pack(side="left")
        self.min_r_var = tk.IntVar(value=5)
        tk.Spinbox(top, from_=1, to=1000, textvariable=self.min_r_var, width=5).pack(side="left")
        tk.Label(top, text=" MaxR:").pack(side="left")
        self.max_r_var = tk.IntVar(value=300)
        tk.Spinbox(top, from_=1, to=5000, textvariable=self.max_r_var, width=5).pack(side="left")
        tk.Label(top, text=" Circ:").pack(side="left")
        self.circ_var = tk.DoubleVar(value=0.6)
        tk.Spinbox(top, from_=0.0, to=1.0, increment=0.05, textvariable=self.circ_var,
                   width=5).pack(side="left")

        # Middle: canvas + side panel
        mid = ttk.Frame(self.sel_tab)
        mid.pack(fill="both", expand=True, padx=8)

        self.canvas = tk.Canvas(mid, bg="black", width=CANVAS_MAX_W, height=CANVAS_MAX_H,
                                cursor="crosshair")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self._photo = None

        # Side panel
        side = ttk.Frame(mid, width=260)
        side.pack(side="right", fill="y", padx=(8, 0))
        side.pack_propagate(False)

        tk.Label(side, text="Selecting:", font=("Helvetica", 11, "bold")).pack(anchor="w")
        self.ball_label_var = tk.StringVar(value=ball_name(0))
        self.ball_label = tk.Label(side, textvariable=self.ball_label_var,
                                   font=("Helvetica", 11), fg=ball_color_hex(0))
        self.ball_label.pack(anchor="w")

        # Ball switch buttons
        self.ball_btn_frame = ttk.Frame(side)
        self.ball_btn_frame.pack(fill="x", pady=4)
        self.ball_btns = []
        for i in range(self.num_balls):
            btn = tk.Button(
                self.ball_btn_frame, text=f"{i+1}", width=3,
                bg=ball_color_hex(i), fg="white",
                font=("Helvetica", 10, "bold"),
                command=lambda idx=i: self._switch_ball(idx),
            )
            btn.pack(side="left", padx=2)
            self.ball_btns.append(btn)
        self._highlight_active_btn()

        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=6)

        # Points table
        tk.Label(side, text="Collected points:", font=("Helvetica", 10, "bold")).pack(anchor="w")
        tree_frame = ttk.Frame(side)
        tree_frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(tree_frame, columns=("ball", "frame", "x", "y"),
                                 show="headings", height=14)
        self.tree.heading("ball", text="Ball")
        self.tree.heading("frame", text="Frame")
        self.tree.heading("x", text="X")
        self.tree.heading("y", text="Y")
        self.tree.column("ball", width=50, anchor="center")
        self.tree.column("frame", width=50, anchor="center")
        self.tree.column("x", width=60, anchor="center")
        self.tree.column("y", width=60, anchor="center")

        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")

        btn_row = ttk.Frame(side)
        btn_row.pack(fill="x", pady=4)
        tk.Button(btn_row, text="Undo last", command=self._undo_last).pack(side="left", padx=2)
        tk.Button(btn_row, text="Clear ball", command=self._clear_current_ball).pack(
            side="left", padx=2)

        # Frame slider
        nav = ttk.Frame(self.sel_tab)
        nav.pack(fill="x", padx=8, pady=(0, 4))
        self.frame_slider = tk.Scale(nav, from_=0, to=0, orient="horizontal",
                                     command=self._on_slider)
        self.frame_slider.pack(fill="x", side="left", expand=True)
        self.frame_info_var = tk.StringVar(value="No video loaded")
        tk.Label(nav, textvariable=self.frame_info_var, width=30).pack(side="right")

        # Status
        help_frame = ttk.Frame(self.sel_tab)
        help_frame.pack(fill="x", padx=8, pady=(0, 6))
        self.status_var = tk.StringVar(
            value="[Left/Right] navigate  |  [Click] select blob  |  "
                  "[1-9] switch ball  |  [Enter] confirm & next  |  "
                  "[Z] undo  |  [F] fit  |  [G] graph"
        )
        tk.Label(help_frame, textvariable=self.status_var, anchor="w",
                 fg="#555555", font=("Helvetica", 9)).pack(fill="x")

        # --- Tab 2: Graph --------------------------------------------------
        self.graph_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.graph_tab, text="  Graph  ")
        self.graph_canvas = GraphCanvas(self.graph_tab, width=800, height=500)
        self.graph_canvas.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Tab 3: Parabolas (text) ---------------------------------------
        self.para_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.para_tab, text="  Parabolas  ")
        self.para_text = tk.Text(self.para_tab, height=25, width=80,
                                 font=("Courier", 11), state="disabled")
        self.para_text.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Tab 4: Export -------------------------------------------------
        self.exp_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.exp_tab, text="  Export  ")
        self._build_export_tab()

    def _build_export_tab(self):
        pad = dict(padx=10, pady=6)
        info = ttk.Frame(self.exp_tab)
        info.pack(fill="x", padx=10, pady=10)

        tk.Label(info, text="Export fitted parabola functions",
                 font=("Helvetica", 13, "bold")).pack(anchor="w")
        tk.Label(info, text="Exports coefficients (a, b, c) for  y = ax\u00b2 + bx + c\n"
                            "Coordinates: bottom-left = (0,0), top-right = (1,1)",
                 fg="#555", justify="left").pack(anchor="w", pady=(4, 8))

        ttk.Separator(info, orient="horizontal").pack(fill="x", pady=4)

        # Preview of current fits
        self.exp_preview = tk.Text(info, height=12, width=70, font=("Courier", 10),
                                   state="disabled", bg="#f8f8f8")
        self.exp_preview.pack(fill="x", pady=6)

        btn_row = ttk.Frame(info)
        btn_row.pack(fill="x", pady=4)

        tk.Button(btn_row, text="Export as CSV...", width=18,
                  command=self._export_csv).pack(side="left", padx=4)
        tk.Button(btn_row, text="Export as TXT...", width=18,
                  command=self._export_txt).pack(side="left", padx=4)
        tk.Button(btn_row, text="Copy to clipboard", width=18,
                  command=self._export_clipboard).pack(side="left", padx=4)

        self.exp_status_var = tk.StringVar(value="Fit parabolas first (press F in Selector).")
        tk.Label(info, textvariable=self.exp_status_var, fg="#555").pack(anchor="w", pady=4)

    # ---- key bindings -----------------------------------------------------

    def _bind_keys(self):
        self.bind("<Left>", lambda e: self._step_frame(-1))
        self.bind("<Right>", lambda e: self._step_frame(1))
        self.bind("<Shift-Left>", lambda e: self._step_frame(-10))
        self.bind("<Shift-Right>", lambda e: self._step_frame(10))
        self.bind("<Return>", lambda e: self._confirm_ball())
        self.bind("<f>", lambda e: self._fit_parabolas())
        self.bind("<F>", lambda e: self._fit_parabolas())
        self.bind("<g>", lambda e: self._show_graph())
        self.bind("<G>", lambda e: self._show_graph())
        self.bind("<z>", lambda e: self._undo_last())
        self.bind("<Z>", lambda e: self._undo_last())
        for i in range(min(self.num_balls, 9)):
            self.bind(str(i + 1), lambda e, idx=i: self._switch_ball(idx))

    # ---- num balls change -------------------------------------------------

    def _on_num_balls_change(self):
        try:
            n = self.num_balls_var.get()
        except tk.TclError:
            return
        n = max(1, min(n, 20))
        if n != self.num_balls:
            self._set_num_balls(n)

    # ---- video loading ----------------------------------------------------

    def _open_video(self):
        path = filedialog.askopenfilename(
            title="Select input video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv"),
                       ("All files", "*.*")])
        if not path:
            return
        self._load_video(path)

    def _load_video(self, path):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open video:\n{path}")
            return
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_slider.config(from_=0, to=max(self.total_frames - 1, 0))

        # Reset selection state
        n = self.num_balls
        self.ball_points = [[] for _ in range(n)]
        self.current_ball_idx = 0
        self.fitted = [None] * n
        self._update_ball_indicator()
        self._highlight_active_btn()
        self._refresh_tree()
        self._goto_start_frame()

    def _goto_start_frame(self):
        sf = self.start_frame_var.get()
        self.frame_slider.set(sf)
        self._seek_and_show(sf)

    # ---- frame navigation -------------------------------------------------

    def _on_slider(self, val):
        self._seek_and_show(int(float(val)))

    def _step_frame(self, delta):
        if not self.cap:
            return
        cur = self.frame_slider.get()
        new = max(0, min(cur + delta, self.total_frames - 1))
        self.frame_slider.set(new)
        self._seek_and_show(new)

    def _seek_and_show(self, frame_idx):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_frame_bgr = frame
        self._detect_and_draw(frame_idx)

    # ---- detection + display ---------------------------------------------

    def _detect_and_draw(self, frame_idx):
        frame = self.current_frame_bgr
        if frame is None:
            return

        mask = build_yellow_mask(frame)
        self.detected_blobs = detect_balls(
            mask, self.min_r_var.get(), self.max_r_var.get(), self.circ_var.get()
        )

        disp = frame.copy()

        # Draw detected blobs
        for i, (cx, cy, r) in enumerate(self.detected_blobs):
            cv2.circle(disp, (cx, cy), r + 3, (0, 255, 255), 2)
            cv2.putText(disp, str(i), (cx - 6, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(disp, str(i), (cx - 6, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw already-selected points
        for bi in range(self.num_balls):
            color = ball_color_bgr(bi)
            for (rx, ry, fidx) in self.ball_points[bi]:
                if fidx == frame_idx:
                    px = int(rx * self.frame_w)
                    py = int((1.0 - ry) * self.frame_h)
                    cv2.drawMarker(disp, (px, py), color, cv2.MARKER_CROSS, 20, 2)

        # Scale to fit canvas
        cw = self.canvas.winfo_width() or CANVAS_MAX_W
        ch = self.canvas.winfo_height() or CANVAS_MAX_H
        scale = min(cw / self.frame_w, ch / self.frame_h, 1.0)
        self.display_scale = scale
        new_w = int(self.frame_w * scale)
        new_h = int(self.frame_h * scale)
        resized = cv2.resize(disp, (new_w, new_h), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        ox = (cw - new_w) // 2
        oy = (ch - new_h) // 2
        self._canvas_offset = (ox, oy)
        self.canvas.create_image(ox, oy, anchor="nw", image=self._photo)

        n = len(self.detected_blobs)
        self.frame_info_var.set(
            f"Frame {frame_idx} / {self.total_frames}  |  "
            f"{n} blob{'s' if n != 1 else ''} detected"
        )

    # ---- canvas click → select blob --------------------------------------

    def _on_canvas_click(self, event):
        if not self.cap or self.current_frame_bgr is None:
            return

        ox, oy = self._canvas_offset
        vx = (event.x - ox) / self.display_scale
        vy = (event.y - oy) / self.display_scale

        best_dist = float("inf")
        best_blob = None
        for (cx, cy, r) in self.detected_blobs:
            d = ((vx - cx) ** 2 + (vy - cy) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_blob = (cx, cy, r)

        if best_blob is None:
            return

        cx, cy, r = best_blob
        if best_dist > r * 3 + 30:
            self.status_var.set("Click closer to a detected blob.")
            return

        rx = cx / self.frame_w
        ry = (self.frame_h - cy) / self.frame_h
        frame_idx = int(self.frame_slider.get())

        self.ball_points[self.current_ball_idx].append((rx, ry, frame_idx))
        self._refresh_tree()
        self._detect_and_draw(frame_idx)

        n = len(self.ball_points[self.current_ball_idx])
        self.status_var.set(
            f"Added point #{n} to {ball_name(self.current_ball_idx)}: "
            f"({rx:.4f}, {ry:.4f}) at frame {frame_idx}"
        )

    # ---- ball workflow ----------------------------------------------------

    def _confirm_ball(self):
        n = len(self.ball_points[self.current_ball_idx])
        if n < 3:
            self.status_var.set(
                f"Need at least 3 points for {ball_name(self.current_ball_idx)} "
                f"(have {n}). Select more frames."
            )
            return
        next_idx = (self.current_ball_idx + 1) % self.num_balls
        self._switch_ball(next_idx)

        all_ready = all(len(self.ball_points[i]) >= 3 for i in range(self.num_balls))
        if all_ready:
            self.status_var.set(
                f"All balls have 3+ points! Press [F] to fit. "
                f"Switched to {ball_name(self.current_ball_idx)}."
            )
            self._fit_parabolas()
        else:
            self.status_var.set(
                f"Confirmed! Now selecting {ball_name(self.current_ball_idx)}."
            )

    def _switch_ball(self, idx):
        if idx < 0 or idx >= self.num_balls:
            return
        self.current_ball_idx = idx
        self._update_ball_indicator()
        self._highlight_active_btn()
        frame_idx = int(self.frame_slider.get())
        if self.cap:
            self._detect_and_draw(frame_idx)
        n = len(self.ball_points[idx])
        self.status_var.set(
            f"Switched to {ball_name(idx)}  ({n} point{'s' if n != 1 else ''})"
        )

    def _highlight_active_btn(self):
        for i, btn in enumerate(self.ball_btns):
            if i == self.current_ball_idx:
                btn.config(relief="sunken", bd=3)
            else:
                btn.config(relief="raised", bd=1)

    def _update_ball_indicator(self):
        idx = self.current_ball_idx
        self.ball_label_var.set(ball_name(idx))
        self.ball_label.config(fg=ball_color_hex(idx))

    def _undo_last(self):
        pts = self.ball_points[self.current_ball_idx]
        if pts:
            pts.pop()
            self._refresh_tree()
            frame_idx = int(self.frame_slider.get())
            self._detect_and_draw(frame_idx)
            self.status_var.set(
                f"Removed last point from {ball_name(self.current_ball_idx)}. "
                f"{len(pts)} points remain."
            )
        else:
            self.status_var.set("No points to undo.")

    def _clear_current_ball(self):
        self.ball_points[self.current_ball_idx].clear()
        self._refresh_tree()
        frame_idx = int(self.frame_slider.get())
        self._detect_and_draw(frame_idx)
        self.status_var.set(f"Cleared all points for {ball_name(self.current_ball_idx)}.")

    # ---- table ------------------------------------------------------------

    def _refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        for bi in range(self.num_balls):
            for (rx, ry, fidx) in self.ball_points[bi]:
                self.tree.insert("", "end",
                                 values=(bi + 1, fidx, f"{rx:.4f}", f"{ry:.4f}"))

    # ---- parabola fitting -------------------------------------------------

    def _fit_parabolas(self):
        lines = []
        lines.append("=" * 60)
        lines.append("  PARABOLA FITS   y = ax\u00b2 + bx + c")
        lines.append("  Coords: bottom-left = (0,0), top-right = (1,1)")
        lines.append("=" * 60)
        lines.append("")

        any_fit = False
        for bi in range(self.num_balls):
            pts_2d = [(rx, ry) for (rx, ry, _) in self.ball_points[bi]]
            if len(pts_2d) < 3:
                self.fitted[bi] = None
                lines.append(f"  {ball_name(bi)}:  not enough points ({len(pts_2d)}/3 min)")
            else:
                coeffs = fit_parabola(pts_2d)
                self.fitted[bi] = coeffs
                a, b, c = coeffs
                lines.append(f"  {ball_name(bi)}:")
                lines.append(f"    f(x) = {a:.5f}x\u00b2 + {b:.5f}x + {c:.5f}")
                xs = np.array([p[0] for p in pts_2d])
                ys = np.array([p[1] for p in pts_2d])
                predicted = a * xs ** 2 + b * xs + c
                residuals = ys - predicted
                mae = np.mean(np.abs(residuals))
                rmse = np.sqrt(np.mean(residuals ** 2))
                lines.append(f"    Points: {len(pts_2d)}   MAE: {mae:.6f}   RMSE: {rmse:.6f}")
                any_fit = True
            lines.append("")

        # Pairwise comparison
        fitted_indices = [i for i in range(self.num_balls) if self.fitted[i] is not None]
        if len(fitted_indices) >= 2:
            lines.append("-" * 60)
            lines.append("  PAIRWISE COMPARISON (MAE over common domain)")
            lines.append("-" * 60)
            lines.append("")
            x_eval = np.linspace(0, 1, 5000)
            for i in range(len(fitted_indices)):
                for j in range(i + 1, len(fitted_indices)):
                    bi, bj = fitted_indices[i], fitted_indices[j]
                    ai, abi, ci = self.fitted[bi]
                    aj, abj, cj = self.fitted[bj]
                    yi = ai * x_eval ** 2 + abi * x_eval + ci
                    yj = aj * x_eval ** 2 + abj * x_eval + cj
                    valid = (yi > 0) & (yj > 0)
                    if valid.any():
                        mae = np.mean(np.abs(yi[valid] - yj[valid]))
                        lines.append(f"  {ball_name(bi)} vs {ball_name(bj)}:  MAE = {mae:.6f}")
                    else:
                        lines.append(f"  {ball_name(bi)} vs {ball_name(bj)}:  no overlap")
            lines.append("")
            all_ys = []
            for idx in fitted_indices:
                a, b, c = self.fitted[idx]
                all_ys.append(a * x_eval ** 2 + b * x_eval + c)
            y_mean = np.mean(all_ys, axis=0)
            maes = []
            for y in all_ys:
                valid = y_mean > 0
                if valid.any():
                    maes.append(np.mean(np.abs(y[valid] - y_mean[valid])))
            if maes:
                lines.append(f"  Average MAE from mean trajectory: {np.mean(maes):.6f}")
            lines.append("")

        text = "\n".join(lines)

        # Update Parabolas tab
        self.para_text.config(state="normal")
        self.para_text.delete("1.0", "end")
        self.para_text.insert("1.0", text)
        self.para_text.config(state="disabled")

        # Update Export preview
        self._update_export_preview()

        # Update graph
        self.graph_canvas.plot(self.ball_points, self.fitted, self.num_balls)

        if any_fit:
            self.notebook.select(self.para_tab)
            self.status_var.set("Parabolas fitted! See Parabolas / Graph / Export tabs.")
        else:
            self.status_var.set("Not enough points to fit. Need >= 3 per ball.")

    # ---- graph ------------------------------------------------------------

    def _show_graph(self):
        self.graph_canvas.plot(self.ball_points, self.fitted, self.num_balls)
        self.notebook.select(self.graph_tab)

    # ---- export tab -------------------------------------------------------

    def _build_function_text(self):
        """Build a human-readable string of all fitted functions."""
        lines = []
        lines.append("# Yellow Ball Tracker — Fitted Parabolas")
        lines.append("# y = ax\u00b2 + bx + c   (origin at bottom-left)")
        lines.append("")
        any_fit = False
        for bi in range(self.num_balls):
            if self.fitted[bi] is None:
                continue
            a, b, c = self.fitted[bi]
            lines.append(f"{ball_name(bi)}:")
            lines.append(f"  f(x) = {a:.6f}x\u00b2 + {b:.6f}x + {c:.6f}")
            lines.append(f"  a = {a:.6f}")
            lines.append(f"  b = {b:.6f}")
            lines.append(f"  c = {c:.6f}")
            lines.append("")
            any_fit = True
        if not any_fit:
            lines.append("(no fitted functions yet)")
        return "\n".join(lines)

    def _update_export_preview(self):
        text = self._build_function_text()
        self.exp_preview.config(state="normal")
        self.exp_preview.delete("1.0", "end")
        self.exp_preview.insert("1.0", text)
        self.exp_preview.config(state="disabled")
        has = any(f is not None for f in self.fitted)
        self.exp_status_var.set("Ready to export." if has else
                                "Fit parabolas first (press F in Selector).")

    def _export_csv(self):
        if not any(f is not None for f in self.fitted):
            messagebox.showwarning("Nothing to export", "Fit parabolas first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export functions as CSV", defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ball", "a", "b", "c"])
            for bi in range(self.num_balls):
                if self.fitted[bi]:
                    a, b, c = self.fitted[bi]
                    w.writerow([bi + 1, f"{a:.6f}", f"{b:.6f}", f"{c:.6f}"])
        self.exp_status_var.set(f"Exported to {path}")

    def _export_txt(self):
        if not any(f is not None for f in self.fitted):
            messagebox.showwarning("Nothing to export", "Fit parabolas first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export functions as TXT", defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        with open(path, "w") as f:
            f.write(self._build_function_text())
        self.exp_status_var.set(f"Exported to {path}")

    def _export_clipboard(self):
        if not any(f is not None for f in self.fitted):
            messagebox.showwarning("Nothing to export", "Fit parabolas first.")
            return
        text = self._build_function_text()
        self.clipboard_clear()
        self.clipboard_append(text)
        self.exp_status_var.set("Copied to clipboard!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = App()
    app.mainloop()
