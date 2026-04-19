#!/usr/bin/env python3
"""
Visualize GAB (Geometric Attention Bias) templates from a model checkpoint.

Loads the GABTemplateMLP from a checkpoint, computes templates for a 19x19 board,
and displays an interactive board where clicking a position shows the attention
bias pattern (as softmax %) from that position to all others, for the selected
template index.

Usage:
    python visualize_gab_templates.py -checkpoint /path/to/checkpoint.ckpt
"""

import argparse
import math
import sys
import tkinter as tk

import numpy as np
import torch

sys.path.append(".")
import katago.train.load_model


def load_gab_template_mlp(checkpoint_path):
    """Load checkpoint config and GABTemplateMLP weights, return (config, templates tensor)."""
    data = katago.train.load_model.load_checkpoint(checkpoint_path)
    config = data["config"]

    needed_keys = ["gab_num_templates", "gab_num_fourier_features", "gab_mlp_hidden"]
    for k in needed_keys:
        if k not in config:
            raise ValueError(
                f"Checkpoint config missing '{k}'. This model does not appear to use GAB.\n"
                f"Config keys: {sorted(config.keys())}"
            )

    from katago.train.model_pytorch import GABTemplateMLP

    pos_len = config.get("pos_len", 19)
    mlp = GABTemplateMLP(
        gab_num_templates=config["gab_num_templates"],
        gab_num_fourier_features=config["gab_num_fourier_features"],
        gab_mlp_hidden=config["gab_mlp_hidden"],
        pos_len=pos_len,
        activation=config["activation"],
    )

    # Extract GABTemplateMLP weights from the model state dict
    model_sd = katago.train.load_model.load_model_state_dict(data)
    prefix = "gab_template_mlp."
    mlp_sd = {}
    for k, v in model_sd.items():
        if k.startswith(prefix):
            mlp_sd[k[len(prefix):]] = v

    if not mlp_sd:
        raise ValueError(
            "Could not find gab_template_mlp weights in checkpoint.\n"
            f"Available top-level key prefixes: {sorted(set(k.split('.')[0] for k in model_sd.keys()))}"
        )

    mlp.load_state_dict(mlp_sd, strict=False)
    mlp.eval()

    with torch.no_grad():
        seq_len = pos_len * pos_len
        templates = mlp(seq_len)  # (S, S, T)

    return config, pos_len, templates.numpy()


# ---------------------------------------------------------------------------
# Color mapping - same style as humanslnet_gui.py policy visualization
# ---------------------------------------------------------------------------

COLORS = [
    (0.00, np.array([100, 0, 0, 15])),
    (0.35, np.array([184, 0, 0, 255])),
    (0.50, np.array([220, 0, 0, 255])),
    (0.65, np.array([255, 100, 0, 255])),
    (0.85, np.array([205, 220, 60, 255])),
    (0.94, np.array([120, 235, 130, 255])),
    (1.00, np.array([100, 255, 245, 255])),
]


def interpolate_color(points, x):
    for i in range(len(points)):
        x1, c1 = points[i]
        if x < x1:
            if i <= 0:
                return c1.copy()
            x0, c0 = points[i - 1]
            interp = (x - x0) / (x1 - x0)
            return c0 + (c1 - c0) * interp
    return points[-1][1].copy()


def prob_to_hex(prob, max_prob):
    """Map a probability to a hex color string for tkinter, using power-law scaling."""
    if max_prob <= 0:
        return "#646464"
    normalized = prob / max_prob ** 0.7
    normalized = max(0.0, min(1.0, normalized))
    r, g, b, _a = interpolate_color(COLORS, normalized ** 0.25)
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class GABVisualizer:
    def __init__(self, config, pos_len, templates):
        """
        templates: numpy array (S, S, T) where S = pos_len*pos_len
        """
        self.pos_len = pos_len
        self.num_templates = templates.shape[2]
        self.templates = templates  # (S, S, T)
        self.config = config

        self.selected_template = 0
        self.selected_pos = None  # (row, col) on the board
        self.logit_scale = 1.0

        self.cell_size = 32
        self.margin = 28

        self.root = tk.Tk()
        self.root.title("GAB Template Visualizer")

        # --- Main frame ---
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Board canvas ---
        board_px = pos_len * self.cell_size + 2 * self.margin
        self.canvas = tk.Canvas(main_frame, width=board_px, height=board_px,
                                bg="#c89664", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, padx=4, pady=4)
        self.canvas.bind("<Button-1>", self.on_board_click)

        # --- Controls panel ---
        ctrl = tk.Frame(main_frame)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        tk.Label(ctrl, text="Template index:", font=("TkDefaultFont", 11)).pack(anchor=tk.W)

        slider_frame = tk.Frame(ctrl)
        slider_frame.pack(anchor=tk.W, fill=tk.X)
        self.template_var = tk.IntVar(value=0)
        self.template_slider = tk.Scale(
            slider_frame, from_=0, to=self.num_templates - 1,
            orient=tk.HORIZONTAL, variable=self.template_var,
            command=self._on_template_change, length=200,
        )
        self.template_slider.pack(side=tk.LEFT)

        tk.Label(ctrl, text="", font=("TkDefaultFont", 2)).pack()  # spacer

        tk.Label(ctrl, text="Logit scale:", font=("TkDefaultFont", 11)).pack(anchor=tk.W)
        # Scale slider from -40.0 to 40.0, using integer -640..640 mapped to value/16
        # so the resolution is 1/16 = 0.0625
        self.scale_var = tk.IntVar(value=16)  # 16/16 = 1.0
        self.scale_slider = tk.Scale(
            ctrl, from_=-640, to=640,
            orient=tk.HORIZONTAL, variable=self.scale_var,
            command=self._on_scale_change, length=200,
            showvalue=False,
        )
        self.scale_slider.pack(anchor=tk.W, fill=tk.X)
        self.scale_label = tk.Label(ctrl, text="1.0", font=("TkFixedFont", 9))
        self.scale_label.pack(anchor=tk.W)

        tk.Label(ctrl, text="", font=("TkDefaultFont", 2)).pack()  # spacer

        self.info_label = tk.Label(ctrl, text="Click a board position", font=("TkDefaultFont", 10),
                                   justify=tk.LEFT, anchor=tk.NW)
        self.info_label.pack(anchor=tk.W, pady=(8, 0))

        self.stats_label = tk.Label(ctrl, text="", font=("TkFixedFont", 9),
                                    justify=tk.LEFT, anchor=tk.NW)
        self.stats_label.pack(anchor=tk.W, pady=(4, 0))

        config_text = (
            f"pos_len={pos_len}\n"
            f"gab_num_templates={config.get('gab_num_templates')}\n"
            f"gab_num_fourier_features={config.get('gab_num_fourier_features')}\n"
            f"gab_mlp_hidden={config.get('gab_mlp_hidden')}"
        )
        tk.Label(ctrl, text=config_text, font=("TkFixedFont", 9),
                 justify=tk.LEFT, anchor=tk.NW, fg="#555555").pack(anchor=tk.W, pady=(16, 0))

        self.draw_board()

    def px_of_col(self, c):
        return self.margin + self.cell_size // 2 + c * self.cell_size

    def py_of_row(self, r):
        return self.margin + self.cell_size // 2 + r * self.cell_size

    def board_coord_of_px(self, px, py):
        c = round((px - self.margin - self.cell_size // 2) / self.cell_size)
        r = round((py - self.margin - self.cell_size // 2) / self.cell_size)
        if 0 <= r < self.pos_len and 0 <= c < self.pos_len:
            return r, c
        return None

    def draw_board(self):
        self.canvas.delete("all")
        n = self.pos_len
        cs = self.cell_size

        # Grid lines
        for i in range(n):
            x = self.px_of_col(i)
            y = self.py_of_row(i)
            self.canvas.create_line(self.px_of_col(0), y, self.px_of_col(n - 1), y, fill="black")
            self.canvas.create_line(x, self.py_of_row(0), x, self.py_of_row(n - 1), fill="black")

        # Star points
        if n == 19:
            stars = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
        elif n == 13:
            stars = [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)]
        elif n == 9:
            stars = [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
        else:
            stars = []
        for r, c in stars:
            x, y = self.px_of_col(c), self.py_of_row(r)
            rad = 3
            self.canvas.create_oval(x - rad, y - rad, x + rad, y + rad, fill="black")

        # Coordinate labels
        col_labels = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
        for c in range(n):
            lbl = col_labels[c]
            self.canvas.create_text(self.px_of_col(c), self.margin // 2, text=lbl, font=("TkDefaultFont", 8))
            self.canvas.create_text(self.px_of_col(c), self.py_of_row(n - 1) + self.margin // 2, text=lbl, font=("TkDefaultFont", 8))
        for r in range(n):
            lbl = str(n - r)
            self.canvas.create_text(self.margin // 2, self.py_of_row(r), text=lbl, font=("TkDefaultFont", 8))
            self.canvas.create_text(self.px_of_col(n - 1) + self.margin // 2, self.py_of_row(r), text=lbl, font=("TkDefaultFont", 8))

        # Draw overlay if a position is selected
        if self.selected_pos is not None:
            self._draw_overlay()

    def _draw_overlay(self):
        r_sel, c_sel = self.selected_pos
        t = self.selected_template
        n = self.pos_len
        cs = self.cell_size

        src_idx = r_sel * n + c_sel
        # Raw template biases from selected source to all destinations
        raw_biases = self.templates[src_idx, :, t]  # (S,)

        # Scale biases before softmax - in the real model, attention heads take
        # weighted multiples of templates, so this simulates different weight magnitudes
        scaled_biases = raw_biases * self.logit_scale

        # Convert to attention-like probabilities via softmax
        # Shift for numerical stability
        biases = scaled_biases - scaled_biases.max()
        exp_biases = np.exp(biases)
        probs = exp_biases / exp_biases.sum()

        max_prob = probs.max()

        half = cs * 0.45

        for dst_idx in range(n * n):
            dr = dst_idx // n
            dc = dst_idx % n
            prob = probs[dst_idx]

            color = prob_to_hex(prob, max_prob)
            x = self.px_of_col(dc)
            y = self.py_of_row(dr)

            self.canvas.create_rectangle(
                x - half, y - half, x + half, y + half,
                fill=color, outline="",
            )

            pct = prob * 100.0
            if pct >= 0.05:
                if pct >= 10.0:
                    label = f"{pct:.0f}"
                elif pct >= 1.0:
                    label = f"{pct:.1f}"
                else:
                    label = f"{pct:.2f}"
                self.canvas.create_text(x, y, text=label, font=("TkFixedFont", 7), fill="black")

        # Mark selected position
        x, y = self.px_of_col(c_sel), self.py_of_row(r_sel)
        rad = cs // 2 - 3
        self.canvas.create_oval(x - rad, y - rad, x + rad, y + rad,
                                outline="white", width=2)

        # Update stats
        raw_min = raw_biases.min()
        raw_max = raw_biases.max()
        raw_mean = raw_biases.mean()
        entropy = -np.sum(probs * np.log(probs + 1e-30))
        col_labels = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
        pos_label = f"{col_labels[c_sel]}{n - r_sel}"
        self.info_label.config(text=f"Source: {pos_label}   Template: {t}")
        self.stats_label.config(text=(
            f"Raw bias range: [{raw_min:.3f}, {raw_max:.3f}]\n"
            f"Raw bias mean:  {raw_mean:.3f}\n"
            f"Logit scale:     {self.logit_scale:.2f}\n"
            f"Softmax entropy: {entropy:.2f} (max {math.log(n*n):.2f})\n"
            f"Top softmax %%:  {max_prob*100:.2f}%"
        ))

    def on_board_click(self, event):
        coord = self.board_coord_of_px(event.x, event.y)
        if coord is not None:
            self.selected_pos = coord
            self.draw_board()

    def _on_template_change(self, _val):
        self.selected_template = self.template_var.get()
        if self.selected_pos is not None:
            self.draw_board()

    def _on_scale_change(self, _val):
        self.logit_scale = self.scale_var.get() / 16.0
        self.scale_label.config(text=f"{self.logit_scale:.2f}")
        if self.selected_pos is not None:
            self.draw_board()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Visualize GAB templates from a KataGo checkpoint")
    parser.add_argument("-checkpoint", required=True, help="Path to checkpoint .ckpt file")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    config, pos_len, templates = load_gab_template_mlp(args.checkpoint)
    print(f"Loaded {templates.shape[2]} templates for {pos_len}x{pos_len} board")

    viz = GABVisualizer(config, pos_len, templates)
    viz.run()


if __name__ == "__main__":
    main()
