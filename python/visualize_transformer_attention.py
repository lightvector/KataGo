#!/usr/bin/env python3
"""
Visualize transformer attention weights from a KataGo model checkpoint on SGF positions.

Loads a model checkpoint and an SGF file, displays the board with stones, and lets the
user click on a position to see the attention pattern (as softmax %) from that position
to all others, for a selected layer and head.

Usage:
    python visualize_transformer_attention.py -checkpoint /path/to/checkpoint.ckpt -sgf /path/to/game.sgf
"""

import argparse
import math
import sys
import tkinter as tk
from tkinter import ttk

import numpy as np
import torch

sys.path.append(".")
from katago.game.board import Board
from katago.game.gamestate import GameState
from katago.game.features import Features
from katago.game.data import load_sgf_moves_exn
from katago.train.load_model import load_model
from katago.train.model_pytorch import ExtraOutputs


# ---------------------------------------------------------------------------
# Color mapping - same style as visualize_gab_templates.py
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
# SGF loading
# ---------------------------------------------------------------------------

def load_sgf_to_game(sgf_path):
    """Load an SGF file and return (game_state, all_moves) where all_moves is the full
    list of (pla, loc) moves. The game_state is at move 0 (setup stones applied)."""
    metadata, setup, moves, rules = load_sgf_moves_exn(sgf_path)
    board_size = metadata.size

    if rules is None:
        rules = GameState.RULES_TT.copy()
    else:
        # Ensure all required keys exist
        for key, val in GameState.RULES_TT.items():
            if key not in rules:
                rules[key] = val

    gs = GameState(board_size, rules)
    for pla, loc in setup:
        gs.board.set_stone(pla, loc)
    # Rebuild boards history after setup
    gs.boards = [gs.board.copy()]

    return gs, moves


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def find_attention_block_names(model):
    """Discover all TransformerAttentionBlock names in the model by inspecting the block list."""
    from katago.train.model_pytorch import TransformerAttentionBlock
    names = []
    for module in model.modules():
        if isinstance(module, TransformerAttentionBlock):
            names.append(module.name)
    return names


def get_input_features_no_history(model, game_state):
    """Compute input features with history suppressed (no previous moves or boards)."""
    features = Features(model.config, model.pos_len)
    pos_len = features.pos_len
    bin_input_data = np.zeros(shape=[1]+features.bin_input_shape, dtype=np.float32)
    global_input_data = np.zeros(shape=[1]+features.global_input_shape, dtype=np.float32)
    pla = game_state.board.pla
    opp = Board.get_opp(pla)
    bin_input_data = np.transpose(bin_input_data, axes=(0,2,3,1))
    bin_input_data = bin_input_data.reshape([1, pos_len*pos_len, -1])
    features.fill_row_features(
        game_state.board, pla, opp,
        boards=[game_state.board.copy()],
        moves=[],
        move_idx=0,
        rules=game_state.rules,
        bin_input_data=bin_input_data,
        global_input_data=global_input_data,
        idx=0,
    )
    bin_input_data = bin_input_data.reshape([1, pos_len, pos_len, -1])
    bin_input_data = np.transpose(bin_input_data, axes=(0,3,1,2))
    return bin_input_data, global_input_data


POLICY_LABEL = "[Policy]"
OPP_POLICY_LABEL = "[Opponent Reply Policy]"
POLICY_LABELS = [POLICY_LABEL, OPP_POLICY_LABEL]


def run_inference(model, game_state, attn_block_names, suppress_history=False):
    """Run model forward pass requesting attention weights for all transformer blocks.
    Returns (attn_dict, policy_probs, model_predictions) where attn_dict maps
    block_name -> numpy array of shape (H, S, S_key), policy_probs is shape
    (num_policy_outputs, move) with softmax applied, and model_predictions is a dict
    with value/score outputs."""
    requested = [name + ".attn_weights" for name in attn_block_names]
    extra_outputs = ExtraOutputs(requested)

    if suppress_history:
        bin_input_data, global_input_data = get_input_features_no_history(model, game_state)
    else:
        features = Features(model.config, model.pos_len)
        bin_input_data, global_input_data = game_state.get_input_features(features)

    with torch.no_grad():
        model.eval()
        outputs_byheads = model(
            torch.tensor(bin_input_data, dtype=torch.float32, device=model.device),
            torch.tensor(global_input_data, dtype=torch.float32, device=model.device),
            extra_outputs=extra_outputs,
        )
        # policy_logits shape: (N, num_policy_outputs, move)
        policy_logits = outputs_byheads[0][0][0]  # batch 0
        policy_probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()

        # Extract value and score predictions via postprocess
        postprocessed = model.postprocess_output(outputs_byheads)
        main_output = postprocessed[0]
        # value_logits: (N, 3) -> softmax -> (win, loss, noresult)
        value_probs = torch.softmax(main_output[1][0], dim=-1).cpu().numpy()
        scoremean = main_output[8][0].item()
        lead = main_output[10][0].item()

    attn_dict = {}
    for name in attn_block_names:
        key = name + ".attn_weights"
        if key in extra_outputs.returned:
            # Shape: (B, H, S_query, S_key) -> take batch 0 -> (H, S_q, S_k)
            attn_dict[name] = extra_outputs.returned[key][0].cpu().numpy()

    model_predictions = {
        "win": value_probs[0],
        "loss": value_probs[1],
        "noresult": value_probs[2],
        "scoremean": scoremean,
        "lead": lead,
    }
    return attn_dict, policy_probs, model_predictions


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class AttentionVisualizer:
    def __init__(self, model, game_state, all_moves, attn_block_names):
        self.model = model
        self.game_state = game_state
        self.all_moves = all_moves
        self.attn_block_names = attn_block_names
        self.board_size = game_state.board.x_size
        self.pos_len = model.pos_len

        # Determine num_heads and num_kv_heads from model config
        self.num_heads = model.config.get("transformer_heads", 6)
        self.num_kv_heads = model.config.get("transformer_kv_heads", self.num_heads)

        self.current_move = 0
        self.total_moves = len(all_moves)
        self.selected_pos = None  # (row, col) on the board (0-indexed from top)

        # Cache: attention_cache[(move_number, suppress_history)] = (attn_dict, policy_probs, predictions)
        # where attn_dict maps block_name -> (H, S_q, S_k), policy_probs is (num_policy_outputs, move),
        # and predictions is a dict with win/loss/noresult/scoremean/lead
        self.attention_cache = {}

        self.cell_size = 32
        self.margin = 28

        self.root = tk.Tk()
        self.root.title("Transformer Attention Visualizer")

        # --- Main frame ---
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Board canvas ---
        board_px = self.pos_len * self.cell_size + 2 * self.margin
        self.canvas = tk.Canvas(main_frame, width=board_px, height=board_px,
                                bg="#c89664", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, padx=4, pady=4)
        self.canvas.bind("<Button-1>", self.on_board_click)

        # --- Controls panel ---
        ctrl = tk.Frame(main_frame)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        # Move navigation
        tk.Label(ctrl, text="Move:", font=("TkDefaultFont", 11)).pack(anchor=tk.W)
        nav_frame = tk.Frame(ctrl)
        nav_frame.pack(anchor=tk.W, fill=tk.X, pady=(0, 4))
        tk.Button(nav_frame, text="<<", width=3, command=lambda: self.change_move(-10)).pack(side=tk.LEFT)
        tk.Button(nav_frame, text="<", width=3, command=lambda: self.change_move(-1)).pack(side=tk.LEFT)
        self.move_label = tk.Label(nav_frame, text="0", font=("TkFixedFont", 11), width=6)
        self.move_label.pack(side=tk.LEFT, padx=4)
        tk.Button(nav_frame, text=">", width=3, command=lambda: self.change_move(1)).pack(side=tk.LEFT)
        tk.Button(nav_frame, text=">>", width=3, command=lambda: self.change_move(10)).pack(side=tk.LEFT)

        self.move_total_label = tk.Label(ctrl, text=f"/ {self.total_moves}", font=("TkFixedFont", 9))
        self.move_total_label.pack(anchor=tk.W)

        tk.Label(ctrl, text="", font=("TkDefaultFont", 2)).pack()  # spacer

        # Layer selector (attention layers + policy outputs)
        tk.Label(ctrl, text="Attention layer:", font=("TkDefaultFont", 11)).pack(anchor=tk.W)
        self.layer_var = tk.StringVar()
        self.layer_values = self.attn_block_names + POLICY_LABELS
        self.layer_combo = ttk.Combobox(ctrl, textvariable=self.layer_var,
                                        values=self.layer_values, state="readonly", width=30)
        if self.layer_values:
            self.layer_combo.current(0)
        self.layer_combo.pack(anchor=tk.W, pady=(0, 4))
        self.layer_combo.bind("<<ComboboxSelected>>", self._on_layer_change)

        tk.Label(ctrl, text="", font=("TkDefaultFont", 2)).pack()  # spacer

        # Head selector
        tk.Label(ctrl, text="Head (query group):", font=("TkDefaultFont", 11)).pack(anchor=tk.W)
        head_frame = tk.Frame(ctrl)
        head_frame.pack(anchor=tk.W, fill=tk.X)
        self.head_var = tk.IntVar(value=0)
        self.head_slider = tk.Scale(
            head_frame, from_=0, to=max(0, self.num_heads - 1),
            orient=tk.HORIZONTAL, variable=self.head_var,
            command=self._on_head_change, length=200,
        )
        self.head_slider.pack(side=tk.LEFT)

        tk.Label(ctrl, text="", font=("TkDefaultFont", 2)).pack()  # spacer

        # Suppress history toggle
        self.suppress_history_var = tk.BooleanVar(value=False)
        self.suppress_history_btn = tk.Checkbutton(
            ctrl, text="Suppress history input", variable=self.suppress_history_var,
            command=self._on_suppress_history_change, font=("TkDefaultFont", 10),
        )
        self.suppress_history_btn.pack(anchor=tk.W)

        tk.Label(ctrl, text="", font=("TkDefaultFont", 2)).pack()  # spacer

        # Info display
        self.info_label = tk.Label(ctrl, text="Click a board position\n(after changing moves, click to compute)",
                                   font=("TkDefaultFont", 10), justify=tk.LEFT, anchor=tk.NW)
        self.info_label.pack(anchor=tk.W, pady=(8, 0))

        self.stats_label = tk.Label(ctrl, text="", font=("TkFixedFont", 9),
                                    justify=tk.LEFT, anchor=tk.NW)
        self.stats_label.pack(anchor=tk.W, pady=(4, 0))

        # Model predictions display
        self.predictions_label = tk.Label(ctrl, text="", font=("TkFixedFont", 9),
                                          justify=tk.LEFT, anchor=tk.NW)
        self.predictions_label.pack(anchor=tk.W, pady=(8, 0))

        # Config info
        config_text = (
            f"board_size={self.board_size}\n"
            f"pos_len={self.pos_len}\n"
            f"heads={self.num_heads}, kv_heads={self.num_kv_heads}\n"
            f"attn_layers={len(self.attn_block_names)}"
        )
        tk.Label(ctrl, text=config_text, font=("TkFixedFont", 9),
                 justify=tk.LEFT, anchor=tk.NW, fg="#555555").pack(anchor=tk.W, pady=(16, 0))

        # Key bindings
        self.root.bind("<Left>", lambda e: self.change_move(-1))
        self.root.bind("<Right>", lambda e: self.change_move(1))
        self.root.bind("<Prior>", lambda e: self.change_move(-10))  # Page Up
        self.root.bind("<Next>", lambda e: self.change_move(10))    # Page Down
        self.root.bind("<Escape>", self._on_escape)

        # Apply moves to reach move 0 and draw initial board
        self.draw_board()

    def px_of_col(self, c):
        return self.margin + self.cell_size // 2 + c * self.cell_size

    def py_of_row(self, r):
        return self.margin + self.cell_size // 2 + r * self.cell_size

    def board_coord_of_px(self, px, py):
        c = round((px - self.margin - self.cell_size // 2) / self.cell_size)
        r = round((py - self.margin - self.cell_size // 2) / self.cell_size)
        if 0 <= r < self.board_size and 0 <= c < self.board_size:
            return r, c
        return None

    def change_move(self, delta):
        new_move = max(0, min(self.total_moves, self.current_move + delta))
        if new_move == self.current_move:
            return
        # Apply or undo moves to reach the new position
        while self.current_move < new_move:
            pla, loc = self.all_moves[self.current_move]
            self.game_state.play(pla, loc)
            self.current_move += 1
        while self.current_move > new_move:
            self.game_state.undo()
            self.current_move -= 1

        self.move_label.config(text=str(self.current_move))
        # Clear selection and overlay (cache may be stale for new position)
        self.selected_pos = None
        self.stats_label.config(text="")
        self.predictions_label.config(text="")
        self.info_label.config(text="Click a board position to compute")
        self.draw_board()

    def _on_escape(self, event):
        # Clear the overlay display but keep the cache
        self.selected_pos = None
        self.stats_label.config(text="")
        self.info_label.config(text="Click a board position")
        self.draw_board()

    def _cache_key(self):
        return (self.current_move, self.suppress_history_var.get())

    def _on_layer_change(self, event):
        if self.selected_pos is not None and self._cache_key() in self.attention_cache:
            self.draw_board()

    def _on_head_change(self, _val):
        if self.selected_pos is not None and self._cache_key() in self.attention_cache:
            self.draw_board()

    def _on_suppress_history_change(self):
        # If we have cached data for the new setting, just redraw; otherwise clear display
        if self._cache_key() in self.attention_cache:
            _, _, predictions = self.attention_cache[self._cache_key()]
            self._update_predictions_label(predictions)
            if self.selected_pos is not None:
                self.draw_board()
        else:
            self.selected_pos = None
            self.stats_label.config(text="")
            self.predictions_label.config(text="")
            self.info_label.config(text="Click a board position to compute")
            self.draw_board()

    def _update_predictions_label(self, predictions):
        win = predictions["win"] * 100.0
        loss = predictions["loss"] * 100.0
        noresult = predictions["noresult"] * 100.0
        scoremean = predictions["scoremean"]
        lead = predictions["lead"]
        self.predictions_label.config(text=(
            f"Win:   {win:.1f}%\n"
            f"Loss:  {loss:.1f}%\n"
            f"NoRes: {noresult:.1f}%\n"
            f"Score: {scoremean:+.1f}\n"
            f"Lead:  {lead:+.1f}"
        ))

    def _ensure_attention_computed(self):
        """Compute attention for the current move if not cached."""
        key = self._cache_key()
        if key not in self.attention_cache:
            self.info_label.config(text="Computing attention...")
            self.root.update_idletasks()
            suppress = self.suppress_history_var.get()
            attn_dict, policy_probs, predictions = run_inference(self.model, self.game_state, self.attn_block_names, suppress_history=suppress)
            self.attention_cache[key] = (attn_dict, policy_probs, predictions)
            self.info_label.config(text="")
            self._update_predictions_label(predictions)

    def draw_board(self):
        self.canvas.delete("all")
        n = self.board_size
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

        # Draw attention overlay rectangles (under stones)
        if self.selected_pos is not None and self._cache_key() in self.attention_cache:
            self._draw_overlay_rects()

        # Draw stones on top of overlay rectangles
        self._draw_stones()

        # Draw attention overlay labels and selection marker on top of stones
        if self.selected_pos is not None and self._cache_key() in self.attention_cache:
            self._draw_overlay_labels()

    def _draw_stones(self):
        n = self.board_size
        cs = self.cell_size
        board = self.game_state.board
        stone_rad = int((cs // 2 - 2) * 0.95)

        for r in range(n):
            for c in range(n):
                loc = board.loc(c, r)
                stone = board.board[loc]
                if stone == Board.BLACK or stone == Board.WHITE:
                    x, y = self.px_of_col(c), self.py_of_row(r)
                    fill = "black" if stone == Board.BLACK else "white"
                    outline = "black"
                    self.canvas.create_oval(
                        x - stone_rad, y - stone_rad, x + stone_rad, y + stone_rad,
                        fill=fill, outline=outline, width=1,
                    )

    def _is_policy_mode(self):
        return self.layer_var.get() in POLICY_LABELS

    def _get_overlay_probs(self):
        """Get the overlay probs array for the current selection.
        Returns (probs, attn_weights_or_none) or None.
        For policy modes, attn_weights_or_none is None."""
        r_sel, c_sel = self.selected_pos
        layer_name = self.layer_var.get()
        head_idx = self.head_var.get()

        cached = self.attention_cache.get(self._cache_key())
        if cached is None:
            return None
        attn_dict, policy_probs, _predictions = cached

        seq_len = self.pos_len * self.pos_len

        if layer_name == POLICY_LABEL:
            # Policy output 0: main policy, shape (move,) where move = seq_len + 1 (pass)
            probs = policy_probs[0, :seq_len]
            return probs, None
        elif layer_name == OPP_POLICY_LABEL:
            # Policy output 1: opponent reply policy
            if policy_probs.shape[0] < 2:
                return None
            probs = policy_probs[1, :seq_len]
            return probs, None

        if layer_name not in attn_dict:
            return None

        attn_weights = attn_dict[layer_name]
        num_heads = attn_weights.shape[0]

        if head_idx >= num_heads:
            head_idx = 0
            self.head_var.set(0)

        src_idx = r_sel * self.pos_len + c_sel
        probs = attn_weights[head_idx, src_idx, :seq_len]
        return probs, attn_weights

    def _draw_overlay_rects(self):
        """Draw colored rectangles for attention weights (called before stones)."""
        result = self._get_overlay_probs()
        if result is None:
            return
        probs, _ = result

        n = self.board_size
        cs = self.cell_size
        max_prob = probs.max()
        half = cs * 0.48

        for dst_idx in range(n * n):
            dr = dst_idx // n
            dc = dst_idx % n
            pos_idx = dr * self.pos_len + dc
            if pos_idx >= self.pos_len * self.pos_len:
                continue
            prob = probs[pos_idx]

            color = prob_to_hex(prob, max_prob)
            x = self.px_of_col(dc)
            y = self.py_of_row(dr)

            self.canvas.create_rectangle(
                x - half, y - half, x + half, y + half,
                fill=color, outline="",
            )

    def _draw_overlay_labels(self):
        """Draw percentage labels and selection marker (called after stones)."""
        result = self._get_overlay_probs()
        if result is None:
            return
        probs, attn_weights = result

        r_sel, c_sel = self.selected_pos
        layer_name = self.layer_var.get()
        head_idx = self.head_var.get()
        n = self.board_size
        cs = self.cell_size
        seq_len = self.pos_len * self.pos_len
        board = self.game_state.board
        max_prob = probs.max()
        is_policy = self._is_policy_mode()

        for dst_idx in range(n * n):
            dr = dst_idx // n
            dc = dst_idx % n
            pos_idx = dr * self.pos_len + dc
            if pos_idx >= seq_len:
                continue
            prob = probs[pos_idx]

            pct = prob * 100.0
            if pct >= 0.05:
                if pct >= 10.0:
                    label = f"{pct:.0f}"
                elif pct >= 1.0:
                    label = f"{pct:.1f}"
                else:
                    label = f"{pct:.2f}"
                x = self.px_of_col(dc)
                y = self.py_of_row(dr)
                loc = board.loc(dc, dr)
                text_color = "white" if board.board[loc] == Board.BLACK else "black"
                self.canvas.create_text(x, y, text=label, font=("TkFixedFont", 7), fill=text_color)

        if is_policy:
            # For policy, show pass probability and entropy
            cached = self.attention_cache.get(self._cache_key())
            _, policy_probs, _ = cached
            policy_idx = 0 if layer_name == POLICY_LABEL else 1
            pass_prob = policy_probs[policy_idx, seq_len] * 100.0
            entropy = -np.sum(probs * np.log(probs + 1e-30))
            self.info_label.config(text=f"{layer_name}")
            self.stats_label.config(text=(
                f"Entropy:  {entropy:.2f} (max {math.log(n*n):.2f})\n"
                f"Top %%:   {max_prob*100:.2f}%\n"
                f"Pass %%:  {pass_prob:.2f}%"
            ))
        else:
            # Mark selected position
            x, y = self.px_of_col(c_sel), self.py_of_row(r_sel)
            rad = cs // 2 - 3
            self.canvas.create_oval(x - rad, y - rad, x + rad, y + rad,
                                    outline="blue", width=2)

            # Show attention to registers if any
            src_idx = r_sel * self.pos_len + c_sel
            num_registers = attn_weights.shape[2] - seq_len
            reg_info = ""
            if num_registers > 0:
                reg_probs = attn_weights[head_idx, src_idx, seq_len:]
                reg_total = reg_probs.sum() * 100.0
                reg_info = f"\nAttn to registers: {reg_total:.2f}% ({num_registers} regs)"

            # Update stats
            entropy = -np.sum(probs * np.log(probs + 1e-30))
            col_labels = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
            pos_label = f"{col_labels[c_sel]}{n - r_sel}"
            self.info_label.config(text=f"Source: {pos_label}\nLayer: {layer_name}\nHead: {head_idx}")
            self.stats_label.config(text=(
                f"Softmax entropy: {entropy:.2f} (max {math.log(n*n):.2f})\n"
                f"Top attn %%:     {max_prob*100:.2f}%{reg_info}"
            ))

    def on_board_click(self, event):
        coord = self.board_coord_of_px(event.x, event.y)
        if coord is not None:
            self._ensure_attention_computed()
            self.selected_pos = coord
            self.draw_board()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Visualize transformer attention from a KataGo checkpoint on SGF positions")
    parser.add_argument("-checkpoint", required=True, help="Path to checkpoint .ckpt file")
    parser.add_argument("-sgf", required=True, help="Path to SGF file")
    parser.add_argument("-pos-len", type=int, default=19, help="Position length for the model (default: 19)")
    parser.add_argument("-device", default="cpu", help="Device to run on (default: cpu)")
    args = parser.parse_args()

    print(f"Loading model from: {args.checkpoint}")
    model, _, _ = load_model(args.checkpoint, use_swa=False, device=args.device, pos_len=args.pos_len)
    model.eval()

    attn_block_names = find_attention_block_names(model)
    if not attn_block_names:
        print("ERROR: No TransformerAttentionBlock found in this model.")
        print("This model may not use transformer attention layers.")
        sys.exit(1)
    print(f"Found {len(attn_block_names)} attention layers: {attn_block_names}")

    print(f"Loading SGF: {args.sgf}")
    game_state, all_moves = load_sgf_to_game(args.sgf)
    print(f"Loaded game with {len(all_moves)} moves, board size {game_state.board.x_size}")

    viz = AttentionVisualizer(model, game_state, all_moves, attn_block_names)
    viz.run()


if __name__ == "__main__":
    main()
