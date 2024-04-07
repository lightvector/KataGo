import wx
import subprocess
import json
import sys
import os
from threading import Thread
import atexit
import datetime

from gamestate import GameState
from board import Board
from sgfmetadata import SGFMetadata

from sgfmill import sgf, sgf_moves

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def interpolateColor(points,x):
    for i in range(len(points)):
        x1,c1 = points[i]
        if x < x1:
            if i <= 0:
                return c1
            x0,c0 = points[i-1]
            interp = (x-x0)/(x1-x0)
            return c0 + (c1-c0)*interp
    return points[-1][1]

POLICY_COLORS = [
  (0.00, np.array([100,0,0,15])),
  (0.35, np.array([184,0,0,255])),
  (0.50, np.array([220,0,0,255])),
  (0.65, np.array([255,100,0,255])),
  (0.85, np.array([205,220,60,255])),
  (0.94, np.array([120,235,130,255])),
  (1.00, np.array([100,255,245,255])),
]

def policy_color(prob):
    r,g,b,a = interpolateColor(POLICY_COLORS,prob**0.25)
    return (round(r), round(g), round(b), round(a))

def load_sgf_game_state(file_path):
    with open(file_path, 'rb') as f:
        game = sgf.Sgf_game.from_bytes(f.read())

    size = game.get_size()
    if size < 9 or size > 19:
        raise ValueError("Board size must be between 9 and 19 inclusive.")

    board, plays = sgf_moves.get_setup_and_moves(game)

    moves = []
    for y in range(size):
        for x in range(size):
            color = board.get(y, x)
            if color is not None:
                moves.append((x, size - 1 - y, (Board.BLACK if color == "b" else Board.WHITE)))

    for color, move in plays:
        if move is not None:
            y, x = move
            moves.append((x, size - 1 - y, (Board.BLACK if color == "b" else Board.WHITE)))

    game_state = GameState(size, GameState.RULES_JAPANESE)
    for (x,y,color) in moves:
        game_state.play(color, game_state.board.loc(x,y))

    return game_state

class GoBoard(wx.Panel):
    def __init__(self, parent, game_state, cell_size=30, margin=30):
        super().__init__(parent)
        self.game_state = game_state
        self.board_size = game_state.board.size
        self.cell_size = cell_size
        self.margin = margin

        self.sgfmeta = SGFMetadata()
        self.latest_model_response = None

        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_LEFT_UP, self.on_click)

    def get_desired_size(self):
        board_width = self.board_size * self.cell_size + 2 * self.margin
        board_height = self.board_size * self.cell_size + self.cell_size + 2 * self.margin
        return board_width, board_height

    def px_of_x(self, x):
        return round(self.cell_size * x + self.margin + self.cell_size / 2)
    def py_of_y(self, y):
        return round(self.cell_size * y + self.margin + self.cell_size / 2)

    def x_of_px(self, px):
        return round((px - self.margin - self.cell_size / 2) / self.cell_size)
    def y_of_py(self, py):
        return round((py - self.margin - self.cell_size / 2) / self.cell_size)


    def on_paint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        gc = wx.GraphicsContext.Create(dc)

        gc.SetBrush(wx.Brush(wx.Colour(200, 150, 100)))
        gc.SetPen(wx.TRANSPARENT_PEN)
        gc.DrawRectangle(0, 0, self.GetSize().Width, self.GetSize().Height)

        gc.SetPen(wx.Pen(wx.BLACK, 1))

        for i in range(self.board_size):
            gc.StrokeLine(
                self.px_of_x(0),
                self.py_of_y(i),
                self.px_of_x(self.board_size - 1),
                self.py_of_y(i),
            )
            gc.StrokeLine(
                self.px_of_x(i),
                self.py_of_y(0),
                self.px_of_x(i),
                self.py_of_y(self.board_size - 1),
            )

        for x in range(self.board_size):
            for y in range(self.board_size):
                loc = self.game_state.board.loc(x, y)

                if self.game_state.board.board[loc] == Board.BLACK:
                    gc.SetBrush(wx.Brush(wx.BLACK))
                    gc.DrawEllipse(self.px_of_x(x) - (self.cell_size // 2 - 2), self.py_of_y(y) - (self.cell_size // 2 - 2), self.cell_size - 4, self.cell_size - 4)
                elif self.game_state.board.board[loc] == Board.WHITE:
                    gc.SetBrush(wx.Brush(wx.WHITE))
                    gc.DrawEllipse(self.px_of_x(x) - (self.cell_size // 2 - 2), self.py_of_y(y) - (self.cell_size // 2 - 2), self.cell_size - 4, self.cell_size - 4)

        gc.SetBrush(wx.Brush(wx.BLACK, wx.TRANSPARENT))
        for x in range(self.board_size):
            for y in range(self.board_size):
                loc = self.game_state.board.loc(x, y)

                if len(self.game_state.moves) > 0 and self.game_state.moves[-1][1] == loc:
                    if self.game_state.moves[-1][0] == Board.BLACK:
                        gc.SetPen(wx.Pen(wx.Colour(0, 120, 255), 2))
                    else:
                        gc.SetPen(wx.Pen(wx.Colour(0, 50, 255), 2))
                    gc.DrawEllipse(self.px_of_x(x) - (self.cell_size // 2 - 6), self.py_of_y(y) - (self.cell_size // 2 - 6), self.cell_size - 12, self.cell_size - 12)

        # Draw column labels
        gc.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL), wx.BLACK)
        for x in range(self.board_size):
            col_label = "ABCDEFGHJKLMNOPQRSTUVWXYZ"[x]
            text_width, text_height = gc.GetTextExtent(col_label)
            text_x = self.px_of_x(x) - text_width // 2
            gc.DrawText(col_label, text_x, self.py_of_y(-0.8)-text_height//2)
            gc.DrawText(col_label, text_x, self.py_of_y(self.board_size-0.2)-text_height//2)

        # Draw row labels
        for y in range(self.board_size):
            row_label = str(self.board_size - y)
            text_width, text_height = gc.GetTextExtent(row_label)
            text_y = self.py_of_y(y) - text_height // 2
            gc.DrawText(row_label, self.px_of_x(-0.8)-text_width//2, text_y)
            gc.DrawText(row_label, self.px_of_x(self.board_size-0.2)-text_width//2, text_y)

        if self.latest_model_response is not None:
            bigger_font = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
            small_font = wx.Font(8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)

            value_prediction = self.latest_model_response["value"]
            score_prediction = self.latest_model_response["lead"]
            scorestdev_prediction = self.latest_model_response["scorestdev"]
            sign = (1 if self.game_state.board.pla == Board.BLACK else -1)

            gc.SetBrush(wx.Brush(wx.BLACK))
            gc.SetPen(wx.Pen(wx.BLACK,1))
            gc.SetFont(bigger_font,wx.BLACK)

            black_wr = 0.5 + 0.5 * (value_prediction[0] - value_prediction[1]) * sign
            label = "Raw NN Black Win%% %.1f" % (black_wr*100.0)
            text_width, text_height = gc.GetTextExtent(label)
            text_x = self.px_of_x(0)
            text_y = self.py_of_y(self.board_size+0.6) - text_height // 2
            gc.DrawText(label, text_x, text_y)

            black_score = score_prediction * sign
            label = "Raw NN Black ScoreMean %.1f +/- %.1f (2 stdv)" % (black_score, 2 * scorestdev_prediction)
            text_width, text_height = gc.GetTextExtent(label)
            text_x = self.px_of_x(8)
            text_y = self.py_of_y(self.board_size+0.6) - text_height // 2
            gc.DrawText(label, text_x, text_y)

            moves_and_probs0 = dict(self.latest_model_response["moves_and_probs0"])
            # print(moves_and_probs0)
            for y in range(self.board_size):
                for x in range(self.board_size):
                    loc = self.game_state.board.loc(x, y)
                    max_prob = max(moves_and_probs0.values())
                    if loc in moves_and_probs0:
                        prob = moves_and_probs0[loc]
                        r,g,b,a = policy_color(prob / max_prob ** 0.7)
                        # print(r,g,b,a,prob)
                        gc.SetBrush(wx.Brush(wx.Colour(r,g,b,alpha=round(a*0.9))))
                        gc.SetPen(wx.Pen(wx.Colour(0,0,0,alpha=a),1))
                        gc.SetFont(small_font,wx.Colour(0,0,0,alpha=a))
                        gc.DrawRectangle(
                            self.px_of_x(x-0.45),
                            self.py_of_y(y-0.45),
                            self.px_of_x(x+0.45)-self.px_of_x(x-0.45),
                            self.py_of_y(y+0.45)-self.py_of_y(y-0.45),
                        )

                        label = "%.1f" % (prob*100.0)
                        text_width, text_height = gc.GetTextExtent(label)
                        text_x = self.px_of_x(x) - text_width // 2
                        text_y = self.py_of_y(y) - text_height // 2
                        gc.DrawText(label, text_x, text_y)


    def on_click(self, event):
        x = self.x_of_px(event.GetX())
        y = self.y_of_py(event.GetY())

        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            loc = self.game_state.board.loc(x, y)
            pla = self.game_state.board.pla

            if self.game_state.board.would_be_legal(pla,loc):
                self.game_state.play(pla, loc)

                command = {"command": "play", "pla": pla, "loc": loc}
                parent = self.GetParent().GetParent()
                parent.send_command(command)
                response = parent.receive_response()
                if response != {"outputs": ""}:
                    parent.handle_error(f"Unexpected response from server: {response}")

                self.Refresh()
                self.refresh_model()

    def set_sgfmeta(self, sgfmeta):
        self.sgfmeta = sgfmeta

    def refresh_model(self):
        sgfmeta = self.sgfmeta
        command = {"command": "get_model_outputs", "sgfmeta": sgfmeta.to_dict()}
        parent = self.GetParent().GetParent()
        parent.send_command(command)
        response = parent.receive_response()
        if "outputs" not in response:
            parent.handle_error(f"Unexpected response from server: {response}")
        self.latest_model_response = response["outputs"]
        self.Refresh()


class LabeledSlider(wx.Panel):
    def __init__(self, parent, title, options, on_scroll_callback=None, start_option=None):
        super().__init__(parent)

        self.options = options
        self.on_scroll_callback = on_scroll_callback
        self.title = title
        self.is_extrapolation = False

        # Create the slider
        start_idx = 0 if start_option is None else options.index(start_option)
        self.slider = wx.Slider(self, value=start_idx, minValue=0, maxValue=len(options) - 1, style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS)
        self.slider.SetTickFreq(1)  # Set the tick frequency to 1
        self.slider.Bind(wx.EVT_SCROLL, self.on_slider_scroll)

        # Create the label to display the selected option
        self.label = wx.StaticText(self, label = self.title + ": " + str(self.options[start_idx]))

        font_size = 12

        font = self.label.GetFont()
        font.SetPointSize(font_size)
        self.label.SetFont(font)
        font = self.slider.GetFont()
        font.SetPointSize(font_size)
        self.slider.SetFont(font)

        # Create a sizer to arrange the widgets
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.label, 0, wx.ALIGN_LEFT | wx.ALL, 10)
        sizer.Add(self.slider, 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizer(sizer)

    def get_selected_index(self):
        return self.slider.GetValue()

    def get_selected_option(self):
        selected_index = self.get_selected_index()
        return self.options[selected_index]

    def refresh_label(self):
        option_index = self.slider.GetValue()
        selected_option = self.options[option_index]
        self.label.SetLabel(self.title + ": " + str(selected_option) + ("" if not self.is_extrapolation else " (No Training Data)"))

    def set_is_extrapolation(self, b):
        if self.is_extrapolation != b:
            self.is_extrapolation = b
            self.refresh_label()

    def on_slider_scroll(self, event):
        option_index = self.slider.GetValue()
        selected_option = self.options[option_index]
        self.refresh_label()
        if self.on_scroll_callback:
            self.on_scroll_callback(option_index, selected_option)


class SliderWindow(wx.Frame):
    def __init__(self, parent):
        super().__init__(parent, title="Sliders")
        panel = wx.Panel(self)
        panel_sizer = wx.BoxSizer(wx.VERTICAL)
        panel_sizer.Add(400,0,0)

        self.source_slider = LabeledSlider(panel, title="Source", options=["KG","OGS","KGS","Fox","Tygem(Unused)","GoGoD","Go4Go"],
            on_scroll_callback = (lambda idx, option: self.update_metadata()),
            start_option="GoGoD",
        )
        panel_sizer.Add(self.source_slider, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        self.rank_slider = LabeledSlider(panel, title="Rank", options=[
            "KG","9d","8d","7d","6d","5d","4d","3d","2d","1d","1k","2k","3k","4k","5k","6k","7k","8k","9k","10k","11k","12k","13k","14k","15k","16k","17k","18k","19k","20k"
            ],
            on_scroll_callback = (lambda idx, option: self.update_metadata()),
            start_option="9d",
        )
        panel_sizer.Add(self.rank_slider, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        self.date_slider = LabeledSlider(panel, title="Date", options=[
            1800,1825,1850,1875,1900,1915,1930,1940,1950,1960,1970,1980,1985,1990,1995,2000,2005,2008,2010,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],
            on_scroll_callback = (lambda idx, option: self.update_metadata()),
            start_option=2020,
        )
        panel_sizer.Add(self.date_slider, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        self.tc_slider = LabeledSlider(panel, title="TimeControl", options=["Blitz","Fast","Slow","Unknown"],
            on_scroll_callback = (lambda idx, option: self.update_metadata()),
            start_option="Unknown",
        )
        panel_sizer.Add(self.tc_slider, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)


        panel.SetSizer(panel_sizer)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizerAndFit(sizer)

        self.Bind(wx.EVT_CLOSE, self.on_close)

    def on_close(self, event):
        self.GetParent().server_process.terminate()
        self.GetParent().Close()

    def update_metadata(self):
        sgfmeta = SGFMetadata(
            inverseBRank = self.rank_slider.get_selected_index(),
            inverseWRank = self.rank_slider.get_selected_index(),
            bIsHuman = self.rank_slider.get_selected_index() != 0,
            wIsHuman = self.rank_slider.get_selected_index() != 0,
            gameIsUnrated = False,
            gameRatednessIsUnknown = self.source_slider.get_selected_option() == "KGS",
            tcIsUnknown = self.tc_slider.get_selected_option() == "Unknown",
            tcIsByoYomi = self.tc_slider.get_selected_option() != "Unknown",
            mainTimeSeconds = [300,900,1800,0][self.tc_slider.get_selected_index()],
            periodTimeSeconds = [10,15,30,0][self.tc_slider.get_selected_index()],
            byoYomiPeriods = [5,5,5,0][self.tc_slider.get_selected_index()],
            boardArea = 361,
            gameDate = datetime.date(self.date_slider.get_selected_option(),6,1),
            source = self.source_slider.get_selected_index(),
        )

        source = self.source_slider.get_selected_option()
        if source == "KG":
            self.rank_slider.set_is_extrapolation(self.rank_slider.get_selected_index() != 0)
            self.tc_slider.set_is_extrapolation(self.tc_slider.get_selected_option() == "Unknown")
            self.date_slider.set_is_extrapolation(self.date_slider.get_selected_option() < 2022)
        elif source == "OGS":
            self.rank_slider.set_is_extrapolation(self.rank_slider.get_selected_index() == 0)
            self.tc_slider.set_is_extrapolation(self.tc_slider.get_selected_option() == "Unknown")
            self.date_slider.set_is_extrapolation(self.date_slider.get_selected_option() < 2007)
        elif source == "KGS":
            self.rank_slider.set_is_extrapolation(self.rank_slider.get_selected_index() == 0)
            self.tc_slider.set_is_extrapolation(self.tc_slider.get_selected_option() == "Unknown")
            self.date_slider.set_is_extrapolation(self.date_slider.get_selected_option() < 2016)
        elif source == "Fox":
            self.rank_slider.set_is_extrapolation(self.rank_slider.get_selected_index() == 0 or self.rank_slider.get_selected_index() >= 28)
            self.tc_slider.set_is_extrapolation(self.tc_slider.get_selected_option() == "Unknown")
            self.date_slider.set_is_extrapolation(self.date_slider.get_selected_option() < 2014 or self.date_slider.get_selected_option() > 2019)
        elif source == "Tygem(Unused)":
            self.rank_slider.set_is_extrapolation(True)
            self.tc_slider.set_is_extrapolation(True)
            self.date_slider.set_is_extrapolation(True)
        elif source == "GoGoD":
            self.rank_slider.set_is_extrapolation(self.rank_slider.get_selected_index() == 0 or self.rank_slider.get_selected_index() > 5)
            self.tc_slider.set_is_extrapolation(self.tc_slider.get_selected_option() != "Unknown")
            self.date_slider.set_is_extrapolation(self.date_slider.get_selected_option() > 2020)
        elif source == "Go4Go":
            self.rank_slider.set_is_extrapolation(self.rank_slider.get_selected_index() != 1)
            self.tc_slider.set_is_extrapolation(self.tc_slider.get_selected_option() != "Unknown")
            self.date_slider.set_is_extrapolation(self.date_slider.get_selected_option() < 2020)

        self.GetParent().board.set_sgfmeta(sgfmeta)
        self.GetParent().board.refresh_model()

class FileDropTarget(wx.FileDropTarget):
    def __init__(self, window):
        super().__init__()
        self.window = window

    def OnDropFiles(self, x, y, sgf_files):
        return self.window.on_drop_files(sgf_files)

class GoClient(wx.Frame):
    def __init__(self, server_command, game_state):
        super().__init__(parent=None, title="HumanSLNetViz")
        self.server_command = server_command
        self.game_state = game_state
        self.board_size = self.game_state.board_size

        self.SetDropTarget(FileDropTarget(self))

        self.start_server()
        self.init_ui()


    def init_ui(self):
        panel = wx.Panel(self)
        self.board = GoBoard(panel, self.game_state)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.board, 1, wx.EXPAND)
        panel.SetSizer(sizer)

        self.Bind(wx.EVT_CHAR_HOOK, self.on_key_down)

        # Set the initial size of the window based on the board size
        board_width, board_height = self.board.get_desired_size()
        self.SetClientSize(board_width, board_height)

        screen_width, screen_height = wx.DisplaySize()
        frame_width, frame_height = self.GetSize()
        pos_x = (screen_width - frame_width) // 2 - 300
        pos_y = (screen_height - frame_height) // 2
        self.SetPosition((pos_x, pos_y))

        self.slider_window = SliderWindow(self)

        frame_width, frame_height = self.slider_window.GetSize()
        pos_x = (screen_width - frame_width) // 2 + 240
        pos_y = (screen_height - frame_height) // 2
        self.slider_window.SetPosition((pos_x, pos_y))


    def start_server(self):
        print(f"Starting server with command: {self.server_command}")
        self.server_process = subprocess.Popen(
            self.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        atexit.register(self.server_process.terminate)

        def print_stderr():
            while True:
                line = self.server_process.stderr.readline()
                if not line:
                    returncode = self.server_process.poll()
                    if returncode is not None:
                        return
                print(line,end="")

        t = Thread(target=print_stderr)
        t.daemon = True
        t.start()

        self.init_server()

    def init_server(self):
        command = {"command": "start", "board_size": self.board_size, "rules": GameState.RULES_JAPANESE}
        self.send_command(command)
        response = self.receive_response()
        if response != {"outputs": ""}:
            self.handle_error(f"Unexpected response from server: {response}")

        for (pla,loc) in self.game_state.moves:
            command = {"command": "play", "pla": pla, "loc": loc}
            self.send_command(command)
            response = self.receive_response()
            if response != {"outputs": ""}:
                self.handle_error(f"Unexpected response from server: {response}")

    def send_command(self, command):
        print(f"Sending: {json.dumps(command)}")
        self.server_process.stdin.write(json.dumps(command) + "\n")
        self.server_process.stdin.flush()

    def receive_response(self):
        print(f"Waiting for response")
        while True:
            returncode = self.server_process.poll()
            if returncode is not None:
                raise OSError(f"Server terminated unexpectedly with {returncode=}")
            response = self.server_process.stdout.readline().strip()
            if response != "":
                break
        print(f"Got response (first 100 chars): {str(response[:100])}")
        return json.loads(response)

    def handle_error(self, error_message):
        print(f"Error: {error_message}")
        self.server_process.terminate()

        sys.exit(1)

    def on_key_down(self, event):
        key_code = event.GetKeyCode()
        if (key_code == wx.WXK_LEFT or key_code == wx.WXK_BACK) and event.ShiftDown():
            self.undo(10)
        elif key_code == wx.WXK_LEFT or key_code == wx.WXK_BACK:
            self.undo()
        elif key_code == wx.WXK_RIGHT and event.ShiftDown():
            self.redo(10)
        elif key_code == wx.WXK_RIGHT:
            self.redo()
        elif key_code == wx.WXK_DOWN:
            self.undo(len(self.game_state.moves))
        elif key_code == wx.WXK_UP:
            self.redo(len(self.game_state.redo_stack))
        event.Skip()

    def undo(self, undo_count = 1):
        is_refresh_needed = False
        for i in range(undo_count):
            if not self.game_state.can_undo():
                break

            is_refresh_needed = True

            self.game_state.undo()

            command = {"command": "undo"}
            self.send_command(command)
            response = self.receive_response()
            if response != {"outputs": ""}:
                self.handle_error(f"Unexpected response from server: {response}")

        if is_refresh_needed:
            self.board.Refresh()
            self.board.refresh_model()

    def redo(self, redo_count = 1):
        is_refresh_needed = False
        for i in range(redo_count):
            if not self.game_state.can_redo():
                break

            is_refresh_needed = True

            self.game_state.redo()

            command = {"command": "redo"}
            self.send_command(command)
            response = self.receive_response()
            if response != {"outputs": ""}:
                self.handle_error(f"Unexpected response from server: {response}")

        if is_refresh_needed:
            self.board.Refresh()
            self.board.refresh_model()

    def on_drop_files(self, sgf_files):
        if len(sgf_files) == 0:
            return False

        sgf_file = sgf_files[0]
        file_extension = os.path.splitext(sgf_file)[1]
        if file_extension != ".sgf":
            return False

        game_state = load_sgf_game_state(sgf_file)

        self.game_state = game_state
        self.board_size = self.game_state.board_size

        self.board.game_state = game_state
        self.board.board_size = self.board_size

        self.init_server()

        self.board.Refresh()
        self.board.refresh_model()

        return True

    def on_close(self, event):
        self.server_process.terminate()
        event.Skip()

def main():
    sgf_file = None
    server_command = sys.argv[1:]

    if len(server_command) >= 2 and server_command[0] == "-sgf":
        sgf_file = server_command[1]
        server_command = server_command[2:]

    if not server_command:
        print("Usage: python humanslnet_gui.py [-sgf FILENAME] <server_command>")
        sys.exit(1)

    if sgf_file is not None:
        game_state = load_sgf_game_state(sgf_file)
    else:
        game_state = GameState(19, GameState.RULES_JAPANESE)

    app = wx.App()
    client = GoClient(server_command, game_state)
    client.Bind(wx.EVT_CLOSE, client.on_close)
    client.Show()
    client.slider_window.Show()
    client.slider_window.update_metadata()

    app.MainLoop()

if __name__ == "__main__":
    main()
