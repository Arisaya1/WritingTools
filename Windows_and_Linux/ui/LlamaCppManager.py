"""
GUI building blocks for the built-in llama.cpp provider's Settings page:

  • ModelPickerWidget  – model library (browse / scan a folder / remove) with GGUF info
  • LlamaControlPanel  – live status plus Load / Unload / Reload, log viewer, binary check
  • LlamaPlayground    – send a test prompt to the loaded model and watch it stream

The widgets only talk to the provider through a small duck-typed interface
(`status()`, `form_config()`, `load_model_async()`, `cancel_load()`, `unload_model()`,
`check_binary()`, `log_tail()`, `playground_generate()`), so this module doesn't
import aiprovider (which imports this module).
"""

import os
import threading
import webbrowser

from PySide6 import QtCore, QtGui, QtWidgets

import gguf_info
from ui.UIUtils import colorMode

DARK = colorMode == "dark"
FG = "#ffffff" if DARK else "#333333"
MUTED = "#aaaaaa" if DARK else "#666666"

INPUT_CSS = f"""
    font-size: 15px;
    padding: 5px;
    background-color: {'#444' if DARK else 'white'};
    color: {'#ffffff' if DARK else '#000000'};
    border: 1px solid {'#666' if DARK else '#ccc'};
"""

# Spin boxes must not get a `border` rule (Qt would then stop drawing their arrows).
SPIN_CSS = f"""
    font-size: 15px;
    padding: 3px;
    background-color: {'#444' if DARK else 'white'};
    color: {'#ffffff' if DARK else '#000000'};
"""

SECONDARY_BUTTON_CSS = f"""
    QPushButton {{
        font-size: 14px;
        padding: 5px 10px;
        background-color: {'#444' if DARK else 'white'};
        color: {'#ffffff' if DARK else '#000000'};
        border: 1px solid {'#666' if DARK else '#ccc'};
    }}
    QPushButton:hover {{ background-color: {'#555' if DARK else '#f0f0f0'}; }}
    QPushButton:disabled {{ color: #888888; }}
"""

PRIMARY_BUTTON_CSS = f"""
    QPushButton {{
        background-color: {'#4CAF50' if DARK else '#008CBA'};
        color: white;
        padding: 7px 14px;
        font-size: 15px;
        border: none;
        border-radius: 5px;
    }}
    QPushButton:hover {{ background-color: {'#45a049' if DARK else '#007095'}; }}
    QPushButton:disabled {{ background-color: #777777; color: #cccccc; }}
"""

LINK_BUTTON_CSS = f"""
    QPushButton {{
        font-size: 13px;
        border: none;
        background: transparent;
        color: {'#6cb6ff' if DARK else '#0969da'};
        text-align: left;
    }}
    QPushButton:hover {{ text-decoration: underline; }}
"""

STATE_COLORS = {"running": "#4CAF50", "loading": "#FFA000", "error": "#E53935", "stopped": "#888888"}


def muted_label(text: str = "", size: int = 12) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(f"font-size: {size}px; color: {MUTED};")
    return label


def section_title(text: str) -> QtWidgets.QWidget:
    """A bold heading with a rule underneath, used to group settings."""
    box = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(box)
    layout.setContentsMargins(0, 8, 0, 0)
    layout.setSpacing(4)
    label = QtWidgets.QLabel(text)
    label.setStyleSheet(f"font-size: 17px; font-weight: bold; color: {FG};")
    layout.addWidget(label)
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
    line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
    layout.addWidget(line)
    return box


def _button(text: str, css: str = SECONDARY_BUTTON_CSS) -> QtWidgets.QPushButton:
    button = QtWidgets.QPushButton(text)
    button.setStyleSheet(css)
    button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
    return button


def _format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


# --------------------------------------------------------------------------------------
# Model library
# --------------------------------------------------------------------------------------

class ModelPickerWidget(QtWidgets.QWidget):
    """
    Dropdown of known GGUF models plus buttons to grow/shrink that list.
    The library is just a list of file paths; files are never copied or deleted.
    """
    changed = QtCore.Signal()

    def __init__(self, display_name: str, selected: str, library: list, parent=None):
        super().__init__(parent)
        self._library = []
        for path in list(library or []) + ([selected] if selected else []):
            if path and path not in self._library:
                self._library.append(path)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        row = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(display_name)
        label.setStyleSheet(f"font-size: 16px; color: {FG};")
        row.addWidget(label)
        self.combo = QtWidgets.QComboBox()
        self.combo.setStyleSheet(INPUT_CSS)
        self.combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(20)
        self.combo.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        row.addWidget(self.combo, 1)
        layout.addLayout(row)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(6)
        self.browse_button = _button("Add Model...")
        self.browse_button.setToolTip("Pick a .gguf file and add it to the list")
        self.browse_button.clicked.connect(self._browse)
        self.scan_button = _button("Scan Folder...")
        self.scan_button.setToolTip("Add every .gguf file found in a folder (and its subfolders)")
        self.scan_button.clicked.connect(self._scan)
        self.remove_button = _button("Remove")
        self.remove_button.setToolTip("Remove the selected model from this list (the file is NOT deleted)")
        self.remove_button.clicked.connect(self._remove)
        for b in (self.browse_button, self.scan_button, self.remove_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        find = _button("Find models online ↗", LINK_BUTTON_CSS)
        find.clicked.connect(lambda: webbrowser.open("https://huggingface.co/models?library=gguf&sort=trending"))
        buttons.addWidget(find)
        layout.addLayout(buttons)

        self.info = muted_label()
        layout.addWidget(self.info)

        self._refresh(select=selected)
        self.combo.currentIndexChanged.connect(self._on_index_changed)

    # -- public API ---------------------------------------------------------------

    def selected_path(self) -> str:
        return self.combo.currentData() or ""

    def library(self) -> list:
        return list(self._library)

    # -- internals ----------------------------------------------------------------

    def _refresh(self, select: str = ""):
        self.combo.blockSignals(True)
        self.combo.clear()
        for path in self._library:
            name = os.path.basename(path)
            if os.path.isfile(path):
                text = f"{name}  ({gguf_info.format_size(os.path.getsize(path))})"
            else:
                text = f"⚠ {name} (file not found)"
            self.combo.addItem(text, path)
            self.combo.setItemData(self.combo.count() - 1, path, QtCore.Qt.ItemDataRole.ToolTipRole)
        if not self._library:
            self.combo.addItem("No models yet — click Add Model...", "")
        index = self.combo.findData(select) if select else -1
        self.combo.setCurrentIndex(index if index >= 0 else 0)
        self.combo.blockSignals(False)
        self.remove_button.setEnabled(bool(self._library))
        self._update_info()

    def _update_info(self, note: str = ""):
        path = self.selected_path()
        if not path:
            text = "Add a .gguf model to get started."
        elif not os.path.isfile(path):
            text = f"File not found: {path}"
        else:
            text = f"{gguf_info.describe(gguf_info.read_gguf_info(path))}\n{path}"
        self.info.setText(f"{note}\n{text}" if note else text)

    def _on_index_changed(self, _index: int):
        self._update_info()
        self.changed.emit()

    def _start_dir(self) -> str:
        path = self.selected_path()
        return os.path.dirname(path) if path else ""

    def _browse(self):
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select GGUF Model", self._start_dir(), "GGUF models (*.gguf);;All files (*)")
        if not path:
            return
        if path not in self._library:
            self._library.append(path)
        self._refresh(select=path)
        self.changed.emit()

    def _scan(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a folder containing GGUF models", self._start_dir())
        if not folder:
            return
        found = gguf_info.find_models(folder)
        new = [p for p in found if p not in self._library]
        self._library.extend(new)
        current = self.selected_path()
        self._refresh(select=current or (new[0] if new else ""))
        self._update_info(note=f"Found {len(found)} model(s) in that folder, added {len(new)} new." if found
                          else "No .gguf files found in that folder.")
        self.changed.emit()

    def _remove(self):
        path = self.selected_path()
        if path in self._library:
            self._library.remove(path)
        self._refresh()
        self.changed.emit()


# --------------------------------------------------------------------------------------
# Load / unload / status
# --------------------------------------------------------------------------------------

class LlamaControlPanel(QtWidgets.QWidget):
    """Shows what the llama-server is doing and lets you load/unload the model on demand."""
    binary_checked = QtCore.Signal(str)

    def __init__(self, provider, parent=None):
        super().__init__(parent)
        self.provider = provider
        self._log_text = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.status_label.setStyleSheet(f"font-size: 15px; color: {FG};")
        self.status_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Minimum)
        layout.addWidget(self.status_label)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(6)
        self.load_button = _button("Load Model", PRIMARY_BUTTON_CSS)
        self.load_button.setToolTip("Start llama-server with the settings below. Takes effect immediately; "
                                    "click Save to keep them.")
        self.load_button.clicked.connect(self._on_load_clicked)
        self.unload_button = _button("Unload")
        self.unload_button.setToolTip("Stop llama-server and free its RAM/VRAM")
        self.unload_button.clicked.connect(self._on_unload_clicked)
        self.log_button = _button("Show Log")
        self.log_button.setCheckable(True)
        self.log_button.toggled.connect(self._on_log_toggled)
        self.check_button = _button("Check llama-server")
        self.check_button.setToolTip("Run llama-server --version to verify the binary and see which GPUs it can use")
        self.check_button.clicked.connect(self._on_check_clicked)
        for b in (self.load_button, self.unload_button, self.log_button, self.check_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.note_label = muted_label()
        self.note_label.setVisible(False)
        self.note_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.note_label)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(150)
        self.log_view.setMaximumHeight(220)
        self.log_view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(9)
        self.log_view.setFont(font)
        self.log_view.setStyleSheet(INPUT_CSS + "font-size: 12px;")
        self.log_view.setVisible(False)
        layout.addWidget(self.log_view)

        self.binary_checked.connect(self._on_binary_checked)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.refresh)
        self.refresh()

    # Poll only while visible so a hidden/closed Settings window costs nothing.
    def showEvent(self, event):
        self.timer.start()
        self.refresh()
        super().showEvent(event)

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    # -- state ----------------------------------------------------------------------

    def refresh(self):
        s = self.provider.status()
        state = s["state"]
        dot = f'<span style="color:{STATE_COLORS.get(state, "#888")}">●</span>'
        if state == "running":
            parts = [f"<b>Loaded</b> — {s['model']}", f"port {s['port']}"]
            if s.get("load_seconds") is not None:
                parts.append(f"loaded in {s['load_seconds']:.1f}s")
            parts.append(f"up {_format_duration(s['uptime'])}")
            if s.get("build"):
                parts.append(f"build {s['build']}")
            if s.get("busy"):
                parts.append("generating…")
            elif s.get("idle_in") is not None:
                parts.append(f"auto-unload in {_format_duration(s['idle_in'])}")
            text = " · ".join(parts)
        elif state == "loading":
            text = f"<b>Loading model…</b> {_format_duration(s['elapsed'])} (large models can take a while)"
        elif state == "error":
            text = "<b>Failed to load</b>"
        else:
            text = "<b>Not loaded</b> — it loads automatically on first use, or click Load Model."
            if s.get("detail"):
                text += f"<br><span style='color:{MUTED}'>{_escape(s['detail'])}</span>"
        self.status_label.setText(f"{dot} {text}")
        if state == "error" and s.get("detail"):
            self._show_note(s["detail"])

        loading = state == "loading"
        self.load_button.setText("Cancel Loading" if loading else "Reload Model" if state == "running" else "Load Model")
        self.unload_button.setEnabled(state == "running")
        if self.log_view.isVisible():
            self._refresh_log()

    def _show_note(self, text: str):
        self.note_label.setText(text)
        self.note_label.setVisible(bool(text))

    def _refresh_log(self):
        text = self.provider.log_tail(200)
        if text == self._log_text:
            return
        self._log_text = text
        bar = self.log_view.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4
        self.log_view.setPlainText(text)
        if at_bottom:
            bar.setValue(bar.maximum())

    # -- actions --------------------------------------------------------------------

    def _on_load_clicked(self):
        if self.provider.status()["state"] == "loading":
            self.provider.cancel_load()
        else:
            self._show_note("")
            self.provider.load_model_async(self.provider.form_config(), force=True)
        self.refresh()

    def _on_unload_clicked(self):
        self.provider.unload_model()
        self.refresh()

    def _on_log_toggled(self, shown: bool):
        self.log_button.setText("Hide Log" if shown else "Show Log")
        self.log_view.setVisible(shown)
        self._log_text = None
        if shown:
            self._refresh_log()

    def _on_check_clicked(self):
        self.check_button.setEnabled(False)
        self._show_note("Checking llama-server…")
        cfg = self.provider.form_config()

        def work():
            try:
                result = self.provider.check_binary(cfg)
            except Exception as e:
                result = str(e)
            try:
                self.binary_checked.emit(result)
            except RuntimeError:
                pass  # widget was destroyed meanwhile

        threading.Thread(target=work, daemon=True).start()

    def _on_binary_checked(self, text: str):
        self.check_button.setEnabled(True)
        self._show_note(text)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")


# --------------------------------------------------------------------------------------
# Playground
# --------------------------------------------------------------------------------------

class LlamaPlayground(QtWidgets.QWidget):
    """Send a prompt to the model using the settings currently shown, and watch it stream."""
    token = QtCore.Signal(str, str)   # text, kind ("text" | "thinking")
    finished = QtCore.Signal(str)     # stats line
    failed = QtCore.Signal(str)

    def __init__(self, provider, parent=None):
        super().__init__(parent)
        self.provider = provider
        self._stop = threading.Event()
        self._running = False
        # Holder so the destroyed-handler (which must not reference self) sees the current event.
        self._stop_holder = {"event": self._stop}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(muted_label(
            "Try your model out here. Uses the load and generation settings currently shown below, "
            "even before you click Save."))

        prompts = provider.playground_prompts
        layout.addWidget(self._caption("System prompt"))
        self.system_edit = self._editor(prompts["system"], 64)
        self.system_edit.textChanged.connect(lambda: prompts.__setitem__("system", self.system_edit.toPlainText()))
        layout.addWidget(self.system_edit)

        layout.addWidget(self._caption("Your text"))
        self.prompt_edit = self._editor(prompts["user"], 90)
        self.prompt_edit.textChanged.connect(lambda: prompts.__setitem__("user", self.prompt_edit.toPlainText()))
        layout.addWidget(self.prompt_edit)

        buttons = QtWidgets.QHBoxLayout()
        self.run_button = _button("Run", PRIMARY_BUTTON_CSS)
        self.run_button.clicked.connect(self._run)
        self.stop_button = _button("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_clicked)
        self.clear_button = _button("Clear Output")
        self.clear_button.clicked.connect(lambda: self.output.clear())
        for b in (self.run_button, self.stop_button, self.clear_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Model output appears here")
        self.output.setMinimumHeight(150)
        self.output.setStyleSheet(INPUT_CSS)
        layout.addWidget(self.output)

        self.stats = muted_label()
        layout.addWidget(self.stats)

        self.token.connect(self._on_token)
        self.finished.connect(self._on_finished)
        self.failed.connect(self._on_failed)
        # If the Settings page is rebuilt/closed mid-generation, stop the stream.
        holder = self._stop_holder
        self.destroyed.connect(lambda *_: holder["event"].set())

    @staticmethod
    def _caption(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setStyleSheet(f"font-size: 14px; color: {FG};")
        return label

    @staticmethod
    def _editor(text: str, height: int) -> QtWidgets.QPlainTextEdit:
        edit = QtWidgets.QPlainTextEdit(text)
        edit.setFixedHeight(height)
        edit.setStyleSheet(INPUT_CSS)
        return edit

    # -- running ----------------------------------------------------------------------

    def _set_running(self, running: bool):
        self._running = running
        self.run_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def _run(self):
        prompt = self.prompt_edit.toPlainText()
        if not prompt.strip():
            self.stats.setText("Type some text to send first.")
            return
        system = self.system_edit.toPlainText()
        cfg = self.provider.form_config()
        stop = threading.Event()
        self._stop = self._stop_holder["event"] = stop
        self.output.clear()
        self.stats.setText("Waiting for the model…")
        self._set_running(True)

        def work():
            try:
                try:
                    stats = self.provider.playground_generate(
                        system, prompt, cfg, lambda text, kind: self.token.emit(text, kind), stop)
                    self.finished.emit(stats)
                except Exception as e:
                    self.failed.emit(str(e))
            except RuntimeError:
                pass  # emit() on a destroyed widget: nobody is listening any more

        threading.Thread(target=work, daemon=True).start()

    def _stop_clicked(self):
        self._stop.set()
        self.provider.cancel_load()
        self.stats.setText("Stopping…")

    def _on_token(self, text: str, kind: str):
        if self._stop.is_set():
            return
        cursor = self.output.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        fmt = QtGui.QTextCharFormat()
        if kind == "thinking":
            fmt.setForeground(QtGui.QColor(MUTED))
            fmt.setFontItalic(True)
        cursor.insertText(text, fmt)
        self.output.setTextCursor(cursor)
        self.output.ensureCursorVisible()
        self.stats.setText("Generating…")

    def _on_finished(self, stats: str):
        self._set_running(False)
        self.stats.setText(stats)

    def _on_failed(self, message: str):
        self._set_running(False)
        self.stats.setText("")
        self.output.setPlainText(f"Error: {message}")
