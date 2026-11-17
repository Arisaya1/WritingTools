"""
AI Provider Architecture for Writing Tools
--------------------------------------------

This module handles different AI model providers (Gemini, OpenAI-compatible, Ollama) and manages their interactions
with the main application. It uses an abstract base class pattern for provider implementations.

Key Components:
1. AIProviderSetting – Base class for provider settings (e.g. API keys, model names)
    • TextSetting      – A simple text input for settings
    • FileSetting      – A text input with a Browse button
    • DropdownSetting  – A dropdown selection setting
    • SpinSetting / CheckboxSetting – Numeric (with hint text) and on/off settings
    • ModelPickerSetting – A GGUF model library picker (see ui/LlamaCppManager.py)
    • SectionSetting / WidgetSetting / HiddenSetting – Headings, embedded widgets and
      invisible stored values (not all of them are saved to the config file)

2. AIProvider – Abstract base class that all providers implement.
   It defines the interface for:
      • Getting a response from the AI model
      • Loading and saving configuration settings
      • Cancelling an ongoing request

3. Provider Implementations:
    • GeminiProvider – Uses Google’s Generative AI API (Gemini) to generate content.
    • OpenAICompatibleProvider – Connects to any OpenAI-compatible API (v1/chat/completions)
    • OllamaProvider – Connects to a locally running Ollama server (e.g. for llama.cpp)
    • LMStudioProvider – Connects to LM Studio's local server (OpenAI-compatible, default port 1234)
    • LlamaCppProvider – Launches and manages a bundled-free `llama-server` (llama.cpp) process itself,
      so any GGUF model (including 1-bit models like Bonsai) can be run without any other app.
      Its Settings page lets you load/unload models, tune load and sampling options and
      try prompts in a built-in playground.

Response Flow:
   • The main app calls get_response() with a system instruction and a prompt.
   • The provider formats and sends the request to its API endpoint.
   • For operations that require a window (e.g. Summary, Key Points), the provider returns the full text.
   • For direct text replacement, the provider emits the full text via the output_ready_signal.
   • Conversation history (for follow-up questions) is maintained by the main app.

Note: Streaming has been fully removed throughout the code.
"""

import atexit
import base64
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser
from abc import ABC, abstractmethod
from typing import List

# External libraries
import httpx
from google import genai
from google.genai import types as genai_types
from ollama import Client as OllamaClient
from openai import OpenAI
from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QVBoxLayout
from ui.LlamaCppManager import (SPIN_CSS, FG, LlamaControlPanel, LlamaPlayground, ModelPickerWidget,
                                SECONDARY_BUTTON_CSS, muted_label, section_title)
from ui.UIUtils import colorMode

# Obfuscation prefix to identify encrypted API keys
_OBFUSCATION_PREFIX = "enc:"
_XOR_KEY = 0x5A  # Simple XOR key for obfuscation


def obfuscate_api_key(key: str) -> str:
    """
    Obfuscate an API key using XOR + Base64 encoding.
    Returns the obfuscated string with 'enc:' prefix.
    """
    if not key or key.startswith(_OBFUSCATION_PREFIX):
        return key  # Already obfuscated or empty
    xored = bytes([b ^ _XOR_KEY for b in key.encode('utf-8')])
    return _OBFUSCATION_PREFIX + base64.b64encode(xored).decode('ascii')


def deobfuscate_api_key(obfuscated: str) -> str:
    """
    Deobfuscate an API key that was obfuscated with obfuscate_api_key().
    If the key doesn't have the 'enc:' prefix, returns it as-is (plaintext).
    """
    if not obfuscated or not obfuscated.startswith(_OBFUSCATION_PREFIX):
        return obfuscated  # Not obfuscated, return as-is
    encoded = obfuscated[len(_OBFUSCATION_PREFIX):]
    xored = base64.b64decode(encoded)
    return bytes([b ^ _XOR_KEY for b in xored]).decode('utf-8')


class AIProviderSetting(ABC):
    """
    Abstract base class for a provider setting (e.g., API key, model selection).

    Settings with `persist = False` are pure UI (section headings, buttons, panels):
    nothing is read from or written to the config file for them.
    """
    persist = True

    def __init__(self, name: str, display_name: str = None, default_value: str = None, description: str = None):
        self.name = name
        self.display_name = display_name if display_name else name
        self.default_value = default_value if default_value else ""
        self.description = description if description else ""

    @abstractmethod
    def render_to_layout(self, layout: QVBoxLayout):
        """Render the setting widget(s) into the provided layout."""
        pass

    @abstractmethod
    def set_value(self, value):
        """Set the internal value from configuration."""
        pass

    @abstractmethod
    def get_value(self):
        """Return the current value from the widget."""
        pass

    def reset(self):
        """Put the setting (and its widget, if shown) back to its default value."""
        pass


class TextSetting(AIProviderSetting):
    """
    A text-based setting (for API keys, URLs, etc.).
    """
    def __init__(self, name: str, display_name: str = None, default_value: str = None, description: str = None):
        super().__init__(name, display_name, default_value, description)
        self.internal_value = default_value
        self.input = None

    def render_to_layout(self, layout: QVBoxLayout):
        row_layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(self.display_name)
        label.setStyleSheet(f"font-size: 16px; color: {'#ffffff' if colorMode=='dark' else '#333333'};")
        row_layout.addWidget(label)
        self.input = QtWidgets.QLineEdit(self.internal_value)
        self.input.setStyleSheet(f"""
            font-size: 16px;
            padding: 5px;
            background-color: {'#444' if colorMode=='dark' else 'white'};
            color: {'#ffffff' if colorMode=='dark' else '#000000'};
            border: 1px solid {'#666' if colorMode=='dark' else '#ccc'};
        """)
        self.input.setPlaceholderText(self.description)
        row_layout.addWidget(self.input)
        layout.addLayout(row_layout)

    def set_value(self, value):
        self.internal_value = value

    def get_value(self):
        return self.input.text()

    def reset(self):
        self.internal_value = self.default_value
        if self.input is not None:
            self.input.setText(self.default_value or "")


class FileSetting(TextSetting):
    """
    A text setting with a "Browse..." button that opens a native file picker.
    The value is stored as a plain path string.
    """
    def __init__(self, name: str, display_name: str = None, default_value: str = None,
                 description: str = None, file_filter: str = "All files (*)"):
        super().__init__(name, display_name, default_value, description)
        self.file_filter = file_filter

    def render_to_layout(self, layout: QVBoxLayout):
        super().render_to_layout(layout)
        # TextSetting appended its row as the last item of the layout; add a browse button to it.
        row_layout = layout.itemAt(layout.count() - 1).layout()
        browse = QtWidgets.QPushButton("Browse...")
        browse.setStyleSheet(f"""
            font-size: 14px;
            padding: 5px 10px;
            background-color: {'#444' if colorMode=='dark' else 'white'};
            color: {'#ffffff' if colorMode=='dark' else '#000000'};
            border: 1px solid {'#666' if colorMode=='dark' else '#ccc'};
        """)

        def pick():
            start = self.input.text().strip()
            start_dir = os.path.dirname(start) if start and os.path.dirname(start) else ""
            path, _selected = QtWidgets.QFileDialog.getOpenFileName(
                browse, f"Select {self.display_name}", start_dir, self.file_filter)
            if path:
                self.input.setText(path)

        browse.clicked.connect(pick)
        row_layout.addWidget(browse)


class DropdownSetting(AIProviderSetting):
    """
    A dropdown setting (e.g., for selecting a model).

    Optionally supports a "Custom" option that reveals a text input for arbitrary values.
    When allow_custom=True, users can select "Custom" from the dropdown and enter any value.
    If the loaded config value doesn't match any preset option, "Custom" is auto-selected.
    """
    # Sentinel value used internally to identify the "Custom" dropdown option
    _CUSTOM_SENTINEL = "__custom__"

    def __init__(self, name: str, display_name: str = None, default_value: str = None,
                 description: str = None, options: list = None, allow_custom: bool = False,
                 custom_placeholder: str = "Enter custom value"):
        super().__init__(name, display_name, default_value, description)
        self.options = options if options else []
        self.internal_value = default_value
        self.dropdown = None
        self.allow_custom = allow_custom
        self.custom_placeholder = custom_placeholder
        self.custom_input = None
        self.custom_input_container = None

    def render_to_layout(self, layout: QVBoxLayout):
        row_layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(self.display_name)
        label.setStyleSheet(f"font-size: 16px; color: {'#ffffff' if colorMode=='dark' else '#333333'};")
        row_layout.addWidget(label)
        self.dropdown = QtWidgets.QComboBox()
        self.dropdown.setStyleSheet(f"""
            font-size: 16px;
            padding: 5px;
            background-color: {'#444' if colorMode=='dark' else 'white'};
            color: {'#ffffff' if colorMode=='dark' else '#000000'};
            border: 1px solid {'#666' if colorMode=='dark' else '#ccc'};
        """)

        # Add preset options
        for option, value in self.options:
            self.dropdown.addItem(option, value)

        # Add "Custom" option if enabled
        if self.allow_custom:
            self.dropdown.addItem("🔧 Custom", self._CUSTOM_SENTINEL)

        # Set initial selection based on internal_value
        index = self.dropdown.findData(self.internal_value)
        if index != -1:
            # Value matches a preset option
            self.dropdown.setCurrentIndex(index)
        elif self.allow_custom and self.internal_value:
            # Value doesn't match any preset - it's a custom value, select "Custom"
            custom_index = self.dropdown.findData(self._CUSTOM_SENTINEL)
            if custom_index != -1:
                self.dropdown.setCurrentIndex(custom_index)

        row_layout.addWidget(self.dropdown)
        layout.addLayout(row_layout)

        # Create custom input row if allow_custom is enabled
        if self.allow_custom:
            self.custom_input_container = QtWidgets.QWidget()
            custom_row_layout = QtWidgets.QHBoxLayout(self.custom_input_container)
            custom_row_layout.setContentsMargins(0, 5, 0, 0)

            self.custom_input = QtWidgets.QLineEdit()
            self.custom_input.setPlaceholderText(self.custom_placeholder)
            self.custom_input.setStyleSheet(f"""
                font-size: 16px;
                padding: 5px;
                background-color: {'#444' if colorMode=='dark' else 'white'};
                color: {'#ffffff' if colorMode=='dark' else '#000000'};
                border: 1px solid {'#666' if colorMode=='dark' else '#ccc'};
            """)

            # If current value is custom (not in presets), populate the input
            if self.dropdown.currentData() == self._CUSTOM_SENTINEL and self.internal_value:
                self.custom_input.setText(self.internal_value)

            custom_row_layout.addWidget(self.custom_input)
            layout.addWidget(self.custom_input_container)

            # Connect signal to show/hide custom input when dropdown changes
            self.dropdown.currentIndexChanged.connect(self._on_dropdown_changed)
            # Set initial visibility
            self._update_custom_input_visibility()

    def _on_dropdown_changed(self):
        """Handle dropdown selection change to show/hide custom input."""
        self._update_custom_input_visibility()

    def _update_custom_input_visibility(self):
        """Show or hide the custom input based on dropdown selection."""
        if self.custom_input_container:
            is_custom = self.dropdown.currentData() == self._CUSTOM_SENTINEL
            self.custom_input_container.setVisible(is_custom)
            # Focus the input when switching to Custom for better UX
            if is_custom and self.custom_input:
                self.custom_input.setFocus()

    def set_value(self, value):
        self.internal_value = value

    def get_value(self):
        # If "Custom" is selected, return the text input value (stripped of whitespace)
        if self.allow_custom and self.dropdown.currentData() == self._CUSTOM_SENTINEL:
            return self.custom_input.text().strip()
        return self.dropdown.currentData()

    def reset(self):
        self.internal_value = self.default_value
        if self.dropdown is not None:
            index = self.dropdown.findData(self.default_value)
            self.dropdown.setCurrentIndex(max(index, 0))


def _add_with_hint(layout: QVBoxLayout, control, description: str):
    """Add a control (widget or layout) with its hint text tucked right underneath it."""
    group = QtWidgets.QVBoxLayout()
    group.setContentsMargins(0, 0, 0, 0)
    group.setSpacing(1)
    if isinstance(control, QtWidgets.QLayout):
        group.addLayout(control)
    else:
        group.addWidget(control)
    if description:
        group.addWidget(muted_label(description))
    layout.addLayout(group)


class _NoWheelSpinBox(QtWidgets.QSpinBox):
    """Spin box that ignores the mouse wheel unless focused, so scrolling the page doesn't change values."""
    def wheelEvent(self, event):
        event.ignore() if not self.hasFocus() else super().wheelEvent(event)


class _NoWheelDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore() if not self.hasFocus() else super().wheelEvent(event)


class SpinSetting(AIProviderSetting):
    """
    A numeric setting shown as a spin box. `decimals=0` gives an integer setting.
    `special_text` is shown instead of the number when the value equals `minimum`
    (e.g. "Auto" for 0), which is how "off / default / unlimited" values are labelled.
    The description is shown as a small hint line under the control.
    """
    def __init__(self, name: str, display_name: str = None, default_value=0, description: str = None,
                 minimum=0, maximum=1_000_000, step=1, decimals: int = 0,
                 special_text: str = None, suffix: str = ""):
        super().__init__(name, display_name, None, description)
        self.default_value = default_value  # (the base class turns falsy values like 0 into "")
        self.minimum, self.maximum, self.step, self.decimals = minimum, maximum, step, decimals
        self.special_text, self.suffix = special_text, suffix
        self.internal_value = default_value
        self.spin = None

    def _coerce(self, value):
        try:
            number = float(value) if self.decimals else int(float(value))
        except (TypeError, ValueError):
            number = self.default_value
        return min(max(number, self.minimum), self.maximum)

    def render_to_layout(self, layout: QVBoxLayout):
        row_layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(self.display_name)
        label.setStyleSheet(f"font-size: 16px; color: {FG};")
        row_layout.addWidget(label)
        row_layout.addStretch(1)
        if self.decimals:
            self.spin = _NoWheelDoubleSpinBox()
            self.spin.setDecimals(self.decimals)
        else:
            self.spin = _NoWheelSpinBox()
        self.spin.setRange(self.minimum, self.maximum)
        self.spin.setSingleStep(self.step)
        if self.special_text:
            self.spin.setSpecialValueText(self.special_text)
        if self.suffix:
            self.spin.setSuffix(self.suffix)
        self.spin.setValue(self._coerce(self.internal_value))
        self.spin.setFixedWidth(150)
        # No `border` here: any border rule makes Qt drop the native up/down arrows.
        self.spin.setStyleSheet(SPIN_CSS)
        self.spin.setToolTip(self.description)
        row_layout.addWidget(self.spin)
        _add_with_hint(layout, row_layout, self.description)

    def set_value(self, value):
        self.internal_value = self._coerce(value)

    def get_value(self):
        return self.spin.value()

    def reset(self):
        self.internal_value = self.default_value
        if self.spin is not None:
            self.spin.setValue(self.default_value)


class CheckboxSetting(AIProviderSetting):
    """An on/off setting shown as a checkbox, with the description as a hint line below."""
    def __init__(self, name: str, display_name: str = None, default_value: bool = False, description: str = None):
        super().__init__(name, display_name, None, description)
        self.default_value = bool(default_value)
        self.internal_value = self.default_value
        self.checkbox = None

    @staticmethod
    def _coerce(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def render_to_layout(self, layout: QVBoxLayout):
        self.checkbox = QtWidgets.QCheckBox(self.display_name)
        self.checkbox.setStyleSheet(f"font-size: 16px; color: {FG};")
        self.checkbox.setChecked(self.internal_value)
        self.checkbox.setToolTip(self.description)
        _add_with_hint(layout, self.checkbox, self.description)

    def set_value(self, value):
        self.internal_value = self._coerce(value)

    def get_value(self):
        return self.checkbox.isChecked()

    def reset(self):
        self.internal_value = self.default_value
        if self.checkbox is not None:
            self.checkbox.setChecked(self.default_value)


class HiddenSetting(AIProviderSetting):
    """A value that is saved in the config but has no widget of its own (e.g. a list of paths)."""
    def __init__(self, name: str, default_value=None):
        super().__init__(name, name, None, None)
        self.default_value = default_value
        self.internal_value = default_value

    def render_to_layout(self, layout: QVBoxLayout):
        pass

    def set_value(self, value):
        self.internal_value = value

    def get_value(self):
        return self.internal_value


class SectionSetting(AIProviderSetting):
    """A heading (with an optional hint line) that groups the settings below it. Not saved."""
    persist = False

    def __init__(self, title: str, description: str = None):
        super().__init__(title, title, None, description)

    def render_to_layout(self, layout: QVBoxLayout):
        layout.addWidget(section_title(self.display_name))
        if self.description:
            layout.addWidget(muted_label(self.description))

    def set_value(self, value):
        pass

    def get_value(self):
        return None


class WidgetSetting(AIProviderSetting):
    """Embeds an arbitrary widget (built by `factory()` each time the page is shown). Not saved."""
    persist = False

    def __init__(self, name: str, factory):
        super().__init__(name, name, None, None)
        self.factory = factory

    def render_to_layout(self, layout: QVBoxLayout):
        layout.addWidget(self.factory())

    def set_value(self, value):
        pass

    def get_value(self):
        return None


class ModelPickerSetting(AIProviderSetting):
    """
    Pick a GGUF model from a library of known files (see ModelPickerWidget).
    The selected file path is the value; the library list is kept in `library_setting`.
    """
    def __init__(self, name: str, display_name: str, library_setting: HiddenSetting):
        super().__init__(name, display_name, "", "Path to a .gguf model file")
        self.library_setting = library_setting
        self.internal_value = ""
        self.widget = None

    def render_to_layout(self, layout: QVBoxLayout):
        library = self.library_setting.get_value() or []
        self.widget = ModelPickerWidget(self.display_name, self.internal_value, library)
        self.widget.changed.connect(self._sync_library)
        layout.addWidget(self.widget)
        self._sync_library()

    def _sync_library(self):
        self.library_setting.set_value(self.widget.library())

    def set_value(self, value):
        self.internal_value = value or ""

    def get_value(self):
        return self.widget.selected_path()

    def reset(self):
        pass


class AIProvider(ABC):
    """
    Abstract base class for AI providers.
    
    All providers must implement:
      • get_response(system_instruction, prompt) -> str
      • after_load() to create their client or model instance
      • before_load() to cleanup any existing client
      • cancel() to cancel an ongoing request
    """
    def __init__(self, app, provider_name: str, settings: List[AIProviderSetting],
                 description: str = "An unfinished AI provider!",
                 logo: str = "generic",
                 button_text: str = "Go to URL",
                 button_action: callable = None):
        self.provider_name = provider_name
        self.settings = settings
        self.app = app
        self.description = description if description else "An unfinished AI provider!"
        self.logo = logo
        self.button_text = button_text
        self.button_action = button_action

    @abstractmethod
    def get_response(self, system_instruction: str, prompt: str) -> str:
        """
        Send the given system instruction and prompt to the AI provider and return the full response text.
        """
        pass

    def load_config(self, config: dict):
        """
        Load configuration settings into the provider.
        """
        for setting in self.settings:
            if not setting.persist:
                continue
            if setting.name in config:
                setattr(self, setting.name, config[setting.name])
                setting.set_value(config[setting.name])
            else:
                setattr(self, setting.name, setting.default_value)
        self.after_load()

    def save_config(self):
        """
        Save provider configuration settings into the main config file.
        """
        config = {}
        for setting in self.settings:
            if setting.persist:
                config[setting.name] = setting.get_value()
        self.app.config["providers"][self.provider_name] = config
        self.app.save_config(self.app.config)

    @abstractmethod
    def after_load(self):
        """
        Called after configuration is loaded; create your API client here.
        """
        pass

    @abstractmethod
    def before_load(self):
        """
        Called before reloading configuration; cleanup your API client here.
        """
        pass

    @abstractmethod
    def cancel(self):
        """
        Cancel any ongoing API request.
        """
        pass


class GeminiProvider(AIProvider):
    """
    Provider for Google's Gemini API (using the new unified `google-genai` SDK).

    Uses `client.models.generate_content()` for single-shot generation. The same
    method is used for follow-up chat too — the entire conversation history is
    passed via `contents` as a list of `Content` objects, so we don't need the
    SDK's chat session abstraction.

    System instruction is passed via `GenerateContentConfig.system_instruction`
    (not concatenated into `contents`, as the legacy SDK required).

    Thinking is disabled (set to "minimal", the lowest level the API exposes)
    on Gemini 3-family models. Gemma models don't have a thinking process, so
    `thinking_config` is omitted for them — passing it could otherwise error.
    """

    # Disable safety filtering across all categories (best-effort; some models
    # may still soft-refuse). Defined once, reused per request.
    _SAFETY_SETTINGS = [
        genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        genai_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        genai_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    ]

    def __init__(self, app):
        self.close_requested = False
        self.client = None

        settings = [
            TextSetting(name="api_key", display_name="API Key", description="Paste your Gemini API key here"),
            DropdownSetting(
                name="model_name",
                display_name="Model",
                default_value="gemini-flash-latest",
                description="Select Gemini model to use",
                options=[
                    # `gemini-flash-latest` is a Google-managed alias that currently
                    # points to Gemini 3 Flash Preview — fast (~1–2s) and high
                    # quality. Capped at 20 free requests/day per the model's free
                    # tier.
                    ("⭐ Gemini Flash Latest (very fast | only 20 free uses/day)", "gemini-flash-latest"),
                    # Gemma 4 models are unlimited on the free tier but noticeably
                    # slower (8–15s typical) since they run on different
                    # infrastructure.
                    ("Gemma 4 31B (slow | unlimited free use)", "gemma-4-31b-it"),
                    ("Gemma 4 26B A4B (slow | unlimited free use)", "gemma-4-26b-a4b-it"),
                ],
                allow_custom=True,
                custom_placeholder="e.g., gemini-3.1-pro-preview"
            )
        ]
        super().__init__(app, "Gemini (Recommended)", settings,
            "• Google's Gemini is a powerful AI model available for free!\n"
            "• An API key is required to connect to Gemini on your behalf.\n"
            "• Click the button below to get your API key.",
            "gemini",
            "Get API Key",
            lambda: webbrowser.open("https://aistudio.google.com/app/apikey"))

    def _build_config(self, system_instruction: str) -> "genai_types.GenerateContentConfig":
        """
        Build a per-call GenerateContentConfig.

        We don't override temperature: Gemini 3 docs explicitly recommend leaving
        it at the default of 1.0 (lower values can cause looping / degraded
        output on reasoning-heavy tasks). The old SDK code set it to 0.5; we drop
        that override here.
        """
        # Thinking is disabled across the board for Writing Tools (latency matters
        # more than reasoning depth for proofread/rewrite/summary flows).
        #
        # • Gemma 4 *is* capable of thinking, but is off by default. Per
        #   https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api#thinking,
        #   thinking on Gemma 4 is binary and "you enable it in the API by setting
        #   the thinking level to 'high'". So omitting thinking_config keeps
        #   Gemma 4 in its default-off state.
        # • Gemini 3 Flash / Flash-Lite cannot fully disable thinking. The
        #   lowest exposed level is "minimal", which the docs say "matches the
        #   'no thinking' setting for most queries".
        is_gemma = "gemma" in (self.model_name or "").lower()
        kwargs = {
            "system_instruction": system_instruction,
            "safety_settings": self._SAFETY_SETTINGS,
            "max_output_tokens": 1000,
        }
        if not is_gemma:
            kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_level="minimal")
        return genai_types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _messages_to_contents(messages: list) -> list:
        """
        Convert OpenAI-style chat history into google-genai `Content` objects.

        Maps roles: assistant → "model", everything else → "user". Any "system"
        entries are dropped because Gemini takes the system instruction via the
        config object instead of as an in-history message.
        """
        contents = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                continue
            text = m.get("content", "")
            gemini_role = "model" if role == "assistant" else "user"
            contents.append(genai_types.Content(role=gemini_role, parts=[genai_types.Part(text=text)]))
        return contents

    def get_response(self, system_instruction: str, prompt, return_response: bool = False) -> str:
        """
        Generate content using Gemini.

        `prompt` may be either a plain string (the typical inline-tool flow) or a
        list of OpenAI-style message dicts (the follow-up chat flow). In both
        cases we make a single-shot non-streaming request.

        Returns the response text when `return_response` is True; otherwise emits
        it via `output_ready_signal` for inline replacement.
        """
        self.close_requested = False

        try:
            contents = self._messages_to_contents(prompt) if isinstance(prompt, list) else prompt

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=self._build_config(system_instruction),
            )

            response_text = (response.text or "").rstrip('\n')

            if not return_response and not hasattr(self.app, 'current_response_window'):
                self.app.output_ready_signal.emit(response_text)
                self.app.replace_text(True)
                return ""
            return response_text
        except Exception as e:
            logging.error(f"Error processing Gemini response: {e}")
            self.app.output_ready_signal.emit("An error occurred while processing the response.")
            return ""
        finally:
            self.close_requested = False

    def load_config(self, config: dict):
        """
        Load configuration, deobfuscating the API key if needed.
        """
        # Deobfuscate API key before loading
        if 'api_key' in config:
            config = config.copy()  # Don't modify the original
            config['api_key'] = deobfuscate_api_key(config['api_key'])
        super().load_config(config)

    def save_config(self):
        """
        Save configuration, obfuscating the API key for storage.
        """
        config = {}
        for setting in self.settings:
            value = setting.get_value()
            # Obfuscate API key before saving
            if setting.name == 'api_key':
                value = obfuscate_api_key(value)
            config[setting.name] = value
        self.app.config["providers"][self.provider_name] = config
        self.app.save_config(self.app.config)

    def after_load(self):
        """
        Construct the new `google-genai` Client. The model and per-call options
        are passed at request time via `client.models.generate_content`, so we
        don't pre-instantiate a model here.
        """
        self.client = genai.Client(api_key=self.api_key)

    def before_load(self):
        self.client = None

    def cancel(self):
        self.close_requested = True


class OpenAICompatibleProvider(AIProvider):
    """
    Provider for OpenAI-compatible APIs.
    
    Uses self.client.chat.completions.create() to obtain a response.
    Streaming is fully removed.
    """
    def __init__(self, app):
        self.close_requested = None
        self.client = None

        settings = [
            TextSetting(name="api_key", display_name="API Key", description="API key for the OpenAI-compatible API."),
            TextSetting("api_base", "API Base URL", "https://api.openai.com/v1", "E.g. https://api.openai.com/v1"),
            TextSetting("api_organisation", "API Organisation", "", "Leave blank if not applicable."),
            TextSetting("api_project", "API Project", "", "Leave blank if not applicable."),
            TextSetting("api_model", "API Model", "gpt-4o-mini", "E.g. gpt-4o-mini"),
        ]
        super().__init__(app, "OpenAI Compatible (For Experts)", settings,
            "• Connect to ANY OpenAI-compatible API (v1/chat/completions).\n"
            "• You must abide by the service's Terms of Service.",
            "openai", "Get OpenAI API Key", lambda: webbrowser.open("https://platform.openai.com/account/api-keys"))

    def get_response(self, system_instruction: str, prompt: str | list, return_response: bool = False) -> str:
        """
        Send a chat request to the OpenAI-compatible API.
        
        Always performs a non-streaming request.
        If prompt is not a list, builds a simple two-message conversation.
        Returns the response text if return_response is True,
        otherwise emits it via output_ready_signal.
        """
        self.close_requested = False

        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ]

        try:
            response = self.client.chat.completions.create(
                model=self.api_model,
                messages=messages,
                temperature=0.5,
                stream=False
            )
            response_text = response.choices[0].message.content.strip()

            if not return_response and not hasattr(self.app, 'current_response_window'):
                self.app.output_ready_signal.emit(response_text)
            return response_text

        except Exception as e:
            error_str = str(e)
            logging.error(f"Error while generating content: {error_str}")
            if "exceeded" in error_str or "rate limit" in error_str:
                self.app.show_message_signal.emit(
                    "Rate Limit Hit",
                    "It appears you have hit an API rate/usage limit. Please try again later or adjust your settings."
                )
            else:
                self.app.show_message_signal.emit("Error", f"An error occurred: {error_str}")
            return ""

    def after_load(self):
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            organization=self.api_organisation,
            project=self.api_project
        )

    def before_load(self):
        self.client = None

    def cancel(self):
        self.close_requested = True


class OllamaProvider(AIProvider):
    """
    Provider for connecting to an Ollama server.
    
    Uses the /chat endpoint of the Ollama server to generate a response.
    Streaming is not used.
    """
    def __init__(self, app):
        self.close_requested = None
        self.client = None
        self.app = app
        settings = [
            TextSetting("api_base", "API Base URL", "http://localhost:11434", "E.g. http://localhost:11434"),
            TextSetting("api_model", "API Model", "llama3.1:8b", "E.g. llama3.1:8b"),
            TextSetting("keep_alive", "Time to keep the model loaded in memory in minutes", "5", "E.g. 5")
        ]
        super().__init__(app, "Ollama (For Experts)", settings,
            "• Connect to an Ollama server (local LLM).",
            "ollama", "Ollama Set-up Instructions",
            lambda: webbrowser.open("https://github.com/theJayTea/WritingTools?tab=readme-ov-file#-optional-ollama-local-llm-instructions-for-windows-v7-onwards"))

    def get_response(self, system_instruction: str, prompt: str | list, return_response: bool = False) -> str:
        """
        Send a chat request to the Ollama server.
        
        Always performs a non-streaming request.
        Returns the response text if return_response is True,
        otherwise emits it via output_ready_signal.
        """
        self.close_requested = False

        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ]

        try:
            response = self.client.chat(model=self.api_model, messages=messages)
            response_text = response['message']['content'].strip()
            if not return_response and not hasattr(self.app, 'current_response_window'):
                self.app.output_ready_signal.emit(response_text)
            return response_text
        except Exception as e:
            logging.error(f"Error during Ollama chat: {e}")
            self.app.output_ready_signal.emit("An error occurred during Ollama chat.")
            return ""

    def after_load(self):
        self.client = OllamaClient(host=self.api_base)

    def before_load(self):
        self.client = None

    def cancel(self):
        self.close_requested = True


# Matches a complete <think>...</think> reasoning block that some local models emit.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class _LocalOpenAIStyleProvider(AIProvider):
    """
    Shared implementation for local providers that talk to an OpenAI-compatible
    /v1/chat/completions endpoint (LM Studio, a managed llama.cpp server).

    Subclasses set `self.client` in `after_load()` and implement `_prepare_request()`,
    which returns the model name to use (and may start servers, resolve defaults, etc.).
    """
    # Extra JSON body fields sent with every request (subclasses may override).
    extra_body = None

    def _prepare_request(self) -> str:
        raise NotImplementedError

    def _friendly_error(self, error: Exception) -> str:
        return f"An error occurred: {error}"

    def _after_request(self):
        """Called once a request finishes (successfully or not)."""
        pass

    def _request_kwargs(self) -> dict:
        """Extra keyword arguments (sampling parameters etc.) for chat.completions.create()."""
        kwargs = {"temperature": 0.5}
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        return kwargs

    @staticmethod
    def _clean(text: str) -> str:
        """Strip reasoning blocks so only the final answer replaces the user's text."""
        return _THINK_BLOCK_RE.sub("", text or "").strip()

    def get_response(self, system_instruction: str, prompt: str | list, return_response: bool = False) -> str:
        self.close_requested = False

        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ]

        try:
            model = self._prepare_request()
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                **self._request_kwargs()
            )
            response_text = self._clean(response.choices[0].message.content)

            if not return_response and not hasattr(self.app, 'current_response_window'):
                self.app.output_ready_signal.emit(response_text)
            return response_text

        except Exception as e:
            if self.close_requested:
                # The user cancelled; don't pop up an error for it.
                return ""
            logging.error(f"Error while generating content with {self.provider_name}: {e}")
            self.app.show_message_signal.emit("Error", self._friendly_error(e))
            return ""
        finally:
            self._after_request()

    def before_load(self):
        self.client = None

    def cancel(self):
        self.close_requested = True

    def shutdown(self):
        """Release external resources (child processes etc.) when the app exits."""
        pass


class LMStudioProvider(_LocalOpenAIStyleProvider):
    """
    Provider for LM Studio's local server (https://lmstudio.ai).

    LM Studio exposes an OpenAI-compatible API (default http://localhost:1234/v1)
    once you start its server (Developer tab → Start Server). If no model name is
    given, the first model reported by the server is used.
    """
    def __init__(self, app):
        self.close_requested = None
        self.client = None

        settings = [
            TextSetting("api_base", "Server URL", "http://localhost:1234/v1", "E.g. http://localhost:1234/v1"),
            TextSetting("api_model", "Model", "", "Leave blank to use the model loaded in LM Studio"),
            TextSetting("api_key", "API Key", "lm-studio", "Only needed if you enabled auth in LM Studio"),
        ]
        super().__init__(app, "LM Studio (For Experts)", settings,
            "• Connect to LM Studio's local server (OpenAI-compatible API).\n"
            "• In LM Studio, open the Developer tab and click \"Start Server\".\n"
            "• Leave Model blank to use whichever model is loaded.",
            "lmstudio", "Get LM Studio", lambda: webbrowser.open("https://lmstudio.ai"))

    def after_load(self):
        self.client = OpenAI(
            api_key=(self.api_key or "lm-studio"),
            base_url=(self.api_base or "http://localhost:1234/v1").strip(),
            timeout=600.0,
        )

    def _prepare_request(self) -> str:
        model = (self.api_model or "").strip()
        if model:
            return model
        # No model configured: use the first non-embedding model the server reports.
        models = [m.id for m in self.client.models.list().data]
        chat_models = [m for m in models if "embed" not in m.lower()]
        if not chat_models:
            raise RuntimeError("LM Studio reports no models. Load a model in LM Studio or set one in Settings.")
        return chat_models[0]

    def _friendly_error(self, error: Exception) -> str:
        text = str(error)
        if "Connection" in type(error).__name__ or "connect" in text.lower():
            return ("Could not reach LM Studio. Make sure its server is running "
                    f"(Developer tab → Start Server) and the URL is correct ({self.api_base}).")
        return f"An error occurred: {text}"


class LlamaCppProvider(_LocalOpenAIStyleProvider):
    """
    Runs models locally by launching and managing a llama.cpp `llama-server` process.

    No other app (Ollama, LM Studio...) is needed: pick a `llama-server` binary and a
    GGUF model in Settings. The server is started on first use on a free local port
    (or on demand from the Settings page), reused for later requests, restarted if its
    launch settings change, shut down after an idle timeout (to free RAM/VRAM) and when
    Writing Tools exits.

    The Settings page offers a model library, Load/Unload buttons with live status,
    every common load option and sampling parameter, a log viewer and a playground for
    trying the model out. Values shown there apply to the Load button and the
    playground immediately; Save makes them permanent.

    1-bit / low-bit models (e.g. PrismML's Bonsai, quantized as Q1_0) work as long as
    the `llama-server` build supports the model's quantization type. Recent upstream
    llama.cpp does for Q1_0; PrismML also publishes known-good binaries.
    """
    HEALTH_TIMEOUT_SECONDS = 600  # Loading a big model from a slow disk can take minutes.

    # Settings that are applied per request (no model reload needed).
    GENERATION_SETTINGS = ("temperature", "top_p", "top_k", "min_p", "repeat_penalty",
                           "max_tokens", "seed", "enable_thinking")
    # Settings that need the server to be (re)started to take effect.
    LOADING_SETTINGS = ("context_size", "gpu_layers", "threads", "batch_size", "flash_attn",
                        "use_mmap", "mlock", "idle_minutes", "extra_args")

    def __init__(self, app):
        self.close_requested = None
        self.client = None
        self._lock = threading.RLock()          # serializes starting/stopping the server
        self._timer_lock = threading.Lock()     # guards the idle timer (never held for long)
        self._counter_lock = threading.Lock()
        self._process = None
        self._port = None
        self._started_with = None
        self._loaded_model = None
        self._log_path = os.path.join(tempfile.gettempdir(), "writingtools_llama_server.log")
        self._idle_timer = None
        self._idle_deadline = None
        self._busy = 0
        # Live state shown in the Settings page.
        self._state = "stopped"                 # stopped | loading | running | error
        self._detail = ""
        self._load_started = None
        self._load_seconds = None
        self._started_at = None
        self._build = ""
        self._loading_requested = False
        self._load_cancel = threading.Event()
        self.playground_prompts = {
            "system": "You are a proofreading assistant. Fix grammar, spelling and punctuation in the "
                      "user's text. Output ONLY the corrected text.",
            "user": "i has went to the store yesterday and buyed three apple, but they was rotten.",
        }

        library = HiddenSetting("model_library", [])
        settings = [
            SectionSetting("Model"),
            FileSetting("server_binary", "llama-server", "llama-server",
                        "Path to llama-server, or just 'llama-server' if it is on your PATH",
                        file_filter="All files (*)"),
            library,
            ModelPickerSetting("model_path", "GGUF Model", library),
            WidgetSetting("control_panel", lambda: LlamaControlPanel(self)),

            SectionSetting("Try It Out"),
            WidgetSetting("playground", lambda: LlamaPlayground(self)),

            SectionSetting("Generation Settings", "Applied to every request. No reload needed."),
            SpinSetting("temperature", "Temperature", 0.5, "Higher = more creative, lower = more predictable",
                        minimum=0.0, maximum=2.0, step=0.05, decimals=2),
            SpinSetting("top_p", "Top P", 0.95, "Only sample from the smallest set of tokens covering this much probability (1 = off)",
                        minimum=0.0, maximum=1.0, step=0.05, decimals=2),
            SpinSetting("top_k", "Top K", 40, "Only sample from the K most likely tokens",
                        minimum=0, maximum=1000, special_text="Off"),
            SpinSetting("min_p", "Min P", 0.05, "Drop tokens less likely than this fraction of the best one (0 = off)",
                        minimum=0.0, maximum=1.0, step=0.01, decimals=2),
            SpinSetting("repeat_penalty", "Repeat Penalty", 1.0, "Above 1 discourages repeating itself (1 = off)",
                        minimum=0.5, maximum=2.0, step=0.05, decimals=2),
            SpinSetting("max_tokens", "Max Output Tokens", 0, "Stop generating after this many tokens",
                        minimum=0, maximum=131072, step=64, special_text="No limit"),
            SpinSetting("seed", "Seed", -1, "Fix a seed for repeatable output (needs Temperature above 0 to matter)",
                        minimum=-1, maximum=2147483647, special_text="Random"),
            CheckboxSetting("enable_thinking", "Let the model think before answering", False,
                            "Only for reasoning models. Slower; the reasoning is removed from the final text."),
            WidgetSetting("reset_generation", lambda: self._reset_button(self.GENERATION_SETTINGS)),

            SectionSetting("Loading Options", "Applied when the model is loaded. Changing these reloads it."),
            SpinSetting("context_size", "Context Size", 8192, "Max tokens of text + answer the model can handle at once (more = more memory)",
                        minimum=256, maximum=1048576, step=1024, suffix=" tokens"),
            SpinSetting("gpu_layers", "GPU Layers", 99, "99 = put everything on the GPU, 0 = CPU only",
                        minimum=0, maximum=999),
            SpinSetting("threads", "CPU Threads", 0, "Threads used for CPU work",
                        minimum=0, maximum=256, special_text="Auto"),
            SpinSetting("batch_size", "Batch Size", 0, "Tokens processed at once while reading your text",
                        minimum=0, maximum=65536, step=128, special_text="Default"),
            DropdownSetting("flash_attn", "Flash Attention", "auto", "Faster and lighter on memory where supported",
                            options=[("Auto", "auto"), ("On", "on"), ("Off", "off")]),
            CheckboxSetting("use_mmap", "Memory-map the model file", True,
                            "Faster loading and shared memory. Turn off if loading is slow from a network drive."),
            CheckboxSetting("mlock", "Keep the model in RAM", False,
                            "Stops the OS from swapping the model out (needs enough free RAM)."),
            SpinSetting("idle_minutes", "Unload After Idle", 10, "Free RAM/VRAM after this long without a request",
                        minimum=0, maximum=1440, special_text="Never", suffix=" min"),
            TextSetting("extra_args", "Extra Arguments", "", "Any other llama-server flags, e.g. --rope-scaling yarn"),
            WidgetSetting("reset_loading", lambda: self._reset_button(self.LOADING_SETTINGS)),
        ]
        super().__init__(app, "Local llama.cpp (Built-in)", settings,
            "• Runs GGUF models directly on your PC with llama.cpp. No other app needed.\n"
            "• Add a model, then load, tweak and test it right here.\n"
            "• Supports 1-bit models like Bonsai (Q1_0) when your llama-server build supports them.",
            "llamacpp", "Get llama.cpp",
            lambda: webbrowser.open("https://github.com/ggml-org/llama.cpp/releases"))
        atexit.register(self.shutdown)

    # -- configuration helpers ---------------------------------------------------

    @staticmethod
    def _int(value, default: int) -> int:
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _float(value, default: float) -> float:
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _bool(value, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _saved_cfg(self) -> dict:
        """The settings last loaded from the config file (defaults for anything missing)."""
        return {s.name: getattr(self, s.name, s.default_value) for s in self.settings if s.persist}

    def form_config(self) -> dict:
        """
        The settings as currently shown in the Settings page (saved or not). Falls back
        to the saved value for any setting whose widget isn't on screen.
        """
        cfg = self._saved_cfg()
        for s in self.settings:
            if not s.persist:
                continue
            try:
                cfg[s.name] = s.get_value()
            except (RuntimeError, AttributeError):
                pass  # widget not created, or already deleted
        return cfg

    def _reset_button(self, names) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton("Reset to Defaults")
        button.setStyleSheet(SECONDARY_BUTTON_CSS)
        button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        def reset():
            for s in self.settings:
                if s.name in names:
                    s.reset()
        button.clicked.connect(reset)
        container = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        row.addWidget(button)
        return container

    def _find_binary(self, cfg: dict = None) -> str:
        cfg = cfg or self._saved_cfg()
        configured = (cfg.get("server_binary") or "llama-server").strip().strip('"')
        exe = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        if os.path.isdir(configured):
            candidate = os.path.join(configured, exe)
            if os.path.isfile(candidate):
                return candidate
        elif os.path.isfile(configured):
            return configured
        found = shutil.which(configured)
        if found:
            return found
        raise RuntimeError(
            f"Could not find llama-server ('{configured}'). Download llama.cpp from "
            "github.com/ggml-org/llama.cpp/releases, then set the llama-server path in Settings.")

    def _build_command(self, port: int, cfg: dict) -> list:
        import shlex
        model = (cfg.get("model_path") or "").strip().strip('"')
        if not model or not os.path.isfile(model):
            raise RuntimeError("No valid GGUF model selected. Add or choose a .gguf file in Settings.")
        cmd = [
            self._find_binary(cfg), "-m", model,
            "--host", "127.0.0.1", "--port", str(port),
            "-c", str(self._int(cfg.get("context_size"), 8192)),
            "-ngl", str(self._int(cfg.get("gpu_layers"), 99)),
        ]
        threads = self._int(cfg.get("threads"), 0)
        if threads > 0:
            cmd += ["-t", str(threads)]
        batch = self._int(cfg.get("batch_size"), 0)
        if batch > 0:
            cmd += ["-b", str(batch)]
        if cfg.get("flash_attn") in ("on", "off"):
            cmd += ["-fa", cfg["flash_attn"]]
        if not self._bool(cfg.get("use_mmap"), True):
            cmd.append("--no-mmap")
        if self._bool(cfg.get("mlock"), False):
            cmd.append("--mlock")
        cmd += shlex.split(cfg.get("extra_args") or "", posix=(sys.platform != "win32"))
        return cmd

    def _launch_signature(self, cfg: dict) -> tuple:
        """Everything that requires a server restart when it changes."""
        return (
            (cfg.get("server_binary") or "").strip(), (cfg.get("model_path") or "").strip(),
            self._int(cfg.get("context_size"), 8192), self._int(cfg.get("gpu_layers"), 99),
            self._int(cfg.get("threads"), 0), self._int(cfg.get("batch_size"), 0),
            cfg.get("flash_attn") or "auto", self._bool(cfg.get("use_mmap"), True),
            self._bool(cfg.get("mlock"), False), (cfg.get("extra_args") or "").strip(),
        )

    def _request_kwargs(self, cfg: dict = None) -> dict:
        """Sampling parameters for chat.completions.create(), from `cfg` or the saved settings."""
        cfg = cfg or self._saved_cfg()
        kwargs = {
            "temperature": self._float(cfg.get("temperature"), 0.5),
            "top_p": self._float(cfg.get("top_p"), 0.95),
        }
        max_tokens = self._int(cfg.get("max_tokens"), 0)
        if max_tokens > 0:
            kwargs["max_tokens"] = max_tokens
        seed = self._int(cfg.get("seed"), -1)
        if seed >= 0:
            kwargs["seed"] = seed
        # llama-server accepts these extra sampling fields on its OpenAI-compatible endpoint.
        kwargs["extra_body"] = {
            "top_k": self._int(cfg.get("top_k"), 40),
            "min_p": self._float(cfg.get("min_p"), 0.05),
            "repeat_penalty": self._float(cfg.get("repeat_penalty"), 1.0),
            "chat_template_kwargs": {"enable_thinking": self._bool(cfg.get("enable_thinking"), False)},
        }
        return kwargs

    # -- status (read by the Settings page; never takes the server lock) ----------

    def status(self) -> dict:
        """A snapshot of the server's state for the UI."""
        process = self._process
        if self._state == "running" and process is not None and process.poll() is not None:
            self._state = "stopped"
            self._detail = (f"llama-server stopped unexpectedly (exit code {process.returncode}). "
                            "Check the log for details.")
        state = self._state
        now = time.time()
        info = {"state": state, "detail": self._detail}
        if state == "loading":
            info["elapsed"] = now - (self._load_started or now)
        elif state == "running":
            info.update({
                "model": self._loaded_model or "model", "port": self._port, "build": self._build,
                "load_seconds": self._load_seconds, "uptime": now - (self._started_at or now),
                "busy": self._busy > 0,
                "idle_in": max(0.0, self._idle_deadline - now) if self._idle_deadline else None,
            })
        return info

    def log_tail(self, lines: int = 8) -> str:
        try:
            with open(self._log_path, "r", errors="replace") as f:
                return "".join(f.readlines()[-lines:]).strip()
        except OSError:
            return ""

    _log_tail = log_tail

    def check_binary(self, cfg: dict = None) -> str:
        """Run `llama-server --version` and describe the result (also lists detected GPUs)."""
        exe = self._find_binary(cfg)
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
        try:
            result = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30,
                                    stdin=subprocess.DEVNULL, errors="replace", **kwargs)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{exe} did not respond to --version within 30 seconds.")
        lines = [l.strip() for l in (result.stdout + "\n" + result.stderr).splitlines() if l.strip()]
        return "✔ " + exe + "\n" + "\n".join(lines[:6])

    # -- server lifecycle --------------------------------------------------------

    def _wait_until_ready(self, process, port: int):
        deadline = time.time() + self.HEALTH_TIMEOUT_SECONDS
        url = f"http://127.0.0.1:{port}/health"
        # Never route the local health check through a system/env HTTP proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.time() < deadline:
            if self.close_requested or self._load_cancel.is_set():
                raise RuntimeError("Cancelled while the model was loading.")
            if process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited while loading the model (code {process.returncode}). "
                    f"Your llama-server build may not support this model's format.\n\n{self.log_tail()}")
            try:
                with opener.open(url, timeout=2) as resp:
                    if resp.status == 200:
                        return opener
            except Exception:
                pass  # 503 while loading, or not listening yet
            time.sleep(0.5)
        raise RuntimeError("Timed out waiting for llama-server to load the model.")

    def _fetch_build(self, opener, port: int) -> str:
        try:
            with opener.open(f"http://127.0.0.1:{port}/props", timeout=3) as resp:
                import json
                return str(json.load(resp).get("build_info") or "")
        except Exception:
            return ""

    def _start_server(self, cfg: dict):
        self._load_cancel.clear()
        self._load_started, self._load_seconds, self._build = time.time(), None, ""
        self._state, self._detail = "loading", ""
        try:
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            cmd = self._build_command(port, cfg)
            logging.info(f"Starting llama-server: {cmd}")

            kwargs = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            log = open(self._log_path, "w")
            try:
                process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kwargs)
            finally:
                log.close()
            self._process, self._port = process, port
            opener = self._wait_until_ready(process, port)
        except Exception as e:
            if self._load_cancel.is_set():
                self._stop_server("stopped", "Loading was cancelled.")
            else:
                self._stop_server("error", str(e))
            raise
        self._started_with = self._launch_signature(cfg)
        self._loaded_model = os.path.basename((cfg.get("model_path") or "model").strip().strip('"'))
        self._build = self._fetch_build(opener, port)
        # The server is on loopback; don't let HTTP(S)_PROXY env vars intercept it.
        self.client = OpenAI(api_key="llama-cpp", base_url=f"http://127.0.0.1:{port}/v1", timeout=600.0,
                             http_client=httpx.Client(trust_env=False, timeout=600.0))
        self._load_seconds = time.time() - self._load_started
        self._started_at = time.time()
        self._state = "running"

    def _stop_server(self, state: str = "stopped", detail: str = ""):
        """Stop the server and record why (the detail is shown in the Settings page)."""
        with self._lock:
            self._detail = detail  # set first so a reader that sees the new state also sees its detail
            self._state = state
            process, self._process = self._process, None
            self._started_with = None
            self._cancel_idle_shutdown()
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    def _stop_server_in_background(self):
        threading.Thread(target=self._stop_server, daemon=True).start()

    def _cancel_idle_shutdown(self):
        with self._timer_lock:
            if self._idle_timer:
                self._idle_timer.cancel()
                self._idle_timer = None
            self._idle_deadline = None

    def _schedule_idle_shutdown(self, minutes=None):
        """(Re)start the countdown to unload the model, unless a request is in flight."""
        if minutes is None:
            minutes = self._saved_cfg().get("idle_minutes")
        minutes = self._int(minutes, 10)
        with self._timer_lock:
            if self._idle_timer:
                self._idle_timer.cancel()
                self._idle_timer = None
            self._idle_deadline = None
            if minutes <= 0 or self._process is None or self._busy > 0:
                return
            self._idle_timer = threading.Timer(minutes * 60, self._unload_idle)
            self._idle_timer.daemon = True
            self._idle_timer.start()
            self._idle_deadline = time.time() + minutes * 60

    def _unload_idle(self):
        self._stop_server("stopped", "Unloaded automatically after being idle.")

    def _ensure_server(self, cfg: dict = None, force: bool = False) -> dict:
        """Make sure a server matching `cfg` (default: saved settings) is running."""
        cfg = cfg or self._saved_cfg()
        with self._lock:
            alive = self._process is not None and self._process.poll() is None
            if force or not alive or self._started_with != self._launch_signature(cfg):
                self._stop_server()
                self._start_server(cfg)
            # Never unload the model while a request is in flight; the idle
            # countdown restarts when the request finishes.
            self._cancel_idle_shutdown()
        return cfg

    def _bump_busy(self, delta: int):
        with self._counter_lock:
            self._busy = max(0, self._busy + delta)

    # -- actions used by the Settings page (all non-blocking) ----------------------

    def load_model_async(self, cfg: dict = None, force: bool = False):
        """Load the model in the background, using `cfg` (default: saved settings)."""
        if self._loading_requested or self._state == "loading":
            return
        self._loading_requested = True
        cfg = cfg or self._saved_cfg()

        def work():
            try:
                self._ensure_server(cfg, force=force)
                self._schedule_idle_shutdown(cfg.get("idle_minutes"))
            except Exception as e:
                logging.warning(f"Loading llama.cpp model failed: {e}")  # shown in the UI via status()
            finally:
                self._loading_requested = False

        threading.Thread(target=work, daemon=True).start()

    def cancel_load(self):
        self._load_cancel.set()

    def unload_model(self):
        self.client = None
        self._detail = ""
        self._stop_server_in_background()

    def playground_generate(self, system: str, prompt: str, cfg: dict, on_token, stop_event) -> str:
        """
        Stream a completion for the Settings playground using `cfg`, loading (or reloading)
        the model first if needed. Calls on_token(text, kind) with kind "text" or "thinking".
        Returns a one-line summary with speed statistics.
        """
        messages = []
        if (system or "").strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        self._bump_busy(1)
        try:
            try:
                self._ensure_server(cfg)
            except RuntimeError:
                if stop_event.is_set():
                    return "Stopped."
                raise
            client, model = self.client, self._loaded_model
            if client is None:
                raise RuntimeError("The model was unloaded.")
            generation_started = time.time()
            chunks, first_token_at, timings, usage = 0, None, None, None
            stream = client.chat.completions.create(
                model=model, messages=messages, stream=True,
                stream_options={"include_usage": True}, **self._request_kwargs(cfg))
            try:
                for chunk in stream:
                    if stop_event.is_set():
                        break
                    extra = getattr(chunk, "model_extra", None) or {}
                    timings = extra.get("timings") or timings
                    usage = getattr(chunk, "usage", None) or usage
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    reasoning = (getattr(delta, "model_extra", None) or {}).get("reasoning_content")
                    if reasoning:
                        on_token(reasoning, "thinking")
                    if delta.content:
                        on_token(delta.content, "text")
                    if reasoning or delta.content:
                        first_token_at = first_token_at or time.time()
                        chunks += 1
            finally:
                stream.close()
            elapsed = time.time() - generation_started
        finally:
            self._bump_busy(-1)
            self._schedule_idle_shutdown(cfg.get("idle_minutes"))

        stopped = stop_event.is_set()
        if timings and timings.get("predicted_n") is not None:
            parts = [f"{timings['predicted_n']} tokens", f"{timings.get('predicted_per_second', 0):.1f} tok/s"]
            if timings.get("prompt_n"):
                parts.append(f"prompt {timings['prompt_n']} tokens at {timings.get('prompt_per_second', 0):.0f} tok/s")
        else:
            tokens = getattr(usage, "completion_tokens", None) or chunks
            gen_time = (time.time() - first_token_at) if first_token_at else 0
            parts = [f"~{tokens} tokens"]
            if gen_time > 0.2 and tokens > 1:
                parts.append(f"~{(tokens - 1) / gen_time:.1f} tok/s")
        parts.append(f"{elapsed:.1f}s")
        if first_token_at:
            parts.append(f"first token after {first_token_at - generation_started:.1f}s")
        return ("Stopped — " if stopped else "Done — ") + " · ".join(parts)

    # -- provider interface ------------------------------------------------------

    def _prepare_request(self) -> str:
        self._bump_busy(1)  # balanced by _after_request(), which always runs
        self._ensure_server()
        # llama-server serves the one model it was started with; the name is informational.
        return self._loaded_model or "model"

    def _friendly_error(self, error: Exception) -> str:
        return str(error) if isinstance(error, RuntimeError) else f"An error occurred: {error}"

    def _after_request(self):
        self._bump_busy(-1)
        self._schedule_idle_shutdown()

    def after_load(self):
        # Only drop a running server if its launch settings changed (saving unrelated
        # settings shouldn't force a slow model reload); it restarts lazily on next use.
        # This runs on the UI thread, so it must never wait for a model that is mid-load.
        if self._started_with is not None and self._started_with != self._launch_signature(self._saved_cfg()):
            self.client = None
            self._stop_server_in_background()
        elif self._process is not None:
            self._schedule_idle_shutdown()  # idle timeout may have changed

    def before_load(self):
        self.client = None
        self._stop_server_in_background()

    def cancel(self):
        self.close_requested = True

    def shutdown(self):
        # Abort any in-progress load first so we don't wait on it.
        self._load_cancel.set()
        self._stop_server()
