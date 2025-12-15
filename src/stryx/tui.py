#!/usr/bin/env python3
"""Interactive TUI for editing Pydantic-based configuration files.

Features:
- Schema-aware field editing with type information
- Fuzzy search through all config fields
- Live preview of nested values
- Quick boolean toggling
- In-place value editing with validation
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Type

import yaml
from rapidfuzz import fuzz, process

from pydantic import BaseModel

from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, VSplit, Window, FloatContainer, Float
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import TextArea, Frame, Dialog, Button
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style

from .config import ConfigManager
from .utils import FieldPath, path_to_str, parse_like_yaml

# UI Constants
SEARCH_HEIGHT = 1
STATUS_HEIGHT = 1
EDIT_AREA_HEIGHT = 1
EDIT_DIALOG_WIDTH = 80
LIST_VIEWPORT_BEFORE = 50
LIST_VIEWPORT_SIZE = 200
VALUE_DISPLAY_LENGTH = 120
FIELD_DISPLAY_WIDTH = 40
PAGE_JUMP_SIZE = 20

# Fuzzy matching constants
FUZZY_MATCH_THRESHOLD = 60  # Minimum score (0-100) to include in results
FUZZY_SCORE_CUTOFF = 40     # Hard cutoff for process.extract


class PydanticConfigTUI:
    """Terminal UI for editing Pydantic-based configuration files."""

    def __init__(self, schema: Type[BaseModel], file_path: Path):
        self.config = ConfigManager(schema, file_path)
        self.schema = schema

        # Search state
        self.query = ""
        self.filtered_items: list[tuple[FieldPath, Any]] = []
        self.selected_index = 0

        # Edit modal state
        self.edit_visible = False
        self.edit_path: FieldPath | None = None

        # Variant selection state
        self.variant_visible = False
        self.variant_path: FieldPath | None = None
        self.variant_options: list[tuple[str, Type[BaseModel] | None]] = []
        self.variant_selected_index = 0
        self.variant_area = None  # TextArea for variant display

        # UI components
        self.search_box = TextArea(
            height=SEARCH_HEIGHT,
            prompt="search> ",
            multiline=False
        )
        self.status_bar = TextArea(height=STATUS_HEIGHT, focusable=False)

        self.list_control = FormattedTextControl(
            text=self._render_list,
            focusable=True
        )
        self.preview_control = FormattedTextControl(
            text=self._render_preview,
            focusable=False
        )

        self.list_window = Window(content=self.list_control, always_hide_cursor=True)
        self.preview_window = Window(content=self.preview_control, always_hide_cursor=True)

        # Edit dialog
        self.edit_area = TextArea(multiline=False, height=EDIT_AREA_HEIGHT)
        self.edit_dialog = Dialog(
            title="Edit value",
            body=HSplit([self.edit_area]),
            buttons=[
                Button(text="Save", handler=self._on_edit_save),
                Button(text="Cancel", handler=self._on_edit_cancel),
            ],
            width=EDIT_DIALOG_WIDTH,
        )

        # Variant selection dialog (will be recreated dynamically)
        self.variant_dialog = None  # Will be created when needed

        # Create a container that will hold the dynamic variant dialog
        from prompt_toolkit.layout.containers import DynamicContainer

        def get_variant_dialog():
            return self.variant_dialog if self.variant_dialog else self.edit_dialog

        self.variant_dialog_container = DynamicContainer(get_variant_dialog)

        # Layout
        body = HSplit([
            Frame(self.search_box, title=f"Stryx Config Editor — {schema.__name__} → {file_path}"),
            VSplit([
                Frame(self.list_window, title="Fields"),
                Frame(self.preview_window, title="Preview"),
            ], padding=1),
            self.status_bar,
        ])

        root = FloatContainer(
            content=body,
            floats=[
                Float(
                    content=ConditionalContainer(
                        self.edit_dialog,
                        filter=Condition(lambda: self.edit_visible),
                    ),
                    top=2,
                    left=2,
                    right=2,
                ),
                Float(
                    content=ConditionalContainer(
                        self.variant_dialog_container,
                        filter=Condition(lambda: self.variant_visible),
                    ),
                    top=2,
                    left=2,
                    right=2,
                ),
            ],
        )

        # Application
        self.app = Application(
            layout=Layout(root),
            key_bindings=self._create_keybindings(),
            full_screen=True,
            style=Style.from_dict({
                "frame.border": "fg:#888888",
                "selected": "bg:#444444 fg:#ffffff",
                "status": "bg:#222222 fg:#dddddd",
                "dialog": "bg:#1e1e1e",
                "dialog.body": "bg:#2d2d2d",
                "dialog frame.label": "fg:#ffffff",
                "dialog.body text-area": "bg:#2d2d2d fg:#ffffff",
                "button": "bg:#444444 fg:#ffffff",
                "button.focused": "bg:#666666 fg:#ffffff",
            }),
        )

        # Initialize
        self._refresh_filter()
        self.search_box.buffer.on_text_changed += lambda _: self._on_search_change()

    def _set_status(self, message: str) -> None:
        """Update the status bar with a message."""
        self.status_bar.text = message

    def _on_search_change(self) -> None:
        """Handle search query changes."""
        self.query = self.search_box.text
        self.selected_index = 0
        self._refresh_filter()

    def _refresh_filter(self) -> None:
        """Recompute filtered items based on current query using fuzzy matching."""
        # Get all actual data fields (not schema possibilities)
        all_items = self.config.flatten_scalars()
        query = self.query.strip()

        if query:
            # Create searchable labels for all items
            items_with_labels = [
                (f"{path_to_str(path)} = {val}", path, val)
                for path, val in all_items
            ]

            # Use rapidfuzz to find and rank matches
            # WRatio is good for general fuzzy matching with good performance
            matches = process.extract(
                query,
                [label for label, _, _ in items_with_labels],
                scorer=fuzz.WRatio,
                score_cutoff=FUZZY_SCORE_CUTOFF,
                limit=None  # Get all matches above cutoff
            )

            # Build filtered list sorted by score (descending, already sorted by rapidfuzz)
            # Filter by stricter threshold for better quality results
            self.filtered_items = [
                (items_with_labels[idx][1], items_with_labels[idx][2])  # (path, val)
                for label, score, idx in matches
                if score >= FUZZY_MATCH_THRESHOLD
            ]
        else:
            self.filtered_items = all_items

        # Clamp selection
        if self.filtered_items:
            self.selected_index = min(self.selected_index, len(self.filtered_items) - 1)
        else:
            self.selected_index = 0

        self._update_status_message()
        self.app.invalidate()

    def _update_status_message(self) -> None:
        """Update status bar with item count, validation status, and help text."""
        total = len(self.config.flatten_scalars())
        filtered = len(self.filtered_items)

        # Check validation status
        is_valid, _ = self.config.validate()
        validation_str = "✓ valid" if is_valid else "✗ invalid"

        self._set_status(
            f"{filtered}/{total} items | {validation_str} | "
            "Enter=edit | Space=toggle bool | Ctrl+S=save | Ctrl+Q=quit"
        )

    def _render_list(self) -> FormattedText:
        """Render the list of filtered config fields."""
        if not self.filtered_items:
            return FormattedText([("", "No matches\n")])

        lines = []
        # Viewport windowing for performance with large configs
        start = max(0, self.selected_index - LIST_VIEWPORT_BEFORE)
        end = min(len(self.filtered_items), start + LIST_VIEWPORT_SIZE)

        for i in range(start, end):
            path, value = self.filtered_items[i]
            path_str = path_to_str(path)
            value_str = str(value)[:VALUE_DISPLAY_LENGTH]
            text = f"{path_str:<{FIELD_DISPLAY_WIDTH}}  {value_str}\n"
            style = "class:selected" if i == self.selected_index else ""
            lines.append((style, text))

        return FormattedText(lines)

    def _render_preview(self) -> FormattedText:
        """Render a preview of the selected field's value."""
        if not self.filtered_items:
            return FormattedText([("", "")])

        path, current_value = self.filtered_items[self.selected_index]
        preview_yaml = yaml.safe_dump(
            current_value,
            sort_keys=False,
            explicit_end=False,
            default_flow_style=False
        ).rstrip()

        return FormattedText([("", f"{path_to_str(path)}\n\n{preview_yaml}")])

    def _move_selection(self, delta: int) -> None:
        """Move selection up or down by delta."""
        if not self.filtered_items or self.edit_visible or self.variant_visible:
            return

        self.selected_index = max(
            0,
            min(len(self.filtered_items) - 1, self.selected_index + delta)
        )
        self.app.invalidate()

    def _toggle_bool(self) -> None:
        """Toggle boolean value at current selection."""
        if not self.filtered_items or self.edit_visible or self.variant_visible:
            return

        path, current = self.filtered_items[self.selected_index]

        if isinstance(current, bool):
            self.config.set_at(path, not current)
            self._refresh_filter()

    def _open_editor(self) -> None:
        """Open the edit dialog or variant selector for the selected field."""
        if not self.filtered_items or self.edit_visible or self.variant_visible:
            return

        path, current_value = self.filtered_items[self.selected_index]

        # Check if this field is a discriminator field (e.g., sched.kind)
        is_discriminator, parent_path = self.config.is_discriminator_field(path)
        if is_discriminator and parent_path is not None:
            # This is a discriminator field - show variant selector for the parent union
            variants = self.config.get_union_variants(parent_path)
            if variants:
                self._open_variant_selector(parent_path, variants)
                return

        # Check if this is a discriminated union field (e.g., sched when it's null)
        variants = self.config.get_union_variants(path)
        if variants:
            # Show variant selector instead of text editor
            self._open_variant_selector(path, variants)
            return

        # Regular text editing for non-union fields
        # Format value for editing based on type
        if isinstance(current_value, str):
            default_text = current_value
        elif isinstance(current_value, (int, float, bool)) or current_value is None:
            # Simple scalars: use string representation (avoids YAML markers)
            default_text = str(current_value)
        else:
            # Complex types (dict, list): use YAML without document end marker
            default_text = yaml.safe_dump(
                current_value,
                sort_keys=False,
                explicit_end=False,
                default_flow_style=False
            ).strip()

        self.edit_area.text = default_text
        # Move cursor to end of text for easier editing
        self.edit_area.buffer.cursor_position = len(default_text)

        self.edit_path = path
        self.edit_visible = True
        self.app.layout.focus(self.edit_area)

    def _on_edit_save(self) -> None:
        """Save the edited value and close the dialog."""
        if self.edit_path is None:
            self._on_edit_cancel()
            return

        path = self.edit_path
        current = self.config.get_at(path)
        new_value = parse_like_yaml(self.edit_area.text, current)

        self.config.set_at(path, new_value)
        self._close_editor()
        self._refresh_filter()
        self._set_status(f"Updated {path_to_str(path)}")

    def _on_edit_cancel(self) -> None:
        """Cancel editing and close the dialog."""
        self._close_editor()

    def _close_editor(self) -> None:
        """Close the edit dialog and return focus to search."""
        self.edit_visible = False
        self.edit_path = None
        self.app.layout.focus(self.search_box)
        self.app.invalidate()

    def _open_variant_selector(self, path: FieldPath, variants: list[tuple[str, Type[BaseModel] | None]]) -> None:
        """Open the variant selection dialog for a discriminated union field."""
        self.variant_path = path
        self.variant_options = variants
        self.variant_selected_index = 0  # Track which variant is selected

        # Build text for the selection menu
        lines = [f"Select variant for {path_to_str(path)}:\n"]
        for i, (variant_name, _) in enumerate(variants):
            prefix = "> " if i == self.variant_selected_index else "  "
            lines.append(f"{prefix}{variant_name}")

        # Create a simple text area showing the options
        self.variant_area = TextArea(
            text="\n".join(lines),
            multiline=True,
            focusable=True,
            read_only=True,
            scrollbar=True,
        )

        # Create new dialog with the text area
        self.variant_dialog = Frame(
            body=self.variant_area,
            title="Select Variant (↑↓ to navigate, Enter to select, Esc to cancel)",
        )

        self.variant_visible = True
        self.app.invalidate()
        self.app.layout.focus(self.variant_area)

    def _update_variant_display(self) -> None:
        """Update the variant selection display."""
        if not self.variant_visible or not self.variant_area or not self.variant_path:
            return

        # Rebuild the text with current selection
        lines = [f"Select variant for {path_to_str(self.variant_path)}:\n"]
        for i, (variant_name, _) in enumerate(self.variant_options):
            prefix = "> " if i == self.variant_selected_index else "  "
            lines.append(f"{prefix}{variant_name}")

        self.variant_area.text = "\n".join(lines)

    def _move_variant_selection(self, delta: int) -> None:
        """Move variant selection up or down."""
        if not self.variant_visible or not self.variant_options:
            return

        self.variant_selected_index = max(
            0,
            min(len(self.variant_options) - 1, self.variant_selected_index + delta)
        )
        self._update_variant_display()

    def _confirm_variant_selection(self) -> None:
        """Confirm the current variant selection."""
        if not self.variant_visible or not self.variant_path or not self.variant_options:
            return

        path = self.variant_path
        _, variant_type = self.variant_options[self.variant_selected_index]

        # Set the new variant value
        if variant_type is None:
            # User selected "None"
            self.config.set_at(path, None)
        else:
            # Instantiate the variant with defaults
            instance = variant_type()
            self.config.set_at(path, instance.model_dump(mode="python"))

        self._close_variant_selector()
        self._refresh_filter()
        self._set_status(f"Changed {path_to_str(path)} to {variant_type.__name__ if variant_type else 'None'}")

    def _on_variant_cancel(self) -> None:
        """Cancel variant selection."""
        self._close_variant_selector()

    def _close_variant_selector(self) -> None:
        """Close the variant selector dialog."""
        self.variant_visible = False
        self.variant_path = None
        self.variant_options = []
        self.variant_selected_index = 0
        self.variant_dialog = None  # Clear the dialog
        self.variant_area = None  # Clear the text area
        self.app.layout.focus(self.search_box)
        self.app.invalidate()

    def _save_config(self) -> None:
        """Save configuration to disk with validation."""
        if not self.edit_visible:
            try:
                self.config.save()
                self._set_status("Saved successfully!")
                self._refresh_filter()  # Refresh to update validation status
            except ValueError as e:
                # Validation error - the error message is already formatted nicely
                self._set_status(f"Save failed: {str(e)}")

    def _clear_search(self) -> None:
        """Clear the search box."""
        self.search_box.text = ""
        self.search_box.buffer.cursor_position = 0
        self._on_search_change()

    def _create_keybindings(self) -> KeyBindings:
        """Create and configure all keybindings."""
        kb = KeyBindings()

        @kb.add("c-q")
        def _quit(event):
            event.app.exit()

        @kb.add("c-s")
        def _save(event):
            self._save_config()

        @kb.add("down")
        def _down(event):
            if self.variant_visible:
                self._move_variant_selection(1)
            else:
                self._move_selection(1)

        @kb.add("up")
        def _up(event):
            if self.variant_visible:
                self._move_variant_selection(-1)
            else:
                self._move_selection(-1)

        @kb.add("pagedown")
        def _pagedown(event):
            self._move_selection(PAGE_JUMP_SIZE)

        @kb.add("pageup")
        def _pageup(event):
            self._move_selection(-PAGE_JUMP_SIZE)

        @kb.add("enter")
        def _enter(event):
            if self.edit_visible:
                self._on_edit_save()
            elif self.variant_visible:
                self._confirm_variant_selection()
            else:
                self._open_editor()

        @kb.add("escape")
        def _escape(event):
            if self.edit_visible:
                self._on_edit_cancel()
            elif self.variant_visible:
                self._on_variant_cancel()
            else:
                self._clear_search()

        @kb.add(" ")
        def _space(event):
            self._toggle_bool()

        return kb

    def run(self) -> None:
        """Start the TUI application."""
        self.app.layout.focus(self.search_box)
        self.app.run()


def launch_tui(
    schema: Type[BaseModel],
    config_name: str,
    config_dir: str = "config"
) -> None:
    """Launch the interactive TUI for editing a Pydantic config.

    Args:
        schema: Pydantic BaseModel class to use for validation
        config_name: Name of the config file (without extension)
        config_dir: Directory where config will be saved (default: "config")
    """
    config_path = Path(config_dir) / f"{config_name}.yaml"
    tui = PydanticConfigTUI(schema, config_path)
    tui.run()
