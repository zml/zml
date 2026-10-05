#!/usr/bin/env python3
import sys
import curses
import json
from safetensors import safe_open


def load_safetensors_headers(filepath):
    """Read metadata and key shapes directly from SafeTensors header."""
    tensors = []
    metadata = {}

    with safe_open(filepath, framework="pt", device="cpu") as f:
        # Load user metadata if present
        metadata = f.metadata() or {}

        # Load keys, shapes, and dtypes
        for key in f.keys():
            slice_obj = f.get_slice(key)
            tensors.append(
                {
                    "key": key,
                    "shape": tuple(slice_obj.get_shape()),
                    "dtype": str(slice_obj.get_dtype()),
                }
            )

    # Sort keys alphabetically by default
    tensors.sort(key=lambda x: x["key"])
    return metadata, tensors


def main(stdscr, filepath):
    # Setup curses settings
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()

    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)  # Header bar
    curses.init_pair(2, curses.COLOR_YELLOW, -1)  # Shapes & metadata
    curses.init_pair(3, curses.COLOR_GREEN, -1)  # Tensor keys
    curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Selection
    curses.init_pair(5, curses.COLOR_MAGENTA, -1)  # Dtypes

    try:
        metadata, all_tensors = load_safetensors_headers(filepath)
    except Exception as e:
        stdscr.addstr(0, 0, f"Error loading file: {e}")
        stdscr.refresh()
        stdscr.getch()
        return

    filter_text = ""
    selected_idx = 0
    top_idx = 0

    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        # Apply search filter
        filtered_tensors = [
            t for t in all_tensors if filter_text.lower() in t["key"].lower()
        ]

        if selected_idx >= len(filtered_tensors):
            selected_idx = max(0, len(filtered_tensors) - 1)

        # 1. Header Bar
        title = f" SafeTensors Header Viewer | File: {filepath} | Count: {len(filtered_tensors)}/{len(all_tensors)} "
        stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
        stdscr.addstr(0, 0, title[:width].ljust(width))
        stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)

        # 2. Search / Status line
        search_str = f" Filter: '{filter_text}' (Press '/' to search, 'Esc' to clear)"
        stdscr.addstr(1, 0, search_str[:width], curses.color_pair(2))

        # 3. Metadata Section (if present)
        row = 2
        if metadata:
            meta_str = f" Metadata: {json.dumps(metadata)}"
            stdscr.addstr(row, 0, meta_str[: width - 1], curses.color_pair(2))
            row += 1

        stdscr.addstr(row, 0, "-" * (width - 1))
        row += 1

        # 4. Compute viewport dimensions
        header_lines = row
        footer_lines = 1
        viewport_height = height - header_lines - footer_lines

        # Keep selected item visible
        if selected_idx < top_idx:
            top_idx = selected_idx
        elif selected_idx >= top_idx + viewport_height:
            top_idx = selected_idx - viewport_height + 1

        # 5. Render Tensor Rows
        for i in range(viewport_height):
            item_idx = top_idx + i
            if item_idx >= len(filtered_tensors):
                break

            tensor = filtered_tensors[item_idx]
            current_row = header_lines + i

            key_str = tensor["key"]
            shape_str = str(tensor["shape"])
            dtype_str = tensor["dtype"]

            # Layout spacing
            max_key_len = max(20, width - 40)
            formatted_line = f" {key_str:<{max_key_len}}  {shape_str:<20}  {dtype_str}"

            if item_idx == selected_idx:
                stdscr.attron(curses.color_pair(4) | curses.A_BOLD)
                stdscr.addstr(
                    current_row, 0, formatted_line[: width - 1].ljust(width - 1)
                )
                stdscr.attroff(curses.color_pair(4) | curses.A_BOLD)
            else:
                # Colored non-selected output
                stdscr.addstr(
                    current_row, 1, key_str[:max_key_len], curses.color_pair(3)
                )
                if len(key_str) + 3 < width:
                    stdscr.addstr(
                        current_row, max_key_len + 3, shape_str, curses.color_pair(2)
                    )
                if len(key_str) + 25 < width:
                    stdscr.addstr(
                        current_row, max_key_len + 25, dtype_str, curses.color_pair(5)
                    )

        # 6. Bottom Instruction Bar
        instructions = (
            " [Up/Down] Navigate | [/] Search | [ESC] Clear Search | [Q] Quit "
        )
        stdscr.attron(curses.color_pair(1))
        # AFTER (Safe drawing for terminal boundaries)
        try:
            stdscr.addstr(height - 1, 0, instructions[: width - 1])
        except curses.error:
            pass
        stdscr.attroff(curses.color_pair(1))
        stdscr.refresh()

        # Handle user input
        key = stdscr.getch()

        if key in (ord("q"), ord("Q")):
            break
        elif key in (curses.KEY_UP, ord("k")):
            selected_idx = max(0, selected_idx - 1)
        elif key in (curses.KEY_DOWN, ord("j")):
            selected_idx = min(len(filtered_tensors) - 1, selected_idx + 1)
        elif key in (curses.KEY_PPAGE,):  # Page Up
            selected_idx = max(0, selected_idx - viewport_height)
        elif key in (curses.KEY_NPAGE,):  # Page Down
            selected_idx = min(
                len(filtered_tensors) - 1, selected_idx + viewport_height
            )
        elif key == 27:  # ESC key clears filter
            filter_text = ""
            selected_idx = 0
        elif key in (ord("/"),):
            # Prompt for inline filtering
            curses.echo()
            curses.curs_set(1)
            stdscr.addstr(1, 0, " Filter: ".ljust(width), curses.color_pair(4))
            stdscr.refresh()

            filter_text = stdscr.getstr(1, 9, 50).decode("utf-8")
            curses.noecho()
            curses.curs_set(0)
            selected_idx = 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python safetensors_viewer.py <path_to_safetensors_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    curses.wrapper(main, filepath)
