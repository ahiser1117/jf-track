from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog


@dataclass
class PromptResult:
    """Container for GUI prompt selections."""

    video_path: str
    is_rotating: bool
    use_feature_sampling: bool
    num_mouths: int
    num_gonads: int
    num_tentacle_bulbs: int | None
    use_custom_roi: bool
    roi_shape: str
    max_frames: int | None
    use_auto_threshold: bool
    manual_threshold: int | None


class TrackingPromptGUI:
    """Tkinter-based wizard for collecting tracking configuration."""

    def __init__(self) -> None:
        self._root = tk.Tk()
        self._root.withdraw()
        self._root.update()

    def _ask_yes_no(self, title: str, question: str, default: bool = False) -> bool:
        result = messagebox.askyesno(title, question, parent=self._root, default="yes" if default else "no")
        return bool(result)

    def _ask_required_path(self) -> str:
        while True:
            path = filedialog.askopenfilename(
                title="Select Jellyfish Video",
                filetypes=[("Video Files", "*.avi *.mp4 *.mov *.mkv"), ("All Files", "*.*")],
                parent=self._root,
            )
            if path:
                return path
            retry = messagebox.askretrycancel("Video Required", "A video is required to continue.")
            if not retry:
                raise RuntimeError("Video selection cancelled")

    def _ask_count(
        self,
        title: str,
        prompt_text: str,
        default: int | None,
        min_value: int = 0,
        max_value: int | None = None,
        allow_blank: bool = False,
    ) -> int | None:
        initial = "" if default is None else str(default)
        while True:
            value = simpledialog.askstring(title, prompt_text, initialvalue=initial, parent=self._root)
            if value is None:
                raise RuntimeError("User cancelled while entering counts")
            stripped = value.strip()
            if stripped == "":
                if allow_blank:
                    return None
                messagebox.showerror(title, "A numeric value is required.")
                continue
            try:
                number = int(stripped)
            except ValueError:
                messagebox.showerror(title, "Please enter a whole number.")
                continue
            if number < min_value:
                messagebox.showerror(title, f"Value must be >= {min_value}.")
                continue
            if max_value is not None and number > max_value:
                messagebox.showerror(title, f"Value must be <= {max_value}.")
                continue
            return number

    def _ask_optional_positive_int(
        self,
        title: str,
        prompt_text: str,
    ) -> int | None:
        while True:
            value = simpledialog.askstring(title, prompt_text, initialvalue="", parent=self._root)
            if value is None:
                raise RuntimeError("User cancelled while entering frame limit")
            stripped = value.strip()
            if stripped == "":
                return None
            try:
                number = int(stripped)
            except ValueError:
                messagebox.showerror(title, "Please enter a whole number or leave blank.")
                continue
            if number <= 0:
                messagebox.showerror(title, "Frame count must be greater than zero.")
                continue
            return number

    def _ask_roi_shape(self) -> str:
        """Prompt the user for an ROI shape string."""

        valid = {"circle", "polygon", "bounding_box"}
        while True:
            value = simpledialog.askstring(
                "ROI Shape",
                "Enter ROI shape (circle, polygon, bounding_box)",
                initialvalue="circle",
                parent=self._root,
            )
            if value is None:
                raise RuntimeError("User cancelled while selecting ROI shape")
            shape = value.strip().lower()
            if shape in valid:
                return shape
            messagebox.showerror("ROI Shape", "Please enter circle, polygon, or bounding_box.")

    def collect_inputs(self) -> PromptResult:
        try:
            video_path = self._ask_required_path()
            is_rotating = self._ask_yes_no("Video Rotation", "Is this video rotating?", default=False)
            use_sampling = self._ask_yes_no(
                "Parameter Optimization",
                "Do you want to select the features for parameter optimization?",
                default=False,
            )

            num_mouths = self._ask_count(
                "Mouth Count",
                "How many mouths should be tracked?",
                default=1,
                min_value=1,
            )
            num_gonads = self._ask_count(
                "Gonad Count",
                "How many gonads are visible? (0-4)",
                default=0,
                min_value=0,
                max_value=4,
            )
            bulbs_prompt = (
                "How many tentacle bulbs should be tracked?\n"
                "Leave blank to auto-detect as many as possible."
            )
            num_tentacle_bulbs = self._ask_count(
                "Tentacle Bulbs",
                bulbs_prompt,
                default=None,
                min_value=0,
                allow_blank=True,
            )

            use_custom_roi = self._ask_yes_no(
                "Search ROI",
                "Do you want to define a custom search region?",
                default=is_rotating,
            )
            roi_shape = "circle"
            if use_custom_roi:
                roi_shape = self._ask_roi_shape()

            max_frames = self._ask_optional_positive_int(
                "Frame Limit",
                "How many frames should be processed?\nLeave blank to use the entire video.",
            )

            use_auto_threshold = self._ask_yes_no(
                "Threshold",
                "Use automatic per-video threshold?",
                default=True,
            )
            manual_threshold = None
            if not use_auto_threshold:
                manual_threshold = self._ask_optional_positive_int(
                    "Manual Threshold",
                    "Enter a positive threshold value (pixels).",
                )
                if manual_threshold is None:
                    use_auto_threshold = True

            return PromptResult(
                video_path=video_path,
                is_rotating=is_rotating,
                use_feature_sampling=use_sampling,
                num_mouths=int(num_mouths or 1),
                num_gonads=int(num_gonads or 0),
                num_tentacle_bulbs=None if num_tentacle_bulbs is None else int(num_tentacle_bulbs),
                use_custom_roi=use_custom_roi,
                roi_shape=roi_shape,
                max_frames=max_frames,
                use_auto_threshold=use_auto_threshold,
                manual_threshold=manual_threshold,
            )
        finally:
            self._root.destroy()


def prompt_for_tracking_configuration() -> PromptResult:
    """Convenience helper to collect GUI prompts."""

    gui = TrackingPromptGUI()
    return gui.collect_inputs()
