from __future__ import annotations

from pathlib import Path

import click

from src.bounding_box_selector import run_bounding_box_selection, run_circle_roi_selection
from src.roi_selector import run_interactive_roi_selection
from src.gui_prompts import prompt_for_tracking_configuration, PromptResult
from src.interactive_sampling import run_interactive_feature_sampling
from src.save_results import save_multi_object_tracking_to_zarr
from src.visualizations import save_multi_object_labeled_video
from src.tracker import TrackingParameters
from src.tracking import run_multi_object_tracking


DEFAULT_OUTPUT_DIR = Path("./tracking_results")


def _merge_parameter_configs(
    base: TrackingParameters,
    tuned: TrackingParameters,
) -> None:
    """Merge object type configuration values from tuned into base."""

    allowed_keys = {
        "min_area",
        "max_area",
        "max_disappeared",
        "max_distance",
        "search_radius",
        "aspect_ratio_min",
        "aspect_ratio_max",
        "eccentricity_min",
        "eccentricity_max",
        "solidity_min",
        "solidity_max",
    }

    for obj_type, config in tuned.object_types.items():
        if obj_type not in base.object_types:
            continue
        for key, value in config.items():
            if key not in allowed_keys:
                continue
            if value is None:
                continue
            base.object_types[obj_type][key] = value


def _initialize_parameters(prompt: PromptResult) -> TrackingParameters:
    params = TrackingParameters()
    params.num_mouths = max(1, prompt.num_mouths)
    params.num_gonads = max(0, prompt.num_gonads)
    params.num_tentacle_bulbs = prompt.num_tentacle_bulbs
    params.update_object_counts()

    params.video_type = "rotating" if prompt.is_rotating else "non_rotating"
    params.adaptive_background = prompt.is_rotating

    params.roi_mode = "auto"
    params.roi_center = None
    params.roi_radius = None
    params.roi_points = []
    params.roi_bbox = None
    params.use_auto_threshold = prompt.use_auto_threshold
    if not prompt.use_auto_threshold and prompt.manual_threshold is not None:
        params.threshold = max(1, prompt.manual_threshold)

    return params


def _apply_bounding_box_roi(
    params: TrackingParameters,
    video_path: str,
    max_frames: int | None,
) -> None:
    click.echo("Select a bounding box on the median intensity projection window.")
    roi_config = run_bounding_box_selection(video_path, max_frames=max_frames)
    bbox = roi_config.get("bbox")
    if bbox is None:
        raise click.ClickException("Bounding box selection did not return coordinates")

    params.apply_roi_config(roi_config)


def _apply_circle_roi(
    params: TrackingParameters,
    video_path: str,
    max_frames: int | None,
) -> None:
    click.echo("Select a circular ROI on the median intensity projection window.")
    roi_config = run_circle_roi_selection(video_path, max_frames=max_frames)
    center = roi_config.get("center")
    radius = roi_config.get("radius")
    if center is None or radius is None:
        raise click.ClickException("Circle selection did not return center/radius")

    params.apply_roi_config(roi_config)


def _apply_polygon_roi(
    params: TrackingParameters,
    video_path: str,
    max_frames: int | None,
) -> None:
    click.echo("Draw a polygonal ROI on the median intensity projection window.")
    roi_config = run_interactive_roi_selection(video_path)
    if roi_config.get("mode") != "polygon" or not roi_config.get("points"):
        raise click.ClickException("Polygon selection failed")

    params.apply_roi_config(roi_config)


def _prepare_tracking_parameters(prompt: PromptResult) -> TrackingParameters:
    params = _initialize_parameters(prompt)

    if prompt.use_feature_sampling:
        click.echo("Launching interactive feature sampling for parameter tuning...")
        try:
            tuned_params = run_interactive_feature_sampling(
                prompt.video_path,
                params,
                apply_roi_mask=False,
            )
        except Exception as exc:  # pragma: no cover - interactive UI failure
            click.echo(f"Feature sampling failed: {exc}")
        else:
            _merge_parameter_configs(params, tuned_params)

    if prompt.use_custom_roi:
        try:
            shape = (prompt.roi_shape or "circle").lower()
            if shape == "circle":
                _apply_circle_roi(params, prompt.video_path, prompt.max_frames)
            elif shape == "polygon":
                _apply_polygon_roi(params, prompt.video_path, prompt.max_frames)
            else:
                _apply_bounding_box_roi(params, prompt.video_path, prompt.max_frames)
        except Exception as exc:  # pragma: no cover - user interaction path
            raise click.ClickException(f"ROI selection failed: {exc}") from exc
    else:
        params.clear_roi()

    params.update_object_counts()
    return params


def _ensure_output_dir() -> Path:
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_prompted_tracking() -> None:
    click.echo("jf-track Prompt Workflow")
    click.echo("==========================")

    try:
        prompt = prompt_for_tracking_configuration()
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    params = _prepare_tracking_parameters(prompt)

    frame_msg = (
        f" (max_frames={prompt.max_frames})" if prompt.max_frames is not None else ""
    )
    click.echo(f"Running tracking with GUI-selected parameters{frame_msg}...")
    tracking_results, fps = run_multi_object_tracking(
        prompt.video_path,
        params,
        max_frames=prompt.max_frames,
    )

    output_dir = _ensure_output_dir()
    zarr_path = output_dir / "multi_object_tracking.zarr"
    save_multi_object_tracking_to_zarr(tracking_results, str(zarr_path), fps, params)

    labeled_video_path = output_dir / "multi_object_labeled.mp4"
    composite_video_path = output_dir / "multi_object_labeled_composite.mp4"
    click.echo(
        f"Rendering labeled videos to {labeled_video_path} and {composite_video_path}..."
    )
    save_multi_object_labeled_video(
        video_path=prompt.video_path,
        zarr_path=str(zarr_path),
        output_path=str(labeled_video_path),
        background_mode="original",
        composite_output_path=str(composite_video_path),
        max_frames=prompt.max_frames,
    )

    click.echo("Tracking completed successfully.")
    click.echo(f"Results saved to {zarr_path}")
    click.echo(f"Labeled video saved to {labeled_video_path}")
    click.echo(f"Composite visualization saved to {composite_video_path}")


@click.command()
def track_jellyfish() -> None:
    """Entry point: launch the prompt-driven tracking workflow."""

    run_prompted_tracking()


if __name__ == "__main__":
    track_jellyfish()
