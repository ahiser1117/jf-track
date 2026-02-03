#!/usr/bin/env python3
"""Utility commands for exercising the jf-track multi-object workflow."""

from __future__ import annotations

import click
from pathlib import Path

try:  # pragma: no cover - normal execution inside package
    from src.cli import track_jellyfish
    from src.interactive_sampling import run_interactive_feature_sampling
    from src.roi_selector import run_interactive_roi_selection
    from src.tracker import TrackingParameters
    from src.tracking import run_multi_object_tracking
    from src.save_results import save_multi_object_tracking_to_zarr
    from src.visualizations import save_multi_object_labeled_video
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from src.cli import track_jellyfish
    from src.interactive_sampling import run_interactive_feature_sampling
    from src.roi_selector import run_interactive_roi_selection
    from src.tracker import TrackingParameters
    from src.tracking import run_multi_object_tracking
    from src.save_results import save_multi_object_tracking_to_zarr
    from src.visualizations import save_multi_object_labeled_video


def _run_cli(args: list[str]) -> None:
    """Invoke the Click CLI with the provided arguments."""
    try:
        track_jellyfish.main(args, standalone_mode=False)
    except SystemExit as exc:  # pragma: no cover - Click uses SystemExit
        if exc.code != 0:
            raise


@click.group()
def test() -> None:
    """Collection of helper commands for the multi-object tracker."""


@test.command()
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--output', default='./test_output', help='Directory for CLI outputs')
@click.option('--max-frames', type=int, default=100, help='Limit frames for faster testing')
def interactive(video_path: str, output: str, max_frames: int) -> None:
    """Launch the CLI in fully interactive mode."""
    click.echo("Launching interactive CLI workflow. Follow on-screen prompts.")
    args = [
        video_path,
        '--output', output,
        '--max-frames', str(max_frames),
        '--interactive'
    ]
    _run_cli(args)


@test.command()
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--output', default='./test_output', help='Directory for CLI outputs')
@click.option('--max-frames', type=int, default=100, help='Limit frames for faster testing')
def batch(video_path: str, output: str, max_frames: int) -> None:
    """Run the CLI with preset object counts (non-interactive)."""
    click.echo("Running batch-mode CLI with sample configuration...")
    args = [
        video_path,
        '--output', output,
        '--max-frames', str(max_frames),
        '--video-type', 'rotating',
        '--mouths', '1',
        '--gonads', '2',
        '--bulbs', '8'
    ]
    _run_cli(args)


@test.command()
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False))
def sampling(video_path: str) -> None:
    """Launch the interactive feature sampler for heuristic tuning."""
    click.echo("Launching interactive feature sampler (press 'q' to finish)...")
    run_interactive_feature_sampling(video_path, TrackingParameters())


@test.command()
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False))
def roi(video_path: str) -> None:
    """Launch the interactive ROI selector dashboard."""
    click.echo("Launching interactive ROI selector (press 'q' to finish)...")
    run_interactive_roi_selection(video_path)


@test.command()
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--output', default='./test_visualization.mp4', help='Output annotated video path')
@click.option('--max-frames', type=int, default=150, help='Limit frames for quicker visualization')
def visualization(video_path: str, output: str, max_frames: int) -> None:
    """Run a short tracking session and render a labeled video."""
    click.echo("Running short tracking session for visualization...")
    params = TrackingParameters()
    setattr(params, "video_type", "rotating")
    params.num_gonads = 2
    params.num_tentacle_bulbs = 6
    params.update_object_counts()

    tracking_results, fps = run_multi_object_tracking(video_path, params, max_frames)
    temp_zarr = Path(output).with_suffix('.zarr')
    save_multi_object_tracking_to_zarr(tracking_results, str(temp_zarr), fps, params)

    click.echo("Rendering labeled video...")
    save_multi_object_labeled_video(
        video_path=video_path,
        zarr_path=str(temp_zarr),
        output_path=output,
        max_frames=max_frames,
        background_mode='original',
        show_search_radii=True
    )

    click.echo(f"Visualization saved to {output}")


if __name__ == '__main__':
    test()
