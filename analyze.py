from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from src.analysis.pulse_analysis import (
    PulseAnalysisResult,
    compute_pulse_distance_series,
    save_pulse_csv,
    save_pulse_plot,
)


RESULT_ZARR_NAME = "multi_object_tracking.zarr"


def _resolve_results(target_path: str | Path) -> tuple[Path, Path]:
    path = Path(target_path).expanduser().resolve()

    if path.is_file():
        if path.suffix == ".zarr":
            zarr_path = path
            results_dir = path.parent
        else:
            results_dir = path.parent / f"{path.stem}_results"
            zarr_path = results_dir / RESULT_ZARR_NAME
    else:
        candidate = path / RESULT_ZARR_NAME
        if candidate.exists():
            zarr_path = candidate
            results_dir = path
        else:
            # Maybe the user passed the parent directory, so look for <dir_name>_results
            alt_dir = path / f"{path.name}_results"
            candidate = alt_dir / RESULT_ZARR_NAME
            if candidate.exists():
                zarr_path = candidate
                results_dir = alt_dir
            else:
                raise click.ClickException(
                    "Could not locate multi_object_tracking.zarr. "
                    "Pass a video file (with completed results) or the results directory."
                )

    if not zarr_path.exists():
        raise click.ClickException(
            f"Expected results at {zarr_path}, but the file does not exist. Run tracking first."
        )

    return zarr_path, results_dir


def _print_summary(result: PulseAnalysisResult) -> None:
    valid = result.mean_distance_px[~np.isnan(result.mean_distance_px)]
    if valid.size:
        click.echo(
            f"Mean distance: {np.nanmean(valid):.2f} px | "
            f"Min: {np.nanmin(valid):.2f} px | Max: {np.nanmax(valid):.2f} px"
        )
    else:
        click.echo("No valid pulse distances were computed.")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("target", type=click.Path(exists=True))
@click.option(
    "--object-type",
    "-o",
    default="tentacle_bulb",
    show_default=True,
    help="Tracking object type to analyze",
)
@click.option(
    "--no-plot",
    is_flag=True,
    help="Skip saving pulse_distance.png (still shown if --show).",
)
@click.option(
    "--no-csv",
    is_flag=True,
    help="Skip saving pulse_distance.csv.",
)
@click.option("--show/--no-show", default=False, help="Display the plot interactively.")
def main(
    target: str,
    object_type: str,
    no_plot: bool,
    no_csv: bool,
    show: bool,
) -> None:
    """Analyze jellyfish pulse distance from a video or results directory."""

    zarr_path, results_dir = _resolve_results(target)
    click.echo(f"Using results from {zarr_path}")

    result = compute_pulse_distance_series(zarr_path, object_type=object_type)
    if np.all(np.isnan(result.mean_distance_px)):
        raise click.ClickException(
            f"No valid detections available for object type '{object_type}'."
        )

    plot_path = None if no_plot else results_dir / "pulse_distance.png"
    csv_path = None if no_csv else results_dir / "pulse_distance.csv"

    if plot_path is not None or show:
        save_pulse_plot(result, output_path=plot_path, show=show)

    if csv_path is not None:
        save_pulse_csv(result, csv_path)

    if (plot_path is None and not show) and csv_path is None:
        _print_summary(result)


if __name__ == "__main__":
    main()
