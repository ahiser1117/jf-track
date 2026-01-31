from __future__ import annotations

import click
from pathlib import Path
from typing import Optional
import sys

try:  # pragma: no cover - normal package execution
    from src.tracker import TrackingParameters
    from src.tracking import run_multi_object_tracking
    from src.save_results import save_multi_object_tracking_to_zarr
    from src.interactive_sampling import run_interactive_feature_sampling
    from src.roi_selector import run_interactive_roi_selection
except ModuleNotFoundError:  # pragma: no cover - fallback for direct invocation
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from src.tracker import TrackingParameters
    from src.tracking import run_multi_object_tracking
    from src.save_results import save_multi_object_tracking_to_zarr
    from src.interactive_sampling import run_interactive_feature_sampling
    from src.roi_selector import run_interactive_roi_selection


def interactive_object_configuration(params: TrackingParameters | None = None) -> TrackingParameters:
    """Interactive prompts for feature counts and detection tuning."""
    click.echo("=== Jellyfish Object Configuration ===")
    
    if params is None:
        params = TrackingParameters()
    
    params.num_mouths = click.prompt(
        'How many mouths should be tracked?',
        type=int,
        default=max(1, params.num_mouths),
    )
    
    # Gonad configuration
    click.echo("\n--- Gonad Configuration ---")
    click.echo("Gonads are oblong structures (aspect ratio >= 1.5).")
    click.echo("Not all animals will have visible gonads.")
    
    if click.confirm('Do you want to detect gonads?'):
        params.num_gonads = click.prompt(
            'How many gonads (0-4)', 
            type=click.IntRange(0, 4), 
            default=0
        )
        
        if params.num_gonads > 0:
            click.echo(f"Will track up to {params.num_gonads} gonads.")
            
            # Advanced gonad parameters
            if click.confirm('Customize gonad detection parameters?'):
                gonad_config = params.object_types["gonad"]
                gonad_config["min_area"] = click.prompt('Min area (pixels)', default=20, type=int)
                gonad_config["max_area"] = click.prompt('Max area (pixels)', default=80, type=int)
                gonad_config["max_disappeared"] = click.prompt('Max disappeared frames', default=15, type=int)
                gonad_config["max_distance"] = click.prompt('Max tracking distance (pixels)', default=40, type=int)
                
                if click.confirm('Customize gonad shape parameters?'):
                    gonad_config["aspect_ratio_min"] = click.prompt('Min aspect ratio', default=1.5, type=float)
                    gonad_config["aspect_ratio_max"] = click.prompt('Max aspect ratio', default=3.0, type=float)
                    gonad_config["eccentricity_min"] = click.prompt('Min eccentricity', default=0.7, type=float)
    else:
        params.num_gonads = 0
    
    # Tentacle bulb configuration
    click.echo("\n--- Tentacle Bulb Configuration ---")
    click.echo("Tentacle bulbs are small, typically round structures.")
    
    if click.confirm('Do you want to detect tentacle bulbs?'):
        count = click.prompt(
            'How many tentacle bulbs', 
            type=int, 
            default=8
        )
        params.num_tentacle_bulbs = count
        
        if count > 0:
            click.echo(f"Will track up to {count} tentacle bulbs.")
            
            # Advanced bulb parameters
            if click.confirm('Customize tentacle bulb parameters?'):
                bulb_config = params.object_types["tentacle_bulb"]
                bulb_config["min_area"] = click.prompt('Min area (pixels)', default=5, type=int)
                bulb_config["max_area"] = click.prompt('Max area (pixels)', default=35, type=int)
                bulb_config["max_disappeared"] = click.prompt('Max disappeared frames', default=10, type=int)
                bulb_config["max_distance"] = click.prompt('Max tracking distance (pixels)', default=30, type=int)
                
                if click.confirm('Customize bulb shape parameters?'):
                    bulb_config["aspect_ratio_max"] = click.prompt('Max aspect ratio (roundness)', default=1.5, type=float)
                    bulb_config["eccentricity_max"] = click.prompt('Max eccentricity', default=0.7, type=float)
    else:
        params.num_tentacle_bulbs = 0
    
    # Update enabled status based on counts
    params.update_object_counts()
    
    params.update_object_counts()
    return params


def interactive_roi_configuration(video_path: str, preferred_mode: Optional[str] = None) -> dict:
    """Prompt user for ROI mode and capture configuration details."""
    click.echo("\n=== ROI Configuration ===")
    
    if preferred_mode is None:
        click.echo("Choose how to define the tracking region:")
        preferred_mode = click.prompt(
            'ROI mode',
            type=click.Choice(['auto', 'circle', 'polygon']),
            default='auto'
        )
    
    if preferred_mode == 'auto':
        click.echo("Using automatic circular ROI (center of frame)")
        return {"mode": "auto"}
    
    click.echo("Launching interactive ROI selector... Press 'q' to finish.")
    roi_config = run_interactive_roi_selection(video_path)
    return roi_config


def apply_roi_config(params: TrackingParameters, roi_config: dict) -> None:
    """Persist ROI information into TrackingParameters."""
    mode = roi_config.get("mode", "auto")
    params.roi_mode = mode
    if mode == "circle":
        center = roi_config.get("center")
        radius = roi_config.get("radius")
        if center is None or radius is None:
            raise click.ClickException("Circle ROI requires a center and radius")
        params.roi_center = (float(center[0]), float(center[1]))
        params.roi_radius = float(radius)
        params.roi_points = []
    elif mode == "polygon":
        points = roi_config.get("points", [])
        if len(points) < 3:
            raise click.ClickException("Polygon ROI requires at least three points")
        params.roi_points = [(float(x), float(y)) for x, y in points]
        params.roi_center = None
        params.roi_radius = None
    else:
        params.roi_center = None
        params.roi_radius = None
        params.roi_points = []


def interactive_sampling_configuration(video_path: str, params: TrackingParameters) -> TrackingParameters:
    """Interactive feature sampling configuration."""
    click.echo("\n=== Feature Sampling ===")
    click.echo("Sample representative objects to set size/shape parameters automatically.")
    
    if click.confirm('Do you want to sample objects for automatic parameter tuning?'):
        click.echo("Starting interactive feature sampling...")
        
        try:
            params = run_interactive_feature_sampling(video_path)
            click.echo("Feature sampling completed. Parameters updated.")
        except Exception as e:
            click.echo(f"Feature sampling failed: {e}")
            if click.confirm('Continue with default parameters?'):
                return params
            else:
                sys.exit(1)
    
    return params


def check_missing_required_params(params: TrackingParameters) -> bool:
    """Check if required parameters are missing and need interactive configuration."""
    
    # Check if any objects are enabled (default behavior)
    enabled_types = params.get_enabled_object_types()
    
    # If no objects enabled, we need interactive configuration
    if not enabled_types:
        return True
    
    # Check if ROI needs to be configured
    if params.roi_mode not in ["auto", "circle", "polygon"]:
        return True
    
    # For now, this is a simple check
    # In a more complex implementation, you might check for other missing parameters
    return False


def run_interactive_configuration(video_path: str) -> TrackingParameters:
    """Run full interactive configuration pipeline."""
    click.echo("jf-track Interactive Configuration")
    click.echo("=" * 40)
    
    params = TrackingParameters()
    video_type_choice = click.prompt(
        'Video type',
        type=click.Choice(['non_rotating', 'rotating']),
        default='non_rotating'
    )
    setattr(params, "video_type", video_type_choice)
    
    # Object configuration
    params = interactive_object_configuration(params)
    
    # Feature sampling
    params = interactive_sampling_configuration(video_path, params)
    
    # ROI configuration
    if getattr(params, "video_type", 'non_rotating') == 'non_rotating':
        click.echo("Non-rotating mode requires selecting a search region.")
        roi_config = interactive_roi_configuration(video_path)
        apply_roi_config(params, roi_config)
        if params.roi_mode == 'auto':
            raise click.ClickException("Please draw a circle or polygon ROI for non-rotating videos.")
    else:
        params.roi_mode = 'auto'
    
    return params


@click.command()
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('-o', '--output', type=click.Path(), default='./tracking_results',
              help='Directory or .zarr file for outputs')
@click.option('--max-frames', type=int, default=None,
              help='Maximum number of frames to process (default: all)')
@click.option('--adaptive-background', is_flag=True, default=False,
              help='Enable rotation-compensated background subtraction')
@click.option('--background-buffer-size', type=int, default=10,
              help='Number of frames for rolling background (default: 10)')
@click.option('--threshold', type=int, default=10,
              help='Background subtraction threshold (default: 10)')
@click.option('--video-type', type=click.Choice(['non_rotating', 'rotating']), default=None,
              help='Video type to process (default: non_rotating)')
@click.option('--mouths', type=int, default=None,
              help='Expected number of mouths (default: 1)')
@click.option('--gonads', type=click.IntRange(0, 4), default=None,
              help='Number of gonads to detect (0-4, prompts if not specified)')
@click.option('--bulbs', type=int, default=None,
              help='Number of tentacle bulbs to detect (prompts if not specified)')
@click.option('--roi-type', type=click.Choice(['auto', 'circle', 'polygon']), default=None,
              help='ROI shape type (prompts if not specified)')
@click.option('--interactive-roi', is_flag=True, default=False,
              help='Force interactive ROI selection')
@click.option('--interactive-sampling', is_flag=True, default=False,
              help='Force interactive feature sampling')
@click.option('--interactive', is_flag=True, default=False,
              help='Run full interactive configuration workflow')
@click.option('--mouth-min-area', type=int, default=35, help='Mouth min area (pixels)')
@click.option('--mouth-max-area', type=int, default=160, help='Mouth max area (pixels)')
@click.option('--mouth-max-disappeared', type=int, default=15, help='Mouth max disappeared frames')
@click.option('--mouth-max-distance', type=int, default=50, help='Mouth max tracking distance')
@click.option('--mouth-search-radius', type=int, default=None, help='Mouth search radius')
@click.option('--bulb-min-area', type=int, default=5, help='Bulb min area (pixels)')
@click.option('--bulb-max-area', type=int, default=35, help='Bulb max area (pixels)')
@click.option('--bulb-max-disappeared', type=int, default=10, help='Bulb max disappeared frames')
@click.option('--bulb-max-distance', type=int, default=30, help='Bulb max tracking distance')
@click.option('--bulb-search-radius', type=int, default=None, help='Bulb search radius')
@click.pass_context
def track_jellyfish(ctx, video_path, output, max_frames, adaptive_background, 
                    background_buffer_size, threshold, video_type, mouths, gonads, bulbs, roi_type, 
                    interactive_roi, interactive_sampling, interactive,
                    mouth_min_area, mouth_max_area, mouth_max_disappeared, mouth_max_distance, mouth_search_radius,
                    bulb_min_area, bulb_max_area, bulb_max_disappeared, bulb_max_distance, bulb_search_radius):
    """Entry point for multi-object jellyfish tracking."""

    params = TrackingParameters()
    params.background_buffer_size = background_buffer_size
    params.threshold = threshold
    params.mouth_min_area = mouth_min_area
    params.mouth_max_area = mouth_max_area
    params.mouth_max_disappeared = mouth_max_disappeared
    params.mouth_max_distance = mouth_max_distance
    params.mouth_search_radius = mouth_search_radius
    params.bulb_min_area = bulb_min_area
    params.bulb_max_area = bulb_max_area
    params.bulb_max_disappeared = bulb_max_disappeared
    params.bulb_max_distance = bulb_max_distance
    params.bulb_search_radius = bulb_search_radius

    selected_video_type = (video_type or 'non_rotating').lower()
    setattr(params, "video_type", selected_video_type)

    if mouths is not None:
        params.num_mouths = max(0, mouths)
    if gonads is not None:
        params.num_gonads = max(0, gonads)
    if bulbs is not None:
        params.num_tentacle_bulbs = max(0, bulbs) if bulbs >= 0 else None
    else:
        params.num_tentacle_bulbs = None

    needs_interactive = interactive

    if needs_interactive:
        click.echo("Running full interactive configuration...")
        params = run_interactive_configuration(video_path)
        params.background_buffer_size = background_buffer_size
        params.threshold = threshold
    else:
        if interactive_sampling:
            params = interactive_sampling_configuration(video_path, params)

        video_mode = getattr(params, "video_type", selected_video_type)
        if video_mode == 'non_rotating':
            click.echo("Non-rotating mode requires selecting a search region.")
            roi_config = interactive_roi_configuration(
                video_path,
                preferred_mode=roi_type if roi_type and roi_type != 'auto' else None,
            )
            apply_roi_config(params, roi_config)
            if params.roi_mode == 'auto':
                raise click.ClickException("Please draw a circle or polygon ROI for non-rotating videos.")
        else:
            if interactive_roi or (roi_type and roi_type != 'auto'):
                roi_config = interactive_roi_configuration(video_path, preferred_mode=roi_type)
                apply_roi_config(params, roi_config)
            else:
                params.roi_mode = 'auto'

    params.update_object_counts()
    params.update_from_legacy_params()

    # Determine background mode automatically based on video type
    if getattr(params, "video_type", 'non_rotating') == 'rotating':
        params.adaptive_background = True
    else:
        params.adaptive_background = False

    enabled_types = params.get_enabled_object_types()
    if not enabled_types:
        raise click.ClickException("No object types enabled. Please configure at least one object type.")

    click.echo("\n=== Configuration Summary ===")
    click.echo(f"Video: {video_path}")
    click.echo(f"Output: {output}")
    click.echo(f"Max frames: {max_frames if max_frames else 'all'}")
    click.echo(f"Enabled objects: {', '.join(enabled_types)}")

    tracking_results, fps = run_multi_object_tracking(video_path, params, max_frames)

    output_path = Path(output)
    if output_path.suffix == '.zarr':
        output_path.parent.mkdir(parents=True, exist_ok=True)
        zarr_path = output_path
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        zarr_path = output_path / "multi_object_tracking.zarr"

    save_multi_object_tracking_to_zarr(tracking_results, str(zarr_path), fps, params)

    click.echo("Tracking completed")
    click.echo(f"Results saved to {zarr_path}")


if __name__ == '__main__':
    track_jellyfish()
