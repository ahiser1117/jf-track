import click
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tracker import TrackingParameters
from src.interactive_sampling import run_interactive_feature_sampling


def interactive_object_configuration() -> TrackingParameters:
    """Interactive prompts for object type configuration."""
    click.echo("=== Jellyfish Object Configuration ===")
    
    params = TrackingParameters()
    
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
        params.num_tentacle_bulbs = click.prompt(
            'How many tentacle bulbs', 
            type=int, 
            default=8
        )
        
        if params.num_tentacle_bulbs > 0:
            click.echo(f"Will track up to {params.num_tentacle_bulbs} tentacle bulbs.")
            
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
    
    return params


def interactive_roi_configuration(video_path: str) -> dict:
    """Interactive ROI selection configuration."""
    click.echo("\n=== ROI Configuration ===")
    click.echo("Choose how to define the tracking region:")
    
    roi_choice = click.prompt(
        'ROI mode',
        type=click.Choice(['auto', 'circle', 'polygon']),
        default='auto'
    )
    
    roi_config = {"mode": roi_choice}
    
    if roi_choice == 'circle':
        click.echo("\nCircle ROI: Click center and drag for radius")
        roi_config["method"] = "interactive"
    elif roi_choice == 'polygon':
        click.echo("\nPolygon ROI: Click vertices to define boundary")
        roi_config["method"] = "interactive"
    else:
        click.echo("Using automatic circular ROI (center of frame)")
        roi_config["method"] = "auto"
    
    return roi_config


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
    
    # Object configuration
    params = interactive_object_configuration()
    
    # Feature sampling
    params = interactive_sampling_configuration(video_path, params)
    
    # ROI configuration
    roi_config = interactive_roi_configuration(video_path)
    params.roi_mode = roi_config["mode"]
    
    return params


@click.command()
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('-o', '--output', type=click.Path(), default='./tracking_results',
              help='Output directory for results (default: ./tracking_results)')
@click.option('--max-frames', type=int, default=None,
              help='Maximum number of frames to process (default: all)')
@click.option('--adaptive-background', is_flag=True, default=False,
              help='Enable rotation-compensated background subtraction')
@click.option('--background-buffer-size', type=int, default=10,
              help='Number of frames for rolling background (default: 10)')
@click.option('--threshold', type=int, default=10,
              help='Background subtraction threshold (default: 10)')

# Object configuration options
@click.option('--gonads', type=click.IntRange(0, 4), default=None,
              help='Number of gonads to detect (0-4, prompts if not specified)')
@click.option('--bulbs', type=int, default=None,
              help='Number of tentacle bulbs to detect (prompts if not specified)')

# ROI configuration options
@click.option('--roi-type', type=click.Choice(['auto', 'circle', 'polygon']), default=None,
              help='ROI shape type (prompts if not specified)')
@click.option('--interactive-roi', is_flag=True, default=False,
              help='Force interactive ROI selection')

# Feature sampling
@click.option('--interactive-sampling', is_flag=True, default=False,
              help='Force interactive feature sampling')

# Interactive mode
@click.option('--interactive', is_flag=True, default=False,
              help='Run in interactive mode (prompts for all configurations)')

# Legacy compatibility options
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
                   background_buffer_size, threshold, gonads, bulbs, roi_type, 
                   interactive_roi, interactive_sampling, interactive,
                   mouth_min_area, mouth_max_area, mouth_max_disappeared, mouth_max_distance, mouth_search_radius,
                   bulb_min_area, bulb_max_area, bulb_max_disappeared, bulb_max_distance, bulb_search_radius):
    """
    jf-track: Jellyfish tracking with configurable object detection.
    
    Supports mouth, gonads (0-4, oblong), and tentacle bulbs.
    Interactive configuration available for object counts, ROI selection, and feature sampling.
    
    Examples:
    
        # Interactive mode with prompts
        python -m src.cli video.mp4 --interactive
        
        # Batch mode with specific parameters
        python -m src.cli video.mp4 --gonads 2 --bulbs 8 --roi-type circle
        
        # Feature sampling for automatic parameter tuning
        python -m src.cli video.mp4 --interactive-sampling
        
        # Adaptive background for rotating videos
        python -m src.cli video.mp4 --adaptive-background --interactive-roi
    """
    
    # Initialize parameters
    params = TrackingParameters()
    
    # Set basic parameters
    params.background_buffer_size = background_buffer_size
    params.threshold = threshold
    params.adaptive_background = adaptive_background
    
    # Set legacy parameters for backward compatibility
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
    
    # Determine if interactive configuration is needed
    need_interactive = (
        interactive or  # User explicitly requested interactive mode
        gonads is None or bulbs is None or roi_type is None or  # Missing required params
        interactive_roi or interactive_sampling or  # Specific interactive features requested
        check_missing_required_params(params)
    )
    
    if need_interactive:
        click.echo("Running in interactive configuration mode...")
        params = run_interactive_configuration(video_path)
    else:
        # Use command-line parameters
        params.num_gonads = gonads if gonads is not None else 0
        params.num_tentacle_bulbs = bulbs if bulbs is not None else 0
        params.roi_mode = roi_type if roi_type is not None else "auto"
        
        if interactive_sampling:
            params = interactive_sampling_configuration(video_path, params)
        
        if interactive_roi:
            roi_config = interactive_roi_configuration(video_path)
            params.roi_mode = roi_config["mode"]
    
    # Update object configurations based on counts
    params.update_object_counts()
    params.update_from_legacy_params()
    
    # Validate configuration
    enabled_types = params.get_enabled_object_types()
    if not enabled_types:
        click.echo("Error: No object types enabled. Please configure at least one object type.")
        ctx.exit(1)
    
    # Print configuration summary
    click.echo("\n=== Configuration Summary ===")
    click.echo(f"Video: {video_path}")
    click.echo(f"Output: {output}")
    click.echo(f"Max frames: {max_frames if max_frames else 'all'}")
    click.echo(f"ROI mode: {params.roi_mode}")
    click.echo(f"Enabled objects: {', '.join(enabled_types)}")
    
    if params.num_gonads > 0:
        click.echo(f"Gonads: {params.num_gonads}")
    if params.num_tentacle_bulbs > 0:
        click.echo(f"Tentacle bulbs: {params.num_tentacle_bulbs}")
    
    click.echo("\nStarting tracking...")
    
    # TODO: This would call the actual tracking function
    # results = run_multi_object_tracking(video_path, params, max_frames)
    # save_tracking_results(results, output, params)
    
    click.echo("Tracking completed!")
    click.echo(f"Results saved to: {output}")


# Backward compatibility wrapper functions
def backward_compatibility_wrapper(
    video_path: str,
    max_frames: int | None = None,
    background_buffer_size: int = 10,
    threshold: int = 10,
    mouth_min_area: int = 35,
    mouth_max_area: int = 160,
    mouth_max_disappeared: int = 15,
    mouth_max_distance: int = 50,
    mouth_search_radius: int | None = None,
    bulb_min_area: int = 5,
    bulb_max_area: int = 35,
    bulb_max_disappeared: int = 10,
    bulb_max_distance: int = 30,
    bulb_search_radius: int | None = None,
    adaptive_background: bool = False,
    rotation_start_threshold_deg: float = 0.01,
    rotation_stop_threshold_deg: float = 0.005,
    rotation_center: tuple[float, float] | None = None,
):
    """
    Backward compatibility wrapper that accepts legacy two-pass parameters
    and runs the appropriate tracking system.
    
    This function maintains the existing API while supporting the new
    multi-object system when additional parameters are configured.
    """
    from .tracking import run_two_pass_tracking, run_multi_object_tracking
    from .tracker import TrackingParameters
    
    # Create parameters from legacy arguments
    params = TrackingParameters(
        background_buffer_size=background_buffer_size,
        threshold=threshold,
        mouth_min_area=mouth_min_area,
        mouth_max_area=mouth_max_area,
        mouth_max_disappeared=mouth_max_disappeared,
        mouth_max_distance=mouth_max_distance,
        mouth_search_radius=mouth_search_radius,
        bulb_min_area=bulb_min_area,
        bulb_max_area=bulb_max_area,
        bulb_max_disappeared=bulb_max_disappeared,
        bulb_max_distance=bulb_max_distance,
        bulb_search_radius=bulb_search_radius,
        adaptive_background=adaptive_background,
        rotation_start_threshold_deg=rotation_start_threshold_deg,
        rotation_stop_threshold_deg=rotation_stop_threshold_deg,
        rotation_center=rotation_center,
    )
    
    # Update object configurations from legacy parameters
    params.update_from_legacy_params()
    
    # Check if additional objects are configured beyond mouth+basic bulbs
    enabled_types = params.get_enabled_object_types()
    has_only_legacy_objects = (
        "mouth" in enabled_types and 
        "tentacle_bulb" in enabled_types and 
        params.num_gonads == 0 and 
        "gonad" not in enabled_types
    )
    
    if has_only_legacy_objects:
        # Use legacy two-pass tracking for exact compatibility
        click.echo("Using legacy two-pass tracking for backward compatibility")
        return run_two_pass_tracking(
            video_path=video_path,
            max_frames=max_frames,
            background_buffer_size=background_buffer_size,
            threshold=threshold,
            mouth_min_area=mouth_min_area,
            mouth_max_area=mouth_max_area,
            mouth_max_disappeared=mouth_max_disappeared,
            mouth_max_distance=mouth_max_distance,
            mouth_search_radius=mouth_search_radius,
            bulb_min_area=bulb_min_area,
            bulb_max_area=bulb_max_area,
            bulb_max_disappeared=bulb_max_disappeared,
            bulb_max_distance=bulb_max_distance,
            bulb_search_radius=bulb_search_radius,
            adaptive_background=adaptive_background,
            rotation_start_threshold_deg=rotation_start_threshold_deg,
            rotation_stop_threshold_deg=rotation_stop_threshold_deg,
            rotation_center=rotation_center,
        )
    else:
        # Use new multi-object tracking system
        click.echo("Using multi-object tracking system")
        return run_multi_object_tracking(video_path, params, max_frames)


# Update CLI to call actual tracking
@click.pass_context
def execute_tracking(ctx, video_path, output, max_frames, adaptive_background, 
                   background_buffer_size, threshold, gonads, bulbs, roi_type, 
                   interactive_roi, interactive_sampling, interactive,
                   mouth_min_area, mouth_max_area, mouth_max_disappeared, mouth_max_distance, mouth_search_radius,
                   bulb_min_area, bulb_max_area, bulb_max_disappeared, bulb_max_distance, bulb_search_radius):
    """
    Execute tracking with the appropriate system based on configuration.
    """
    from .tracking import run_multi_object_tracking
    from .save_results import save_multi_object_tracking_to_zarr
    
    # Initialize parameters
    params = TrackingParameters()
    params.background_buffer_size = background_buffer_size
    params.threshold = threshold
    params.adaptive_background = adaptive_background
    
    # Set legacy parameters
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
    
    # Handle interactive configuration
    if interactive or gonads is None or bulbs is None or roi_type is None:
        params = run_interactive_configuration(video_path)
        params.background_buffer_size = background_buffer_size
        params.threshold = threshold
        params.adaptive_background = adaptive_background
    else:
        # Use command-line parameters
        params.num_gonads = gonads if gonads is not None else 0
        params.num_tentacle_bulbs = bulbs if bulbs is not None else 0
        params.roi_mode = roi_type if roi_type is not None else "auto"
        
        if interactive_sampling:
            from .interactive_sampling import run_interactive_feature_sampling
            params = run_interactive_feature_sampling(video_path)
        
        if interactive_roi:
            from .roi_selector import run_interactive_roi_selection
            roi_config = run_interactive_roi_selection(video_path)
            params.roi_mode = roi_config["mode"]
            if roi_config["mode"] == "circle":
                params.roi_center = roi_config["center"]
                params.roi_radius = roi_config["radius"]
            elif roi_config["mode"] == "polygon":
                params.roi_points = roi_config["points"]
    
    # Update object configurations
    params.update_object_counts()
    params.update_from_legacy_params()
    
    # Determine tracking system to use
    enabled_types = params.get_enabled_object_types()
    if "gonad" in enabled_types or params.num_gonads > 0:
        # Use multi-object tracking for gonads
        click.echo("Using multi-object tracking (gonads detected)")
        results = run_multi_object_tracking(video_path, params, max_frames)
        
        # Save multi-object results
        save_multi_object_tracking_to_zarr(results, output, params.background_buffer_size, params)
        
    elif len(enabled_types) > 2:  # More than just mouth + basic bulbs
        click.echo("Using multi-object tracking (extended configuration)")
        results = run_multi_object_tracking(video_path, params, max_frames)
        
        # Save multi-object results  
        save_multi_object_tracking_to_zarr(results, output, params.background_buffer_size, params)
        
    else:
        # Use legacy two-pass for exact compatibility
        click.echo("Using legacy two-pass tracking for backward compatibility")
        from .tracking import run_two_pass_tracking
        
        mouth_tracking, bulb_tracking, fps, tracking_params = run_two_pass_tracking(
            video_path=video_path,
            max_frames=max_frames,
            background_buffer_size=background_buffer_size,
            threshold=threshold,
            mouth_min_area=mouth_min_area,
            mouth_max_area=mouth_max_area,
            mouth_max_disappeared=mouth_max_disappeared,
            mouth_max_distance=mouth_max_distance,
            mouth_search_radius=mouth_search_radius,
            bulb_min_area=bulb_min_area,
            bulb_max_area=bulb_max_area,
            bulb_max_disappeared=bulb_max_disappeared,
            bulb_max_distance=bulb_max_distance,
            bulb_search_radius=bulb_search_radius,
            adaptive_background=adaptive_background,
        )
        
        # Save two-pass results
        from .save_results import save_two_pass_tracking_to_zarr
        from .direction_analysis import compute_direction_analysis
        
        # Compute direction analysis
        direction_analysis = compute_direction_analysis(mouth_tracking, bulb_tracking)
        
        # Save results
        save_two_pass_tracking_to_zarr(
            mouth_tracking, bulb_tracking, direction_analysis,
            output, tracking_params.background_buffer_size, tracking_params
        )
    
    click.echo("Tracking completed!")
    click.echo(f"Results saved to: {output}")


# Replace the main function with the execute_tracking function
track_jellyfish = execute_tracking


if __name__ == '__main__':
    track_jellyfish()