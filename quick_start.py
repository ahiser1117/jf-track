from src.tracking import run_multi_object_tracking
from src.tracker import TrackingParameters
params = TrackingParameters()
params.update_object_counts()  # or load params from the GUI run
results, fps = run_multi_object_tracking(
    "path/to/video.avi",
    params,
    max_frames=1000,
)