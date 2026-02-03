import math
import numpy as np

from src.tracker import TrackingParameters
from src.tracking import get_roi_mask_for_video
from src.interactive_sampling import (
    FeatureSample,
    _apply_samples_to_parameters,
    SEARCH_RADIUS_MARGIN,
)


def test_non_rotating_defaults_to_full_frame_mask():
    params = TrackingParameters()
    mask = get_roi_mask_for_video(params, height=8, width=12, video_type="non_rotating")

    assert mask.shape == (8, 12)
    assert np.all(mask == 255)


def test_rotating_defaults_to_circular_mask():
    params = TrackingParameters()
    mask = get_roi_mask_for_video(params, height=10, width=10, video_type="rotating")

    assert mask[5, 5] == 255  # center is inside circle
    assert mask[0, 0] == 0  # corner should be outside


def test_custom_bounding_box_roi_is_respected():
    params = TrackingParameters()
    params.apply_roi_config(
        {
            "mode": "bounding_box",
            "bbox": (2, 1, 3, 4),
        }
    )

    mask = get_roi_mask_for_video(params, height=8, width=12, video_type="non_rotating")

    # Inside bbox should be white, outside black
    assert mask[2, 3] == 255
    assert mask[0, 0] == 0


def test_pinned_mouth_applies_search_radius_without_mouth_samples():
    params = TrackingParameters()
    params.mouth_pinned = True
    params.pinned_mouth_point = (100.0, 100.0)

    gonad_sample = FeatureSample(
        feature_type="gonad",
        frame_idx=0,
        centroid=(130.0, 100.0),
        area=50.0,
        bbox=(0, 0, 10, 10),
        aspect_ratio=2.0,
        eccentricity=0.8,
        solidity=0.9,
    )

    samples = {
        "mouth": [],
        "gonad": [gonad_sample],
        "tentacle_bulb": [],
    }

    _apply_samples_to_parameters(params, samples)

    expected_radius = int(math.ceil(30.0 * SEARCH_RADIUS_MARGIN))
    assert params.object_types["gonad"]["search_radius"] == expected_radius
