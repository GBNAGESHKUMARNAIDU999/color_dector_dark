"""
OpenCV CSRT-based multi-object tracker.
Supports manual ROI selection and re-initialization for precision tracking.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from utils.logger import logger


@dataclass
class CSRTTrack:
    """State of a single CSRT tracked object."""
    track_id: int
    label: str
    tracker: cv2.Tracker
    bounding_box: Tuple[int, int, int, int]
    trail: List[Tuple[int, int]] = field(default_factory=list)
    active: bool = True
    frames_tracked: int = 0


class MultiObjectTracker:
    """
    CSRT (Channel and Spatial Reliability Tracker) based multi-object tracker.

    CSRT is the most accurate single-object tracker in OpenCV,
    suitable for tracking one or more user-selected regions of interest.

    Usage:
        tracker = MultiObjectTracker()
        tracker.add_object(frame, bbox=(x, y, w, h), label="ball")
        for frame in video:
            results = tracker.update(frame)
    """

    def __init__(self, trail_length: int = 40):
        self.trail_length = trail_length
        self._tracks: Dict[int, CSRTTrack] = {}
        self._next_id = 0

    def add_object(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str = "object",
    ) -> int:
        """
        Initialize a new CSRT tracker for a region of interest.

        Args:
            frame: BGR frame where the object is currently visible.
            bbox: Bounding box (x, y, w, h).
            label: Human-readable object label.

        Returns:
            Assigned track ID.
        """
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)
        cx = bbox[0] + bbox[2] // 2
        cy = bbox[1] + bbox[3] // 2

        track = CSRTTrack(
            track_id=self._next_id,
            label=label,
            tracker=tracker,
            bounding_box=bbox,
            trail=[(cx, cy)],
        )
        self._tracks[self._next_id] = track
        logger.info(f"CSRT tracker initialized: ID={self._next_id} label='{label}' bbox={bbox}")
        self._next_id += 1
        return track.track_id

    def update(self, frame: np.ndarray) -> List[CSRTTrack]:
        """
        Update all active trackers with a new frame.

        Args:
            frame: Latest BGR video frame.

        Returns:
            List of all CSRTTrack objects (active and lost).
        """
        for track in self._tracks.values():
            if not track.active:
                continue

            success, bbox = track.tracker.update(frame)

            if success:
                x, y, w, h = (int(v) for v in bbox)
                track.bounding_box = (x, y, w, h)
                cx, cy = x + w // 2, y + h // 2
                track.trail.append((cx, cy))
                if len(track.trail) > self.trail_length:
                    track.trail.pop(0)
                track.frames_tracked += 1
            else:
                logger.warning(f"CSRT tracker lost: ID={track.track_id}")
                track.active = False

        return list(self._tracks.values())

    def remove_object(self, track_id: int) -> None:
        """Remove a tracked object by ID."""
        if track_id in self._tracks:
            del self._tracks[track_id]
            logger.info(f"Removed tracker ID={track_id}")

    def reset(self) -> None:
        """Remove all tracked objects."""
        self._tracks.clear()
        self._next_id = 0

    @property
    def active_count(self) -> int:
        return sum(1 for t in self._tracks.values() if t.active)

    def draw_tracks(
        self,
        frame: np.ndarray,
        show_trail: bool = True,
    ) -> np.ndarray:
        """
        Draw all tracked objects on a frame.

        Args:
            frame: BGR frame to annotate.
            show_trail: Whether to draw the movement trail.

        Returns:
            Annotated BGR frame.
        """
        annotated = frame.copy()
        active_color = (0, 255, 0)
        lost_color = (0, 0, 200)

        for track in self._tracks.values():
            color = active_color if track.active else lost_color
            x, y, w, h = track.bounding_box

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            label = f"ID:{track.track_id} {track.label}"
            if not track.active:
                label += " [LOST]"
            cv2.putText(
                annotated, label, (x, max(y - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )

            if show_trail and len(track.trail) > 1:
                for i in range(1, len(track.trail)):
                    alpha = i / len(track.trail)
                    faded = tuple(int(c * alpha) for c in color)
                    cv2.line(annotated, track.trail[i - 1], track.trail[i], faded, 2)

        return annotated
