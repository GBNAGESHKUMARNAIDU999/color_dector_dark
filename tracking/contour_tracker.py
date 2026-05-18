"""
Real-time contour-based color object tracker.
Tracks colored blobs across video frames using centroid tracking and
bounding box smoothing.  Capped at max_objects to prevent visual clutter.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
from dataclasses import dataclass, field

from cv_pipeline.color_detector import detect_colors, DetectionResult
from utils.logger import logger


@dataclass
class TrackedObject:
    """State of a single tracked color blob."""
    object_id: int
    color_name: str
    center: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]
    trail: List[Tuple[int, int]] = field(default_factory=list)
    frames_alive: int = 0
    frames_missing: int = 0
    confidence: float = 1.0


class ContourTracker:
    """
    Multi-color, multi-object contour tracker.

    Tracks blobs by centroid proximity across frames.
    Supports:
    - Multiple simultaneous color targets
    - Centroid trail visualization
    - Smooth bounding box display
    - Object persistence (re-association after brief occlusion)
    - Hard cap on total tracked objects
    """

    def __init__(
        self,
        max_disappeared: int = 8,
        max_distance: int = 100,
        trail_length: int = 10,
        max_objects: int = 10,
    ):
        """
        Args:
            max_disappeared: Frames to keep a lost object before dropping it.
            max_distance: Max centroid pixel distance to consider the same object.
            trail_length: Number of centroid positions to retain in the trail.
            max_objects: Hard cap on total simultaneously tracked objects.
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.trail_length = trail_length
        self.max_objects = max_objects
        self._next_id = 0
        self._objects: Dict[int, TrackedObject] = OrderedDict()

    @property
    def active_objects(self) -> List[TrackedObject]:
        """Return only currently active (not missing) tracked objects."""
        return [o for o in self._objects.values() if o.frames_missing == 0]

    def update(
        self,
        frame: np.ndarray,
        target_colors: Optional[List[str]] = None,
    ) -> List[TrackedObject]:
        """
        Process one frame: detect colors → associate with existing tracks.

        Args:
            frame: BGR video frame.
            target_colors: Colors to detect (None = detect all).

        Returns:
            List of currently active TrackedObject instances.
        """
        # Detect color regions in this frame
        detections_map = detect_colors(frame, target_colors=target_colors)

        # Flatten all detections into (color_name, center, bbox, confidence) tuples
        new_detections: List[Tuple[str, Tuple[int, int], Tuple, float]] = []
        for color_name, detections in detections_map.items():
            for det in detections:
                new_detections.append(
                    (color_name, det.center, det.bounding_box, det.confidence)
                )

        if not new_detections:
            # Mark all existing objects as missing
            for obj in self._objects.values():
                obj.frames_missing += 1
            # Remove objects missing too long
            self._objects = OrderedDict(
                (oid, obj)
                for oid, obj in self._objects.items()
                if obj.frames_missing <= self.max_disappeared
            )
            return self.active_objects

        # If no existing tracks, register all as new (up to cap)
        if not self._objects:
            for color_name, center, bbox, conf in new_detections[:self.max_objects]:
                self._register(color_name, center, bbox, conf)
            return self.active_objects

        # Match new detections to existing tracks by minimum centroid distance
        existing_ids = list(self._objects.keys())
        existing_centers = [self._objects[oid].center for oid in existing_ids]

        used_detection_indices = set()
        used_object_ids = set()

        for oid, ex_center in zip(existing_ids, existing_centers):
            best_idx = None
            best_dist = float("inf")
            for i, (_, det_center, _, _) in enumerate(new_detections):
                if i in used_detection_indices:
                    continue
                dist = np.linalg.norm(
                    np.array(ex_center) - np.array(det_center)
                )
                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_idx = i

            if best_idx is not None:
                color_name, center, bbox, conf = new_detections[best_idx]
                obj = self._objects[oid]
                obj.center = center
                obj.bounding_box = bbox
                obj.confidence = conf
                obj.frames_alive += 1
                obj.frames_missing = 0
                obj.trail.append(center)
                if len(obj.trail) > self.trail_length:
                    obj.trail.pop(0)
                used_detection_indices.add(best_idx)
                used_object_ids.add(oid)
            else:
                self._objects[oid].frames_missing += 1

        # Register unmatched detections as new objects (respect cap)
        for i, (color_name, center, bbox, conf) in enumerate(new_detections):
            if i not in used_detection_indices:
                if len(self._objects) < self.max_objects:
                    self._register(color_name, center, bbox, conf)

        # Remove lost objects
        self._objects = OrderedDict(
            (oid, obj)
            for oid, obj in self._objects.items()
            if obj.frames_missing <= self.max_disappeared
        )

        return self.active_objects

    def _register(
        self,
        color_name: str,
        center: Tuple[int, int],
        bbox: Tuple[int, int, int, int],
        confidence: float,
    ) -> None:
        obj = TrackedObject(
            object_id=self._next_id,
            color_name=color_name,
            center=center,
            bounding_box=bbox,
            trail=[center],
            frames_alive=0,
            frames_missing=0,
            confidence=confidence,
        )
        self._objects[self._next_id] = obj
        self._next_id += 1

    def reset(self) -> None:
        """Clear all tracked objects."""
        self._objects.clear()
        self._next_id = 0

    def draw_tracks(
        self,
        frame: np.ndarray,
        show_trail: bool = True,
        show_id: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes, trails, and IDs on a frame.

        Args:
            frame: BGR frame to annotate.
            show_trail: Draw centroid movement trail.
            show_id: Display object ID.

        Returns:
            Annotated BGR frame.
        """
        COLORS = {
            "red":        (0, 0, 220),
            "orange":     (0, 128, 255),
            "yellow":     (0, 220, 220),
            "lime":       (0, 255, 128),
            "green":      (0, 200, 0),
            "teal":       (128, 128, 0),
            "cyan":       (220, 220, 0),
            "sky blue":   (235, 206, 135),
            "blue":       (220, 50, 0),
            "navy":       (128, 0, 0),
            "purple":     (180, 0, 180),
            "magenta":    (255, 0, 255),
            "pink":       (140, 80, 220),
            "brown":      (42, 42, 165),
            "maroon":     (0, 0, 128),
            "olive":      (0, 128, 128),
            "gold":       (0, 215, 255),
            "coral":      (80, 127, 255),
            "beige":      (220, 245, 245),
            "lavender":   (250, 230, 230),
            "turquoise":  (208, 224, 64),
            "gray":       (128, 128, 128),
            "white":      (220, 220, 220),
            "black":      (80, 80, 80),
        }

        annotated = frame.copy()

        for obj in self.active_objects:
            color = COLORS.get(obj.color_name, (0, 255, 0))
            x, y, w, h = obj.bounding_box

            # Bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Trail — short, subtle line
            if show_trail and len(obj.trail) > 1:
                for i in range(1, len(obj.trail)):
                    alpha = i / len(obj.trail)
                    thickness = max(1, int(alpha * 2))
                    cv2.line(annotated, obj.trail[i - 1], obj.trail[i], color, thickness)

            # Label
            if show_id:
                label = f"{obj.color_name} {obj.confidence:.0%}"
                cv2.putText(
                    annotated, label, (x, max(y - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )

            # Center dot
            cv2.circle(annotated, obj.center, 4, color, -1)

        return annotated
