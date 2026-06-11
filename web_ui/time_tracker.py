"""Task time tracker - per-task-type time estimation."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from .schemas import ExperimentTask

# Keep only the most recent N execution times per task type
MAX_HISTORY_PER_TASK = 10


def _task_key(task: "ExperimentTask") -> str:
    """Generate task type key for grouping.

    Key is sim_type: throughput_optimizer, text_generate, or video_generate
    """
    return task.sim_type


class TaskTimeTracker:
    """Track execution time per task type."""

    def __init__(self):
        # key -> list of recent durations (seconds)
        self._history: Dict[str, List[float]] = defaultdict(list)
        # key -> last completion time
        self._last_seen: Dict[str, float] = {}

    def record(self, task: "ExperimentTask", duration_s: float):
        """Record execution time for a task."""
        key = _task_key(task)
        self._history[key].append(duration_s)

        # Keep only recent history
        if len(self._history[key]) > MAX_HISTORY_PER_TASK:
            self._history[key].pop(0)

        self._last_seen[key] = time.time()

    def get_estimate(self, task: "ExperimentTask") -> float:
        """
        Get estimated time for a task based on similar tasks.

        Returns:
            Estimated seconds, or 60.0 if no history
        """
        key = _task_key(task)
        history = self._history.get(key)

        if not history:
            # No history for this exact task type
            # Try to find similar tasks (same model_id at least)
            model_id = task.params.get("model_id", "")
            for other_key, other_history in self._history.items():
                if other_key.startswith(model_id):
                    # Similar model found, use its average
                    return sum(other_history) / len(other_history)
            return 60.0  # Default fallback

        # Use average of recent history
        return sum(history) / len(history)

    def get_stats(self) -> Dict[str, Dict]:
        """Get statistics for all tracked task types."""
        stats = {}
        for key, history in self._history.items():
            if history:
                stats[key] = {
                    "count": len(history),
                    "avg": sum(history) / len(history),
                    "min": min(history),
                    "max": max(history),
                    "recent": history[-1],
                }
        return stats


# Global instance
_tracker = TaskTimeTracker()


def get_tracker() -> TaskTimeTracker:
    return _tracker
