"""Validate and synchronize observation references for later dataset export."""
import json
from pathlib import Path
import numpy as np
from .data import nearest_indices


def align_episode(state_rows, cameras, max_offset):
    """Align camera records to state rows. Each row has timestamp; cameras also have path.

    Missing camera matches are represented as null, never silently extrapolated.
    Timestamps must already share a clock and use seconds.
    """
    if not state_rows:
        raise ValueError('episode requires state rows')
    times = [float(row['timestamp']) for row in state_rows]
    aligned = [dict(row, cameras={}) for row in state_rows]
    for name, frames in cameras.items():
        if not name or any(not frame.get('path') for frame in frames):
            raise ValueError('camera names and frame paths are required')
        indices = nearest_indices(times, [float(frame['timestamp']) for frame in frames], max_offset)
        for row, index in zip(aligned, indices):
            row['cameras'][name] = None if index < 0 else dict(frames[int(index)])
    if not cameras:
        nearest_indices(times, [], max_offset)
    return aligned


def write_jsonl(rows, destination):
    """Write JSONL records; reject NaN and Infinity."""
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
