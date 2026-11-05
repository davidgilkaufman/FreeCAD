"""Reader for the adaptive cut load dump (CutLoadDumper in libarea/Adaptive.cpp).

load(path) -> Dump with .header (dict) and .passes (list of Pass). Each Pass has .region, .index,
and .variants: {kind name: numpy array of shape (n, 4) with columns x, y, area, aux}.
"""

import struct
from dataclasses import dataclass, field

import numpy as np

KINDS = {
    0: "computed",
    1: "computed_replay",
    2: "output",
    5: "cleared_start",  # polygon vertices; aux = polygon index
    6: "cleared_end",
    7: "output_cleared_start",  # the same in the output world (passes swept along gcode points)
    8: "output_cleared_end",
    # per point: x = cut region perimeter on the previous tool circle, y = on the current circle,
    # area = the rest (mm); aux = cut region polygon area (mm^2)
    9: "computed_perimeter",
    10: "output_perimeter",
    # per computed point: x = search exit reason (1 tolerance, 2 adjacent integers, 3 iteration
    # limit, 4 no area), y = rotated back inside the boundary (0/1), area = iterations
    11: "computed_search",
    # tries of the angle search for steps that ended by exit 2 (adjacent integers), one row per try
    12: "search_tries_pos",  # x, y = try end point, area = its cut area, aux = point index
    13: "search_tries_info",  # x = angle (rad, relative to straight), y = conventional, area = iteration
    14: "search_tries_bounds",  # x, y = interp best / bound angles (rad) after the try, NaN if unset
    # (12-14 cover every accepted step, plus the failed search ending the pass, keyed by point n)
    15: "fail",  # one row: x = exit reason, y = rotated, area = fail bits (1 area<1, 2 conventional,
    #              4 over cut, 8 rotate failed), aux = iterations; of the search that ended the pass
    16: "lead_out",  # lead out path after the pass (x, y mm)
    17: "initial_dir",  # one row: x, y = unit tool direction before the pass's first step
}

POLYGON_KINDS = ("cleared_start", "cleared_end", "output_cleared_start", "output_cleared_end")


def polygons(arr):
    """Split a cleared_* array into a list of (n, 2) polygon vertex arrays."""
    if len(arr) == 0:
        return []
    idx = arr[:, 3].astype(int)
    cuts = np.where(np.diff(idx) != 0)[0] + 1
    return [a[:, :2] for a in np.split(arr, cuts)]


@dataclass
class Pass:
    region: int
    index: int
    variants: dict = field(default_factory=dict)


@dataclass
class Dump:
    header: dict
    passes: list


def load(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"ADPDUMP1":
        raise ValueError(f"{path}: not an adaptive dump")
    version, _reserved, scale, toolDiameter, stepOverFactor, optimalPD = struct.unpack_from("<iidddd", data, 8)
    pos = 8 + struct.calcsize("<iidddd")
    header = {
        "version": version,
        "scaleFactor": scale,
        "toolDiameter": toolDiameter,
        "stepOverFactor": stepOverFactor,
        "optimalCutAreaPD": optimalPD,
    }
    passes = []
    while pos < len(data):
        if data[pos : pos + 4] != b"PASS":
            # a truncated record at the end (the writer was killed mid-write): stop
            break
        try:
            region, index, nVariants = struct.unpack_from("<iii", data, pos + 4)
            p = pos + 16
            variants = {}
            for _ in range(nVariants):
                kind, n = struct.unpack_from("<ii", data, p)
                p += 8
                arr = np.frombuffer(data, dtype="<f8", count=n * 4, offset=p).reshape(n, 4)
                p += n * 32
                variants[KINDS.get(kind, f"kind{kind}")] = arr
        except (struct.error, ValueError):
            break
        passes.append(Pass(region, index, variants))
        pos = p
    return Dump(header, passes)


def loadPerDistance(arr):
    """Area per distance of travel for each step (len n-1), and the step lengths."""
    d = np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        pd = np.where(d > 0, arr[1:, 2] / d, 0.0)
    return pd, d
