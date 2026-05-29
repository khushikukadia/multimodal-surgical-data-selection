"""Build (or describe) the metadata CSV expected by `scripts/run_experiment.py`.

Three modes:

  1. ``cholectrack20`` (recommended): scan an extracted CholecTrack20 release
     (1 fps frames + per-video JSON). Much smaller than full Cholec80.

  2. ``cholec80`` (legacy): scan Cholec80-style frame folders + TSV phase files.

  3. ``synthetic``: tiny solid-color JPEGs for pipeline smoke tests only.

The output CSV always has columns:

    frame_path,phase

Optional extra columns (CholecTrack20): ``video_id``, ``split``, ``frame_id``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import ensure_dir, log  # noqa: E402


# CholecTrack20 phase IDs (0-indexed) per the dataset paper / JSON annotations.
# Order differs from legacy Cholec80 ID ordering (clipping vs gallbladder dissection).
CHOLECTRACK20_PHASE_ID_TO_NAME: Dict[int, str] = {
    0: "Preparation",
    1: "CalotTriangleDissection",
    2: "GallbladderDissection",
    3: "ClippingCutting",
    4: "GallbladderPackaging",
    5: "CleaningCoagulation",
    6: "GallbladderExtraction",
}

SURGICAL_PHASES = list(CHOLECTRACK20_PHASE_ID_TO_NAME.values())

# Legacy Cholec80 CamelCase names (kept for the cholec80 subcommand).
CHOLEC80_PHASES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]

# Official CholecTrack20 split folders inside the zip.
DEFAULT_SPLITS = ("training", "validation", "testing")
SPLIT_ALIASES = {
    "train": "training",
    "tr": "training",
    "val": "validation",
    "valid": "validation",
    "test": "testing",
}


# -----------------------------------------------------------------------------
# CholecTrack20
# -----------------------------------------------------------------------------


def _normalize_split_name(name: str) -> str:
    key = name.strip().lower()
    return SPLIT_ALIASES.get(key, key)


def _find_split_root(root_dir: str, split: str) -> Optional[str]:
    """Resolve split folder case-insensitively (e.g. ``Training`` vs ``training``)."""
    want = _normalize_split_name(split)
    if not os.path.isdir(root_dir):
        return None
    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path) and name.lower() == want:
            return path
    return None


def _find_video_dirs(split_root: str) -> List[str]:
    """Return absolute paths to per-video folders (e.g. .../training/VID01)."""
    if not os.path.isdir(split_root):
        return []
    out = []
    for entry in sorted(os.listdir(split_root)):
        path = os.path.join(split_root, entry)
        if os.path.isdir(path):
            out.append(path)
    return out


def _find_json_for_video(video_dir: str) -> Optional[str]:
    """Locate the annotation JSON for a video folder."""
    base = os.path.basename(video_dir.rstrip(os.sep))
    candidates = [
        os.path.join(video_dir, f"{base}.json"),
        os.path.join(video_dir, f"{base.lower()}.json"),
    ]
    # VID01 -> also try VID01.json variants
    m = re.match(r"VID(\d+)", base, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        candidates.extend(
            [
                os.path.join(video_dir, f"VID{num:02d}.json"),
                os.path.join(video_dir, f"vid{num:02d}.json"),
            ]
        )
    for path in candidates:
        if os.path.isfile(path):
            return path
    # Fallback: any single JSON in the folder.
    jsons = sorted(glob.glob(os.path.join(video_dir, "*.json")))
    return jsons[0] if len(jsons) == 1 else None


def _find_images_dir(video_dir: str) -> Optional[str]:
    """CholecTrack20 uses ``Frames/``; other releases may use ``images/``."""
    preferred = ("frames", "images", "img", "frame")
    try:
        children = os.listdir(video_dir)
    except OSError:
        return None
    by_lower = {
        name.lower(): name
        for name in children
        if os.path.isdir(os.path.join(video_dir, name))
    }
    for sub in preferred:
        if sub in by_lower:
            return os.path.join(video_dir, by_lower[sub])
    return None


def _load_annotations_dict(json_path: str) -> Dict[str, list]:
    """Load frame -> [tool records] mapping from a CholecTrack20 JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "annotations" in data:
        return data["annotations"]
    if isinstance(data, dict):
        # Some exports may store annotations at the top level.
        sample_key = next(iter(data.keys()), None)
        if sample_key is not None and isinstance(data[sample_key], list):
            return data
    raise ValueError(
        f"Unrecognized JSON layout in {json_path}. "
        "Expected top-level key 'annotations' mapping frame_id -> [records]."
    )


def _phase_categories_from_json(json_path: str) -> Dict[int, str]:
    """Try to read phase id -> name from JSON metadata; else use defaults."""
    with open(json_path, "r") as f:
        data = json.load(f)
    mapping = dict(CHOLECTRACK20_PHASE_ID_TO_NAME)
    categories = data.get("categories") or data.get("phase_categories")
    if isinstance(categories, list):
        for cat in categories:
            if not isinstance(cat, dict):
                continue
            name = cat.get("name") or cat.get("phase") or cat.get("label")
            cid = cat.get("id")
            if name is not None and cid is not None:
                mapping[int(cid)] = str(name).replace(" ", "")
    return mapping


def _extract_phase_from_records(
    records: list, phase_map: Dict[int, str]
) -> Optional[str]:
    """Read phase label from one frame's tool records (same for all tools)."""
    if not records:
        return None
    phase_keys = ("phase", "phase_id", "phase_identity", "phase_label")
    values = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        for key in phase_keys:
            if key not in rec:
                continue
            raw = rec[key]
            if isinstance(raw, str):
                values.append(raw.replace(" ", ""))
            elif isinstance(raw, (int, float)):
                pid = int(raw)
                if pid in phase_map:
                    values.append(phase_map[pid])
                else:
                    values.append(f"Phase{pid}")
            break
    if not values:
        return None
    # All tools in a frame should share the same phase; take the mode.
    return Counter(values).most_common(1)[0][0]


def _frame_image_path(images_dir: str, frame_id: str) -> Optional[str]:
    """Resolve the on-disk image for a JSON frame id."""
    fid = str(int(frame_id))
    patterns = [
        os.path.join(images_dir, f"{int(fid):06d}.png"),
        os.path.join(images_dir, f"{int(fid):06d}.jpg"),
        os.path.join(images_dir, f"frame_{int(fid):06d}.png"),
        os.path.join(images_dir, f"frame_{int(fid):06d}.jpg"),
        os.path.join(images_dir, f"{fid}.png"),
        os.path.join(images_dir, f"{fid}.jpg"),
    ]
    for path in patterns:
        if os.path.isfile(path):
            return path
    return None


def build_from_cholectrack20_dir(
    root_dir: str,
    output_csv: str,
    splits: Optional[List[str]] = None,
    video_ids: Optional[List[str]] = None,
    write_extra_columns: bool = True,
) -> None:
    """Scan an extracted CholecTrack20 tree and write ``frame_path,phase`` CSV.

    Expected layout (after unzipping the Synapse release):

        CholecTrack20/
            training/
                VID01/
                    VID01.json
                    images/000000.png
                    images/000001.png
                    ...
                VID02/
                    ...
            validation/
                ...
            testing/
                ...

    Frames are already sampled at **1 fps** (~35k images total for all 20
    videos). You can delete the ``.mp4`` files inside each video folder to
    save disk space once you confirm the ``images/`` folder is present.

    Parameters
    ----------
    root_dir:
        Path to ``CholecTrack20/`` (or ``RELEASE/CholecTrack20/``).
    splits:
        Which split folders to include. Default: training, validation, testing.
    video_ids:
        Optional whitelist, e.g. ``["VID01", "VID02"]`` to build a smaller CSV
        while downloading / experimenting.
    """
    splits = splits or list(DEFAULT_SPLITS)
    splits = [_normalize_split_name(s) for s in splits]

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(
            f"CholecTrack20 root not found: {root_dir}\n"
            "Download from https://www.synapse.org/Synapse:syn53182642/wiki/ "
            "(request form + access key on the CAMMA CholecTrack20 README)."
        )

    rows: List[Tuple] = []
    n_skip_no_json = 0
    n_skip_no_images = 0
    n_skip_no_phase = 0

    for split in splits:
        split_root = _find_split_root(root_dir, split)
        if split_root is None:
            log(f"Split folder not found (skipping): {split} under {root_dir}")
            continue
        log(f"Scanning split '{os.path.basename(split_root)}' at {split_root} ...")
        for video_dir in _find_video_dirs(split_root):
            video_id = os.path.basename(video_dir.rstrip(os.sep))
            if video_ids is not None and video_id not in video_ids:
                continue

            json_path = _find_json_for_video(video_dir)
            if json_path is None:
                n_skip_no_json += 1
                log(f"  skip {video_id}: no JSON annotation found.")
                continue

            images_dir = _find_images_dir(video_dir)
            if images_dir is None:
                n_skip_no_images += 1
                log(f"  skip {video_id}: no images/ or frames/ subdirectory.")
                continue

            phase_map = _phase_categories_from_json(json_path)
            annotations = _load_annotations_dict(json_path)

            for frame_id in sorted(annotations.keys(), key=lambda x: int(x)):
                records = annotations[frame_id]
                phase = _extract_phase_from_records(records, phase_map)
                if phase is None:
                    n_skip_no_phase += 1
                    continue
                img_path = _frame_image_path(images_dir, frame_id)
                if img_path is None:
                    continue
                if write_extra_columns:
                    rows.append((img_path, phase, video_id, split, int(frame_id)))
                else:
                    rows.append((img_path, phase))

    if not rows:
        raise RuntimeError(
            "No (frame_path, phase) rows produced from CholecTrack20.\n"
            "Check that:\n"
            "  - root_dir points at the unzipped dataset (contains training/ etc.)\n"
            "  - each VIDxx folder has VIDxx.json and images/*.png\n"
            "  - JSON records include a 'phase' or 'phase_id' field\n"
            f"Skipped: no_json={n_skip_no_json}, no_images={n_skip_no_images}, "
            f"no_phase={n_skip_no_phase}"
        )

    ensure_dir(os.path.dirname(output_csv) or ".")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        if write_extra_columns:
            writer.writerow(
                ["frame_path", "phase", "video_id", "split", "frame_id"]
            )
        else:
            writer.writerow(["frame_path", "phase"])
        writer.writerows(rows)

    n_videos = len({r[2] for r in rows}) if write_extra_columns else "?"
    n_phases = len({r[1] for r in rows})
    log(
        f"CholecTrack20 metadata ready: {len(rows)} frames, "
        f"{n_phases} phases, videos={n_videos}. Wrote {output_csv}"
    )
    if n_skip_no_phase:
        log(f"  ({n_skip_no_phase} JSON frames had no phase field and were skipped.)")


# -----------------------------------------------------------------------------
# Cholec80 (legacy)
# -----------------------------------------------------------------------------


def _parse_cholec80_phase_file(path: str, fps_target: float) -> List[Tuple[int, str]]:
    """Parse a Cholec80 phase annotation file (TSV: frame, phase)."""
    rows: List[Tuple[int, str]] = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i == 0 and not line.split()[0].isdigit():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            frame_idx = int(parts[0])
            phase = parts[1]
            rows.append((frame_idx, phase))
    return rows


def build_from_cholec80_dir(
    cholec80_dir: str,
    output_csv: str,
    frames_subdir: str = "frames",
    phases_subdir: str = "phase_annotations",
    frame_glob: str = "*.jpg",
    fps_target: float = 1.0,
) -> None:
    """Scan a Cholec80-style layout and emit a metadata CSV (legacy)."""
    frames_root = os.path.join(cholec80_dir, frames_subdir)
    phases_root = os.path.join(cholec80_dir, phases_subdir)
    if not os.path.isdir(frames_root):
        raise FileNotFoundError(f"frames directory not found: {frames_root}")
    if not os.path.isdir(phases_root):
        raise FileNotFoundError(f"phases directory not found: {phases_root}")

    rows: List[Tuple[str, str]] = []
    video_dirs = sorted(
        d for d in os.listdir(frames_root) if os.path.isdir(os.path.join(frames_root, d))
    )
    log(f"Scanning {len(video_dirs)} video directories under {frames_root}.")
    for video in video_dirs:
        frame_files = sorted(glob.glob(os.path.join(frames_root, video, frame_glob)))
        if not frame_files:
            continue
        candidates = [
            os.path.join(phases_root, f"{video}-phase.txt"),
            os.path.join(phases_root, f"{video}.txt"),
            os.path.join(phases_root, f"{video}_phase.txt"),
        ]
        phase_file = next((c for c in candidates if os.path.exists(c)), None)
        if phase_file is None:
            log(f"  skip {video}: no phase annotation file found.")
            continue

        phase_rows = _parse_cholec80_phase_file(phase_file, fps_target=fps_target)
        phase_by_frame = dict(phase_rows)

        for fpath in frame_files:
            name = os.path.splitext(os.path.basename(fpath))[0]
            digits = "".join(ch for ch in name if ch.isdigit())
            if not digits:
                continue
            frame_idx = int(digits)
            phase = phase_by_frame.get(frame_idx)
            if phase is None:
                lower_keys = [k for k in phase_by_frame.keys() if k <= frame_idx]
                if not lower_keys:
                    continue
                phase = phase_by_frame[max(lower_keys)]
            rows.append((fpath, phase))

    if not rows:
        raise RuntimeError(
            "No (frame_path, phase) pairs produced; check directory layout."
        )

    ensure_dir(os.path.dirname(output_csv) or ".")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_path", "phase"])
        writer.writerows(rows)
    log(f"Wrote {len(rows)} rows to {output_csv}.")


# -----------------------------------------------------------------------------
# Synthetic smoke-test data
# -----------------------------------------------------------------------------


def _color_for_phase(phase: str) -> Tuple[int, int, int]:
    h = abs(hash(phase))
    return (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)


def build_synthetic(
    output_csv: str,
    frames_dir: str,
    n_videos: int = 3,
    frames_per_phase: int = 8,
    phases: Optional[List[str]] = None,
    image_size: int = 64,
    seed: int = 0,
) -> None:
    """Render tiny solid-color JPEGs per phase (pipeline smoke test only)."""
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError("Pillow is required. pip install Pillow") from e
    import numpy as np

    rng = np.random.default_rng(seed)
    phases = list(phases or SURGICAL_PHASES)

    ensure_dir(frames_dir)
    rows: List[Tuple[str, str]] = []
    for vid in range(1, n_videos + 1):
        video_name = f"video{vid:02d}"
        video_dir = os.path.join(frames_dir, video_name)
        ensure_dir(video_dir)
        frame_idx = 0
        for phase in phases:
            base = np.array(_color_for_phase(phase), dtype=np.float32)
            for _ in range(frames_per_phase):
                noise = rng.normal(loc=0, scale=15, size=(image_size, image_size, 3))
                img = np.clip(base[None, None, :] + noise, 0, 255).astype("uint8")
                fname = f"frame_{frame_idx:06d}.jpg"
                fpath = os.path.join(video_dir, fname)
                Image.fromarray(img).save(fpath, quality=80)
                rows.append((fpath, phase))
                frame_idx += 1

    random.Random(seed).shuffle(rows)
    ensure_dir(os.path.dirname(output_csv) or ".")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_path", "phase"])
        writer.writerows(rows)
    log(
        f"Synthetic dataset ready: {len(rows)} frames across "
        f"{n_videos} videos and {len(phases)} phases. Metadata: {output_csv}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    p_ct = sub.add_parser(
        "cholectrack20",
        help="Build CSV from extracted CholecTrack20 (recommended; smaller than Cholec80).",
    )
    p_ct.add_argument(
        "--root_dir",
        required=True,
        help="Unzipped CholecTrack20 root (contains training/, validation/, testing/).",
    )
    p_ct.add_argument(
        "--output_csv",
        default="data/cholectrack20_metadata.csv",
    )
    p_ct.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split folders to include (default: training validation testing).",
    )
    p_ct.add_argument(
        "--video_ids",
        nargs="*",
        default=None,
        help="Optional whitelist, e.g. VID01 VID02 (for a smaller pilot run).",
    )

    p80 = sub.add_parser(
        "cholec80",
        help="Build CSV from legacy Cholec80 frame + TSV layout.",
    )
    p80.add_argument("--cholec80_dir", required=True)
    p80.add_argument("--output_csv", default="data/cholec80_metadata.csv")
    p80.add_argument("--frames_subdir", default="frames")
    p80.add_argument("--phases_subdir", default="phase_annotations")
    p80.add_argument("--frame_glob", default="*.jpg")

    p_synth = sub.add_parser("synthetic", help="Toy smoke-test data only.")
    p_synth.add_argument("--output_csv", default="data/synthetic_metadata.csv")
    p_synth.add_argument("--frames_dir", default="data/synthetic_frames")
    p_synth.add_argument("--n_videos", type=int, default=3)
    p_synth.add_argument("--frames_per_phase", type=int, default=8)
    p_synth.add_argument("--seed", type=int, default=0)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "cholectrack20":
        build_from_cholectrack20_dir(
            root_dir=args.root_dir,
            output_csv=args.output_csv,
            splits=args.splits,
            video_ids=args.video_ids,
        )
    elif args.mode == "cholec80":
        build_from_cholec80_dir(
            cholec80_dir=args.cholec80_dir,
            output_csv=args.output_csv,
            frames_subdir=args.frames_subdir,
            phases_subdir=args.phases_subdir,
            frame_glob=args.frame_glob,
        )
    elif args.mode == "synthetic":
        build_synthetic(
            output_csv=args.output_csv,
            frames_dir=args.frames_dir,
            n_videos=args.n_videos,
            frames_per_phase=args.frames_per_phase,
            seed=args.seed,
        )
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
