#!/usr/bin/env python
"""Extract video frames as JPEGs at a fixed frame rate.

StreamReady never decodes video at train or eval time. Every dataset is read as
pre-extracted frames laid out as::

    <frames_root>/<video_id>/frame%06d.jpg      (1-indexed)

Bare ``%06d.jpg`` dumps are also accepted by the loader, so an existing
extraction can be reused without renaming.

The default rate is **1 FPS** (paper supplementary Sec. D). That rate is load
bearing: ProReady-QA and the four streaming benchmarks annotate
``question_timestamp`` / ``evidence_start`` / ``evidence_end`` in *seconds*, and
the loaders convert seconds to frame indices with the ``fps`` recorded in the
annotation. Extract at a different rate and you must set ``fps`` in the
annotation to match, or every timestamp will land on the wrong frame.

Usage::

    # one video directory
    python data/extract_frames.py --src-dir /path/to/videos \
        --dst-dir data/proready_qa/frames --fps 1

    # videos nested in per-category subdirectories
    python data/extract_frames.py --src-dir /path/to/OVO-Bench/videos \
        --dst-dir data/ovo_bench/frames --fps 1 --recursive

    # check an annotation's frame_length against what is on disk
    python data/extract_frames.py --dst-dir data/ovo_bench/frames \
        --verify data/ovo_bench/annotation/ovo_bench_test.json

Requires the ``ffmpeg`` binary on PATH.

Note on frame counts: different ffmpeg builds can return slightly different
totals for the same video. Nothing downstream assumes a specific count --
``tools/convert_benchmarks.py`` counts frames on disk when it writes an
annotation, and ``--verify`` reports any annotation that has drifted.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # keep the script usable without the extra dependency
    def tqdm(it, **_kwargs):
        return it

VIDEO_SUFFIXES = {
    ".mp4", ".avi", ".mkv", ".webm", ".mov", ".m4v", ".mpg", ".mpeg", ".flv", ".ts",
}


def count_frames(d: Path) -> int:
    """Frames on disk for one video, under either accepted naming scheme."""
    if not d.is_dir():
        return 0
    n = len(list(d.glob("frame*.jpg")))
    return n if n else len(list(d.glob("*.jpg")))


def extract_one(
    video_path: Path,
    dst_root: Path,
    fps: float,
    scale_height: int,
    quality: int,
    overwrite: bool,
) -> tuple[str, bool, str]:
    """Extract one video. Returns ``(video_id, ok, message)``."""
    video_id = video_path.stem
    out_dir = dst_root / video_id

    if out_dir.is_dir() and not overwrite:
        existing = count_frames(out_dir)
        if existing > 0:
            return video_id, True, f"skipped ({existing} frames already present)"

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"scale=-1:{scale_height}",
        "-pix_fmt", "yuvj422p",
        "-q:v", str(quality),
        "-r", str(fps),
        "-y",
        str(out_dir / "frame%06d.jpg"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        return video_id, False, err[-1] if err else "ffmpeg failed"

    n = count_frames(out_dir)
    if n == 0:
        return video_id, False, "ffmpeg produced no frames"
    return video_id, True, f"{n} frames"


def _expected_counts(path: Path):
    """``{video_id: expected_frame_count}`` from either accepted file shape.

    Two shapes turn up, so both are read rather than making the caller care:

    * ProReady-QA's ``frames_manifest.json`` -- a flat object keyed by
      ``video_id``, each entry carrying ``frame_count``. This is the file the
      benchmark ships for exactly this purpose.
    * a converted annotation -- an object with a ``videos`` list, each entry
      carrying ``video_id`` and ``frame_length``.
    """
    with open(path) as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "videos" in payload:
        out = {}
        for v in payload["videos"]:
            vid = str(v.get("video_id", ""))
            if vid and v.get("frame_length") is not None:
                out[vid] = int(v["frame_length"])
        return out, "annotation"

    if isinstance(payload, dict) and payload and all(
        isinstance(v, dict) for v in payload.values()
    ):
        out = {
            str(k): int(v["frame_count"])
            for k, v in payload.items()
            if v.get("frame_count") is not None
        }
        if out:
            return out, "manifest"

    sys.exit(
        f"{path}: expected either ProReady-QA's frames_manifest.json (an object "
        f"keyed by video_id with `frame_count`) or an annotation with a `videos` "
        f"list carrying `frame_length`."
    )


def verify(ann_path: Path, frames_root: Path) -> int:
    """Check the frames on disk against their expected counts.

    Reports only; nothing is rewritten, because the shipped files define the
    benchmark.

    A count mismatch is the usual symptom of extracting at the wrong frame rate,
    which is the failure this is here to catch: ProReady-QA's timestamps are in
    the same units as frame indices, so a wrong rate silently puts every
    annotation on the wrong frame instead of erroring.
    """
    expected, kind = _expected_counts(ann_path)

    missing, drifted, ok = [], [], 0
    for vid, want in sorted(expected.items()):
        n = count_frames(frames_root / vid)
        if n == 0:
            missing.append(vid)
        elif n != want:
            drifted.append((vid, want, n))
        else:
            ok += 1

    print(f"{ann_path}  ({kind})")
    print(f"  videos          {len(expected)}")
    print(f"  frame count OK  {ok}")
    print(f"  no frames       {len(missing)}")
    print(f"  wrong count     {len(drifted)}")
    for vid in missing[:10]:
        print(f"    missing: {vid}")
    if len(missing) > 10:
        print(f"    ... and {len(missing) - 10} more")
    for vid, want, got in drifted[:10]:
        print(f"    wrong count: {vid} expected={want} on_disk={got}")
    if len(drifted) > 10:
        print(f"    ... and {len(drifted) - 10} more")

    if kind == "manifest":
        # The manifest also carries a `frame_digest`. Reproducing it needs the
        # exporter's exact serialisation of the (filename, size) pairs, which is
        # not documented here, so it is deliberately left unchecked rather than
        # guessed at and reported as a spurious mismatch.
        print("  note: frame_digest is not checked, only frame_count")

    if drifted:
        print("\nA wrong count usually means the wrong extraction rate. "
              "ProReady-QA is 1 FPS.")
    if missing or drifted:
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src-dir", help="directory holding the video files")
    ap.add_argument("--dst-dir", required=True,
                    help="frame output root; one subdirectory per video")
    ap.add_argument("--fps", type=float, default=1.0,
                    help="extraction frame rate; keep at 1 unless you also set "
                         "`fps` in the annotation (default: 1)")
    ap.add_argument("--scale-height", type=int, default=256,
                    help="scale the short side to this many pixels (default: 256)")
    ap.add_argument("--quality", type=int, default=1,
                    help="ffmpeg -q:v, 1 is best (default: 1)")
    ap.add_argument("--num-workers", type=int, default=4,
                    help="parallel ffmpeg processes (default: 4)")
    ap.add_argument("--recursive", action="store_true",
                    help="search --src-dir recursively, for datasets that nest "
                         "videos in per-category subdirectories")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-extract videos that already have frames")
    ap.add_argument("--verify", metavar="ANNOTATION_JSON",
                    help="instead of extracting, compare this annotation's "
                         "frame_length against the frames in --dst-dir")
    args = ap.parse_args()

    dst_dir = Path(args.dst_dir)

    if args.verify:
        sys.exit(verify(Path(args.verify), dst_dir))

    if not args.src_dir:
        ap.error("--src-dir is required unless --verify is given")
    src_dir = Path(args.src_dir)

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH. Install it "
                 "(e.g. `conda install -c conda-forge ffmpeg`).")
    if not src_dir.is_dir():
        sys.exit(f"video directory not found: {src_dir}")

    walk = src_dir.rglob("*") if args.recursive else src_dir.iterdir()
    videos = sorted(
        p for p in walk
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        sys.exit(f"no video files found in {src_dir} "
                 f"(looked for {sorted(VIDEO_SUFFIXES)}"
                 f"{', recursively' if args.recursive else ''})")

    # Frame directories are keyed by the file stem, so two videos with the same
    # stem in different subdirectories would overwrite each other.
    stems: dict[str, Path] = {}
    clashes = []
    for p in videos:
        if p.stem in stems:
            clashes.append((p.stem, stems[p.stem], p))
        else:
            stems[p.stem] = p
    if clashes:
        print(f"ERROR: {len(clashes)} video(s) share a filename stem and would "
              f"collide in {dst_dir}:", file=sys.stderr)
        for stem, first, second in clashes[:10]:
            print(f"  {stem}: {first} and {second}", file=sys.stderr)
        sys.exit("Rename them, or extract each subdirectory to its own --dst-dir.")

    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {len(videos)} videos at fps={args.fps:g}")
    print(f"  from : {src_dir}")
    print(f"  to   : {dst_dir}")

    failures = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [
            pool.submit(extract_one, v, dst_dir, args.fps, args.scale_height,
                        args.quality, args.overwrite)
            for v in videos
        ]
        for fut in tqdm(futures, total=len(futures), unit="video"):
            video_id, ok, msg = fut.result()
            if not ok:
                failures.append((video_id, msg))

    print(f"\nDone: {len(videos) - len(failures)}/{len(videos)} videos extracted.")
    print("Now write the annotation with tools/convert_benchmarks.py, which counts "
          "these frames so the result is loadable by construction.")
    if failures:
        print(f"\n{len(failures)} failed:")
        for video_id, msg in failures[:20]:
            print(f"  {video_id}: {msg}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        sys.exit(1)


if __name__ == "__main__":
    main()
