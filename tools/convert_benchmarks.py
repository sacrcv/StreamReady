"""
Convert OVO-Bench into StreamReady's unified annotation schema (see
lavis/datasets/datasets/streaming_qa_datasets.py).

OVO-Bench ships one flat JSON holding three different record shapes, which this
normalises away:

=================================  ======================  ===================
tasks                              answer encoding         options
=================================  ======================  ===================
backward:    EPM, ASI, HLD         0-based index ``gt``    bare text
real-time:   STU, OJR, ATR, ACR,   0-based index ``gt``    bare text
             OCR, FPD
forward:     REC                   integer count           none (open-ended)
             SSR, CRR              Yes / No                none (open-ended)
=================================  ======================  ===================

Every record carries its own ask time (``realtime``), and the forward tasks put
one question per ``test_info`` entry, each at its own ask time. The loader cuts
the stream there, so a model never sees past the moment the question was asked.

Frames are expected at ``<frames-root>/<video_id>/``. ``video_id`` is sanitised
so a nested native id such as ``Ego4D/clips/abc.mp4`` becomes a single flat
directory name ``Ego4D__clips__abc``. ``frame_length`` is counted from disk when
the frames are present and otherwise estimated from the annotation's duration,
with the loader clamping indices either way.

Usage:
    python tools/convert_benchmarks.py --benchmark ovo_bench \
        --raw /path/to/OVO-Bench \
        --frames-root data/ovo_bench/frames \
        --output data/ovo_bench/annotation/ovo_bench_test.json
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

CHOICE_LETTERS = "ABCDEFGH"


# --------------------------------------------------------------------- helpers
def as_list(value) -> List:
    """Coerce an options field to a list.

    Some dumps store options as numpy arrays, and ``value or []`` raises on
    those because an array's truth value is ambiguous.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):          # numpy array / pandas Series
        return list(value.tolist())
    return [value]


def flat_id(raw: str) -> str:
    """Flatten a nested native video id into one directory-safe name."""
    s = str(raw).strip()
    for ext in (".mp4", ".webm", ".mkv", ".avi", ".gif", ".MP4"):
        if s.endswith(ext):
            s = s[: -len(ext)]
    s = s.replace("\\", "/").strip("/")
    return s.replace("/", "__")


def count_frames(frames_root: str, video_id: str) -> int:
    d = os.path.join(frames_root, video_id)
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d)
               if f.lower().endswith((".jpg", ".jpeg", ".png")))


# When True (the default) a video is only emitted if its frames are on disk, so
# the resulting annotation is guaranteed loadable. Pass --no-require-frames to
# fall back to estimating the count from the annotation's duration, which is
# useful for preparing annotations before extracting frames.
REQUIRE_FRAMES = True


def resolve_length(frames_root: str, video_id: str, duration: Optional[float],
                   fps: float) -> int:
    """Frame count from disk; optionally estimated from duration as a fallback.

    The count matters because it is what the frame sampler indexes into. An
    estimate for a video whose frames are missing produces an annotation that
    fails at load time, so by default the video is dropped instead.
    """
    n = count_frames(frames_root, video_id)
    if n >= 2:
        return n
    if not REQUIRE_FRAMES and duration and duration > 0:
        return max(2, int(round(float(duration) * fps)))
    return 0


def letter_for_index(i: int) -> str:
    return CHOICE_LETTERS[i] if 0 <= i < len(CHOICE_LETTERS) else ""


def strip_option_prefix(opt: str) -> str:
    t = str(opt).strip()
    if len(t) > 2 and t[0] in CHOICE_LETTERS and t[1] in ".)":
        return t[2:].strip()
    return t


SKIPPED = {"no_frames": 0}


def note_skip():
    SKIPPED["no_frames"] += 1


def make_payload(benchmark: str, videos: List[Dict]) -> Dict:
    n_turns = sum(len(d["turns"]) for v in videos for d in v["dialogues"])
    tasks = defaultdict(int)
    for v in videos:
        for d in v["dialogues"]:
            for t in d["turns"]:
                tasks[t.get("task", "unknown")] += 1
    return {
        "benchmark": benchmark,
        "num_videos": len(videos),
        "num_qa": n_turns,
        "task_distribution": dict(sorted(tasks.items())),
        "videos": videos,
    }


def one_video(video_id: str, source: str, fps: float, frame_length: int,
              duration: float, turns: List[Dict]) -> Dict:
    """Pack turns for one video. Turns are ordered by ask time so a single
    streaming pass can answer them in sequence."""
    turns = sorted(turns, key=lambda t: float(t.get("question_timestamp", 0) or 0))
    return {
        "video_id": video_id,
        "source": source,
        "fps": fps,
        "frame_length": frame_length,
        "duration": round(float(duration), 2),
        "dialogues": [{"dialogue_id": f"{video_id}_d000", "turns": turns}],
    }


# ------------------------------------------------------------------- converter
def convert_ovo_bench(raw: str, frames_root: str, fps: float) -> Dict:
    """OVO-Bench: one flat JSON holding three different record shapes.

    Backward (EPM/ASI/HLD) and real-time (STU/OJR/ATR/ACR/OCR/FPD) are MCQ with a
    0-based ``gt``. The forward tasks are not MCQ: REC wants an integer count and
    SSR/CRR want Yes/No, with one question per ``test_info`` entry at its own ask
    time. Scoring for all of these is implemented in the video_qa task.
    """
    path = os.path.join(raw, "ovo_bench_new.json")
    if not os.path.isfile(path):
        hits = glob.glob(os.path.join(raw, "*.json"))
        if not hits:
            sys.exit(f"no ovo_bench_new.json under {raw}")
        path = hits[0]
    with open(path) as f:
        records = json.load(f)
    if isinstance(records, dict):           # some dumps nest by mode
        flat = []
        for v in records.values():
            flat.extend(v if isinstance(v, list) else [])
        records = flat

    by_video: Dict[str, List[Dict]] = defaultdict(list)
    for rec in records:
        task = rec.get("task", "unknown")
        video_id = flat_id(rec.get("video", ""))
        if not video_id:
            continue

        if task in ("REC", "SSR", "CRR"):
            for i, info in enumerate(rec.get("test_info", []) or []):
                t_ask = float(info.get("realtime", 0) or 0)
                turn = {
                    "qa_id": f"ovo_{rec.get('id')}_{i}",
                    "task": task,
                    "question_timestamp": t_ask,
                }
                if task == "REC":
                    turn["question"] = ("How many times did they "
                                        f"{rec.get('activity', 'do this')}?")
                    turn["answer"] = str(info.get("count", ""))
                elif task == "SSR":
                    turn["question"] = (
                        "Has the following step been completed: "
                        f"{info.get('step', '')}? Answer Yes or No.")
                    turn["answer"] = "Yes" if int(info.get("type", 0)) == 1 else "No"
                else:  # CRR
                    turn["question"] = (
                        f"{rec.get('question', '')} "
                        "Answer Yes if you can answer this now, otherwise No.")
                    turn["answer"] = "Yes" if int(info.get("type", 0)) == 1 else "No"
                by_video[video_id].append(turn)
        else:
            options = [str(o) for o in as_list(rec.get("options"))]
            gt = rec.get("gt")
            letter = letter_for_index(int(gt)) if isinstance(gt, int) else ""
            by_video[video_id].append({
                "qa_id": f"ovo_{rec.get('id')}",
                "task": task,
                "question": rec.get("question", ""),
                "options": [strip_option_prefix(o) for o in options],
                "answer_letter": letter,
                "answer": rec.get("answer", letter),
                "question_timestamp": float(rec.get("realtime", 0) or 0),
            })

    videos = []
    for vid, turns in sorted(by_video.items()):
        dur = max(float(t["question_timestamp"]) for t in turns) + 1.0
        n = resolve_length(frames_root, vid, dur, fps)
        if n < 2:
            note_skip()
        else:
            videos.append(one_video(vid, "ovo_bench", fps, n, dur, turns))
    return make_payload("ovo_bench", videos)


CONVERTERS = {
    "ovo_bench": convert_ovo_bench,
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--benchmark", choices=sorted(CONVERTERS), default="ovo_bench")
    ap.add_argument("--raw", help="directory holding the benchmark's native annotations")
    ap.add_argument("--frames-root", help="<root>/<video_id>/frame%%06d.jpg")
    ap.add_argument("--output")
    ap.add_argument("--fps", type=float, default=1.0,
                    help="frame rate the frames were extracted at (paper: 1 FPS)")
    ap.add_argument("--no-require-frames", action="store_true",
                    help="also emit videos whose frames are not on disk yet, "
                         "estimating frame_length from the annotation's duration. "
                         "Handy for preparing annotations ahead of extraction, but "
                         "the result is not loadable until the frames exist.")
    args = ap.parse_args()

    for req in ("raw", "frames_root", "output"):
        if not getattr(args, req):
            sys.exit(f"--{req.replace('_', '-')} is required")

    global REQUIRE_FRAMES
    REQUIRE_FRAMES = not args.no_require_frames

    payload = CONVERTERS[args.benchmark](args.raw, args.frames_root, args.fps)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"benchmark  {payload['benchmark']}")
    print(f"videos     {payload['num_videos']}")
    print(f"QA         {payload['num_qa']}")
    print(f"tasks      {len(payload['task_distribution'])}")
    if SKIPPED["no_frames"]:
        reason = ("frames not on disk" if REQUIRE_FRAMES
                  else "frames not on disk and no usable duration")
        print(f"skipped     {SKIPPED['no_frames']} video(s) ({reason})")
    if payload["num_videos"] == 0:
        print("\nNo videos had frames on disk or a usable duration. Extract frames "
              "first:\n  python data/extract_frames.py --src-dir <videos> "
              f"--dst-dir {args.frames_root} --fps {args.fps}")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
