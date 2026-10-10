"""
ProReady-QA -- StreamReady paper Sec. 4.

Proactive, multi-turn QA over long streaming video, where every question is asked
at a timestamp ``question_timestamp`` and can only be answered from evidence that
appears later, inside an annotated window ``[evidence_start, evidence_end]``.

One dataset item is one QA *turn*. Preceding turns of the same dialogue travel
with it as ``history`` so the Contextual Memory Bank can be primed before the
current question is asked; this is what makes the local/global multi-turn
dependencies of Sec. 4.2 reachable without variable-length collation.

ProReady-QA is **evaluation-only**. The benchmark measures *when* a model
answers, so there is no train split: training on it would defeat its purpose.
StreamReady is trained on the separate corpus of supplementary Table 11 (the
``video_qa_train`` builder) and evaluated here.

Released annotation schema (one JSON file)::

    {
      "benchmark": "ProReady-QA",
      "videos": [
        {
          "video_id": "ego4d_000000",
          "source": "ego4d",                 # or "movienet"
          "fps": 1.0,                        # frame rate of the extracted frames
          "frame_length": 3600,              # number of frames on disk
          "duration": 3600.0,                # seconds
          "dialogues": [
            {
              "dialogue_id": "ego4d_000000_d00",
              "turns": [
                {
                  "qa_id": "ego4d_000000_000001",
                  "task": "SSR",             # SSR | REC | CRR | CTD | GSD
                  "question": "...",
                  "answer": "...",
                  "question_timestamp": 180.0,
                  "evidence_start": 214.0,
                  "evidence_end": 263.0,
                  "difficulty": "easy"       # optional
                }
              ]
            }
          ]
        }
      ]
    }

Frames are read from ``<vis_root>/<video_id>/frame%06d.jpg`` (1-indexed), the same
layout the rest of the repo uses, produced by ``data/extract_frames.py``.

Timestamps are in **seconds** throughout. ``fps`` converts to frame indices, so an
annotation set stays valid if frames are re-extracted at a different rate.
"""

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.frame_loading import FrameLoaderMixin, sample_segment_indices
# The readiness cue lives beside MC_PROMPT because both are prompt shaping,
# and both benchmarks must append it identically.
from lavis.datasets.datasets.streaming_qa_datasets import append_readiness_cue

TASKS = ("SSR", "REC", "CRR", "CTD", "GSD")


class ProReadyQADataset(FrameLoaderMixin, BaseDataset):
    """Streaming proactive QA with annotated answer evidence windows.

    Each item yields the frames the model is allowed to observe, the question, the
    target answer, and the timing annotation needed by ARS.
    """

    # Evaluation-only benchmark, so sampling is deterministic. Kept as a class
    # attribute because FrameLoaderMixin reads it.
    randomize_frames = False

    def __init__(
        self,
        vis_processor,
        text_processor,
        ann_paths,
        vis_root,
        num_frames,
        prompt="",
        cached_vit_features_root=None,
        observe_full_video=True,
        frames_manifest=None,
        skip_missing_videos=True,
        readiness_cue="",
    ):
        # BaseDataset's __init__ expects a flat annotation list; items here are
        # whole dialogues, so the grouping happens below and the parent is given
        # an empty list.
        super().__init__(vis_processor, text_processor, vis_root, ann_paths=[])

        self.vis_root = vis_root
        self.num_frames = num_frames
        self.prompt = prompt or ""
        self.cached_vit_features_root = cached_vit_features_root
        self.frames_manifest_path = frames_manifest
        self.readiness_cue = readiness_cue or ""
        # Paper Sec. 5: offline baselines are fed "progressively truncated
        # prefixes of the full video", while streaming models "process frames
        # sequentially". Either way the model may observe the whole video and it
        # is the readiness signal that decides when to answer, so by default we
        # hand over the full stream rather than truncating at the evidence window
        # (which would leak the answer's location).
        self.observe_full_video = bool(observe_full_video)

        self.video_meta: Dict[str, Dict] = {}
        self.annotation: List[Dict] = []
        self.manifest = self._load_manifest(ann_paths, self.frames_manifest_path)

        for path in ann_paths:
            with open(path, "r") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                self._ingest_flat(payload)
            else:
                self._ingest_nested(payload)

        # One item per dialogue, ordered by first ask time so a run walks the
        # benchmark in a stable, readable order.
        self.annotation.sort(
            key=lambda d: (d["video_id"], d["turns"][0]["question_timestamp"])
        )
        self.skipped_videos: List[str] = []
        if skip_missing_videos:
            self._drop_unloadable_videos()
        self._add_instance_ids()

    def _drop_unloadable_videos(self):
        """Drop dialogues whose video has no frames, and report it at startup.

        The released annotation is read in place, so it describes every video in
        the benchmark whether or not you hold an extraction of each. Ego4D needs
        its own licence agreement, so a partial frame tree is the normal case,
        and evaluating one source at a time is legitimate.

        Without this check the first unextracted video raises
        ``FileNotFoundError`` from a DataLoader worker *partway through* the run
        -- after everything before it has already been scored and thrown away.
        Since items are ordered by video id, that can be an hour in. Reporting
        the gap up front is the difference between a scope statement and a
        late crash.

        The message names the evaluated subset explicitly, because metrics
        computed over 10 of 32 videos must not be mistaken for a full result.
        """
        missing = sorted(v for v in self.video_meta if not self.has_clip(v))
        if not missing:
            return

        kind = "cached features" if self.uses_cached_vit_features else "frames"
        before = len(self.annotation)
        self.annotation = [
            d for d in self.annotation if d["video_id"] not in set(missing)
        ]
        for v in missing:
            self.video_meta.pop(v, None)
        self.skipped_videos = missing

        if not self.annotation:
            raise RuntimeError(
                f"ProReady-QA: none of the {len(missing)} video(s) in the "
                f"annotation have {kind} under {self.vis_root!r}. Extract them "
                f"with data/extract_frames.py at 1 FPS, or point the dataset's "
                f"`images.storage` at an existing dump. Missing: "
                f"{', '.join(missing[:5])}"
                f"{' ...' if len(missing) > 5 else ''}"
            )

        logging.warning(
            "ProReady-QA: %d of %d video(s) have no %s, so their %d dialogue(s) "
            "are skipped. EVALUATING A SUBSET: %d video(s), %d dialogue(s), "
            "%d question(s). Missing: %s%s",
            len(missing), len(missing) + len(self.video_meta), kind,
            before - len(self.annotation),
            len(self.video_meta), len(self.annotation),
            sum(len(d["turns"]) for d in self.annotation),
            ", ".join(missing[:5]), " ..." if len(missing) > 5 else "",
        )

    # ------------------------------------------------------------- ingestion
    @staticmethod
    def _load_manifest(ann_paths, explicit) -> Dict[str, Dict]:
        """Per-video ``frame_count`` from the benchmark's ``frames_manifest.json``.

        The released annotation is a flat list of QA records and carries no
        per-video frame count -- that lives in the manifest shipped beside it,
        which also exists so you can verify an extraction. Looked up next to the
        annotation files unless given explicitly; absent is not an error, since
        ``duration`` can stand in at 1 FPS.
        """
        candidates = []
        if explicit:
            candidates.append(explicit)
        for p in ann_paths:
            candidates.append(os.path.join(os.path.dirname(p), "frames_manifest.json"))
        for c in candidates:
            if c and os.path.isfile(c):
                with open(c, "r") as f:
                    man = json.load(f)
                if isinstance(man, dict):
                    logging.info("ProReady-QA: read frame counts from %s", c)
                    return man
        logging.warning(
            "ProReady-QA: no frames_manifest.json found next to the annotation; "
            "falling back to `duration` for frame counts, which is correct only "
            "at 1 FPS."
        )
        return {}

    def _register_video(self, vid: str, duration: float, source: str, fps: float):
        """Record one video's frame geometry, preferring the manifest's count."""
        if vid in self.video_meta:
            return
        entry = self.manifest.get(vid, {})
        frame_length = entry.get("frame_count")
        if frame_length is None:
            # At 1 FPS one second is one frame, which is the released rate.
            frame_length = int(round(duration * fps))
        self.video_meta[vid] = {
            "fps": float(fps),
            "frame_length": max(2, int(frame_length)),
            "duration": float(duration),
            "source": entry.get("source", source),
        }

    def _ingest_flat(self, records: List[Dict]):
        """The released schema: a flat list of QA records.

        Dialogues are implicit -- ``dialogue_id`` groups turns and
        ``question_timestamp`` orders them, which is exactly the
        ``groupby("dialogue_id")`` the benchmark's own documentation describes.
        Field names are aliased rather than requiring a conversion pass:
        ``id -> qa_id`` and ``answer_type -> task``.
        """
        by_dialogue: Dict[str, List[Dict]] = {}
        for r in records:
            vid = str(r["video_id"])
            self._register_video(
                vid,
                duration=float(r.get("duration", 0.0)),
                source=str(r.get("source", "unknown")),
                fps=float(r.get("fps", 1.0)),
            )
            turn = dict(r)
            turn["qa_id"] = r.get("qa_id", r.get("id"))
            turn["task"] = r.get("task", r.get("answer_type", "unknown"))
            by_dialogue.setdefault(
                (vid, str(r.get("dialogue_id") or turn["qa_id"])), []
            ).append(turn)

        for (vid, did), turns in by_dialogue.items():
            self.annotation.append({
                "video_id": vid,
                "dialogue_id": did,
                # Ask-time order is what the stream services turns in. The
                # released `turn_index` agrees with it, but is not relied on:
                # some dialogues are not numbered from 0.
                "turns": sorted(
                    turns, key=lambda x: float(x.get("question_timestamp", 0) or 0)
                ),
            })

    def _ingest_nested(self, payload: Dict):
        """The pre-grouped schema: ``videos -> dialogues -> turns``.

        Kept so an already-converted annotation still loads.
        """
        for video in payload.get("videos", []):
            vid = str(video["video_id"])
            fps = float(video.get("fps", 1.0))
            frame_length = video.get("frame_length")
            duration = float(
                video.get(
                    "duration",
                    (int(frame_length) / max(fps, 1e-6)) if frame_length else 0.0,
                )
            )
            self._register_video(vid, duration, video.get("source", "unknown"), fps)
            if frame_length:
                self.video_meta[vid]["frame_length"] = max(2, int(frame_length))
            for dlg in video.get("dialogues", []):
                turns = dlg.get("turns", [])
                if not turns:
                    continue
                normalised = []
                for t in turns:
                    turn = dict(t)
                    turn["qa_id"] = t.get("qa_id", t.get("id"))
                    turn["task"] = t.get("task", t.get("answer_type", "unknown"))
                    normalised.append(turn)
                self.annotation.append({
                    "video_id": vid,
                    "dialogue_id": dlg.get("dialogue_id", vid),
                    "turns": sorted(
                        normalised,
                        key=lambda x: float(x.get("question_timestamp", 0) or 0),
                    ),
                })

    def __len__(self):
        return len(self.annotation)

    def _frame_times(self, frame_indices, fps: float) -> List[float]:
        """Seconds corresponding to each sampled frame index."""
        return [float(idx) / max(fps, 1e-6) for idx in frame_indices]

    @staticmethod
    def _ask_steps(frame_times: List[float], turns: List[Dict]) -> List[int]:
        """Index of the first sampled frame at or after each question's ask time.

        The stream the model sees is ``num_frames`` samples spanning the video,
        not every frame, so a question asked at 180 s has to be placed on that
        sampled grid. Clamped into range: a question asked past the last sampled
        frame is serviced at the end of the stream, which ARS scores as late.
        """
        steps = []
        n = len(frame_times)
        for t in turns:
            q_t = float(t.get("question_timestamp", 0) or 0)
            k = n - 1
            for i, ft in enumerate(frame_times):
                if ft >= q_t:
                    k = i
                    break
            steps.append(int(k))
        # Non-decreasing: turns are answered in order, so their ask steps must
        # not go backwards even if two questions land on the same frame.
        for i in range(1, len(steps)):
            steps[i] = max(steps[i], steps[i - 1])
        return steps

    def __getitem__(self, index):
        """One whole dialogue: the video once, plus every turn in ask order."""
        ann = self.annotation[index]
        meta = self.video_meta[ann["video_id"]]
        fps = meta["fps"]
        turns = ann["turns"]

        frame_length = meta["frame_length"]
        if not self.observe_full_video:
            # Debug switch: truncate just past the last turn's evidence window.
            last_end = max(float(t["evidence_end"]) for t in turns)
            horizon = int(round(last_end * fps)) + 1
            frame_length = max(2, min(frame_length, horizon))

        frame_indices = sample_segment_indices(
            frame_length, self.num_frames, randomize=self.randomize_frames
        )
        video = self.load_clip(ann["video_id"], frame_indices)
        frame_times = self._frame_times(frame_indices, fps)

        questions, prompts = [], []
        for t in turns:
            q = self.text_processor(t["question"])
            questions.append(q)
            base = (
                self.prompt.format(q) if "{}" in self.prompt
                else (self.prompt + q if self.prompt else q)
            )
            prompts.append(append_readiness_cue(base, self.readiness_cue))

        return {
            "image": video,
            # One entry per turn, in the order they are asked.
            "text_input": prompts,
            "text_output": [t["answer"] for t in turns],
            "ask_steps": self._ask_steps(frame_times, turns),
            # --- identity ---
            "video_id": ann["video_id"],
            "dialogue_id": ann["dialogue_id"],
            "qa_id": [
                t.get("qa_id", f"{ann['video_id']}_{index}_{i}")
                for i, t in enumerate(turns)
            ],
            "task": [t.get("task", "unknown") for t in turns],
            "turn_index": list(range(len(turns))),
            # --- timing annotation, in seconds (consumed by lavis.metrics.ars) ---
            "question_timestamp": [float(t["question_timestamp"]) for t in turns],
            "evidence_start": [float(t["evidence_start"]) for t in turns],
            "evidence_end": [float(t["evidence_end"]) for t in turns],
            "duration": meta["duration"],
            # Maps a readiness trigger at frame k back to a wall-clock answer time.
            "frame_times": torch.tensor(frame_times, dtype=torch.float32),
        }

    def collater(self, samples):
        """Batch dialogues without flattening their turn structure.

        Every per-turn field stays a list-of-lists, one inner list per dialogue,
        because turn counts differ. ``stream_dialogue`` processes one dialogue at
        a time anyway, so nothing here needs padding.
        """
        if len(samples) == 0:
            return {}
        out = {
            "image": torch.stack([s["image"] for s in samples], dim=0),
            "frame_times": torch.stack([s["frame_times"] for s in samples], dim=0),
        }
        for key in (
            "text_input", "text_output", "ask_steps", "qa_id", "task",
            "turn_index", "question_timestamp", "evidence_start", "evidence_end",
            "video_id", "dialogue_id", "duration",
        ):
            out[key] = [s[key] for s in samples]
        return out


class ProReadyQAEvalDataset(ProReadyQADataset):
    """The split the builder constructs.

    ProReady-QA is evaluation-only, so this exists to make the deterministic
    frame sampling explicit at the call site rather than implied by a default.
    """

    randomize_frames = False
