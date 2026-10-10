"""
Unified video-QA loader, shared by OVO-Bench and the training corpus.

Native annotation formats disagree about almost everything -- answers as letters
vs indices vs full option text, options with or without ``"A. "`` prefixes, 2 to
5 options, ask timestamps present or absent. Rather than carry a bespoke
``Dataset`` class per source, everything is converted once into the single schema
below and read by this one class, which keeps the model and task code identical
across them. ``tools/convert_benchmarks.py`` does that for OVO-Bench; the
training corpus has to be converted to the same schema (see the README).

Unified schema, one JSON per split::

    {
      "benchmark": "ovo_bench",
      "videos": [
        {
          "video_id": "Ego4D_clips_ec4a3ba3",     # also the frame directory name
          "source": "ego4d",
          "fps": 1.0,
          "frame_length": 215,
          "duration": 215.0,
          "dialogues": [
            {
              "dialogue_id": "...",
              "turns": [
                {
                  "qa_id": "ovo_0",
                  "task": "EPM",
                  "question": "Who did I communicate to when chopping eggplants?",
                  "answer": "a person with blue shirt",
                  "options": ["a person with brown shirt", "..."],  # omit => open-ended
                  "answer_letter": "C",                             # MC only
                  "question_timestamp": 215.0,   # ask time; 0 for offline benchmarks
                  "evidence_start": 214.0,       # optional, enables ARS
                  "evidence_end": 263.0
                }
              ]
            }
          ]
        }
      ]
    }

Two behaviours are driven by the annotation rather than by config:

* **Streaming truncation.** When a turn has a non-zero ``question_timestamp`` and
  the dataset is built with ``truncate_at_question_time=True``, only frames up to
  that moment are observable. OVO-Bench defines its ask time this way and its
  official protocol cuts the video there, so a model that saw later frames would
  be reading the future. The training corpus has no ask time, so truncation is
  off there and the whole clip is observable.
* **Multiple choice.** When a turn carries ``options`` they are rendered into the
  prompt as lettered choices and the task scores by letter. Without them the turn
  is open-ended and scored by text match.
"""

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.frame_loading import FrameLoaderMixin, sample_segment_indices

CHOICE_LETTERS = "ABCDEFGH"

MC_PROMPT = (
    "{question}\n"
    "Options:\n{options}\n"
    "Answer with the option's letter from the given choices directly."
)

# Paper Sec. 5: "streaming models process frames sequentially with the prompt
# 'Answer whenever you are ready'". It is the streaming evaluation protocol, so
# it belongs on the benchmarks that have an ask time, not on training data.
READINESS_CUE = "Answer whenever you are ready."


def append_readiness_cue(text_input: str, cue: str) -> str:
    """Put the readiness instruction at the very end of an assembled prompt.

    Kept separate from the generic ``prompt`` prefix on purpose. The multiple
    choice branch builds its text from ``MC_PROMPT`` alone and ignores
    ``prompt`` entirely, so routing this through ``prompt`` would silently drop
    it for every MCQ task -- which is most of OVO-Bench. Appending after
    assembly means it survives both branches, and it lands after "Answer with
    the option's letter", which is where an instruction about *when* to answer
    reads correctly relative to one about *how*.
    """
    cue = (cue or "").strip()
    if not cue:
        return text_input
    return f"{text_input.rstrip()} {cue}"


def render_options(options: List[str]) -> str:
    """Render options as lettered lines, without double-prefixing.

    Sources disagree: some ship options that already read ``"A. foo"`` while
    OVO-Bench ships bare text. Re-prefixing the former would produce
    ``"A. A. foo"``, so an existing prefix is stripped and the letter renormalised
    to the option's position.
    """
    lines = []
    for i, opt in enumerate(options):
        text = str(opt).strip()
        letter = CHOICE_LETTERS[i]
        if len(text) > 2 and text[0] in CHOICE_LETTERS and text[1] in ".)":
            # Already prefixed; normalise the letter to its position.
            text = text[2:].strip()
        lines.append(f"{letter}. {text}")
    return "\n".join(lines)


class StreamingQADataset(FrameLoaderMixin, BaseDataset):
    """Video QA over pre-extracted frames, with optional ask-time truncation."""

    randomize_frames = True

    def __init__(
        self,
        vis_processor,
        text_processor,
        ann_paths,
        vis_root,
        num_frames,
        prompt="",
        cached_vit_features_root=None,
        truncate_at_question_time=True,
        observe_full_video=False,
        skip_missing_videos=True,
        readiness_cue="",
    ):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths=[])

        self.vis_root = vis_root
        self.num_frames = num_frames
        self.prompt = prompt or ""
        self.cached_vit_features_root = cached_vit_features_root
        self.truncate_at_question_time = bool(truncate_at_question_time)
        self.observe_full_video = bool(observe_full_video)
        self.skip_missing_videos = bool(skip_missing_videos)
        self.readiness_cue = readiness_cue or ""

        self.benchmark = "unknown"
        self.video_meta: Dict[str, Dict] = {}
        self.annotation: List[Dict] = []

        for path in ann_paths:
            with open(path, "r") as f:
                payload = json.load(f)
            self.benchmark = payload.get("benchmark", self.benchmark)
            for video in payload.get("videos", []):
                vid = video["video_id"]
                fps = float(video.get("fps", 1.0))
                frame_length = int(video["frame_length"])
                self.video_meta[vid] = {
                    "fps": fps,
                    "frame_length": frame_length,
                    "duration": float(video.get("duration", frame_length / max(fps, 1e-6))),
                    "source": video.get("source", "unknown"),
                }
                for dlg in video.get("dialogues", []):
                    turns = dlg.get("turns", [])
                    for i, turn in enumerate(turns):
                        rec = dict(turn)
                        rec["video_id"] = vid
                        rec["dialogue_id"] = dlg.get("dialogue_id", vid)
                        rec["turn_index"] = i
                        self.annotation.append(rec)

        self.skipped_videos: List[str] = []
        if self.skip_missing_videos:
            self._drop_unloadable_videos()
        self._add_instance_ids()

    def _drop_unloadable_videos(self):
        """Drop questions whose video has no frames, and report it at startup.

        ``tools/convert_benchmarks.py`` already omits frameless videos when it
        writes an annotation, so this is a backstop for the case where frames
        are removed, moved, or only partially extracted after conversion --
        which would otherwise surface as a ``FileNotFoundError`` from a
        DataLoader worker partway through a run.
        """
        missing = sorted(v for v in self.video_meta if not self.has_clip(v))
        if not missing:
            return

        kind = "cached features" if self.uses_cached_vit_features else "frames"
        gone = set(missing)
        before = len(self.annotation)
        self.annotation = [r for r in self.annotation if r["video_id"] not in gone]
        for v in missing:
            self.video_meta.pop(v, None)
        self.skipped_videos = missing

        if not self.annotation:
            raise RuntimeError(
                f"{self.benchmark}: none of the {len(missing)} video(s) in the "
                f"annotation have {kind} under {self.vis_root!r}. Extract them "
                f"with data/extract_frames.py, or re-run "
                f"tools/convert_benchmarks.py against the frames you do have."
            )

        logging.warning(
            "%s: %d of %d video(s) have no %s, so their %d question(s) are "
            "skipped. EVALUATING A SUBSET: %d video(s), %d question(s). "
            "Missing: %s%s",
            self.benchmark, len(missing), len(missing) + len(self.video_meta),
            kind, before - len(self.annotation),
            len(self.video_meta), len(self.annotation),
            ", ".join(missing[:5]), " ..." if len(missing) > 5 else "",
        )

    def __len__(self):
        return len(self.annotation)

    # --------------------------------------------------------------- framing
    def _observable_frames(self, ann, meta) -> int:
        """How many frames the model is allowed to see for this turn."""
        frame_length = meta["frame_length"]
        if self.observe_full_video:
            return frame_length

        q_ts = float(ann.get("question_timestamp", 0) or 0)
        if self.truncate_at_question_time and q_ts > 0:
            # Official protocol for the streaming benchmarks: the stream is cut at
            # the ask time, so nothing after it is observable.
            horizon = int(round(q_ts * meta["fps"])) + 1
            return max(2, min(frame_length, horizon))

        if ann.get("evidence_end"):
            # ProReady-QA style: evidence lies ahead of the question, so the model
            # must be allowed to watch past it.
            return frame_length

        return frame_length

    def __getitem__(self, index):
        ann = self.annotation[index]
        meta = self.video_meta[ann["video_id"]]
        fps = meta["fps"]

        n_obs = self._observable_frames(ann, meta)
        frame_indices = sample_segment_indices(
            n_obs, self.num_frames, randomize=self.randomize_frames
        )
        video = self.load_clip(ann["video_id"], frame_indices)

        question = self.text_processor(ann["question"])
        options = ann.get("options") or []

        if options:
            text_input = MC_PROMPT.format(
                question=question, options=render_options(options)
            )
            target = ann.get("answer_letter") or ""
        else:
            text_input = (self.prompt + question) if self.prompt else question
            target = str(ann["answer"])
        text_input = append_readiness_cue(text_input, self.readiness_cue)

        return {
            "image": video,
            "text_input": text_input,
            "text_output": target,
            "qa_id": ann.get("qa_id", f"{ann['video_id']}_{index}"),
            "question_id": ann.get("qa_id", f"{ann['video_id']}_{index}"),
            "video_id": ann["video_id"],
            "dialogue_id": ann["dialogue_id"],
            "turn_index": ann["turn_index"],
            "task": ann.get("task", "unknown"),
            "benchmark": self.benchmark,
            "options": options,
            "answer_text": str(ann["answer"]),
            "question_timestamp": float(ann.get("question_timestamp", 0) or 0),
            # Zero when the benchmark has no evidence annotation; the task skips
            # ARS in that case and reports accuracy only.
            "evidence_start": float(ann.get("evidence_start", 0) or 0),
            "evidence_end": float(ann.get("evidence_end", 0) or 0),
            "duration": meta["duration"],
            "frame_times": torch.tensor(
                [float(i) / max(fps, 1e-6) for i in frame_indices], dtype=torch.float32
            ),
            "answers": [str(ann["answer"])],
        }

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        out = {
            "image": torch.stack([s["image"] for s in samples], dim=0),
            "frame_times": torch.stack([s["frame_times"] for s in samples], dim=0),
            "text_input": [s["text_input"] for s in samples],
            "text_output": [s["text_output"] for s in samples],
            "answers": [s["answers"] for s in samples],
        }
        for key in (
            "qa_id", "question_id", "video_id", "dialogue_id", "turn_index", "task",
            "benchmark", "options", "answer_text", "question_timestamp",
            "evidence_start", "evidence_end", "duration",
        ):
            out[key] = [s[key] for s in samples]
        return out


class StreamingQAEvalDataset(StreamingQADataset):
    """Evaluation split: deterministic frame sampling."""

    randomize_frames = False
