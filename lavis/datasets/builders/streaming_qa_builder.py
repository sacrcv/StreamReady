"""
Builders for the unified video-QA annotation schema.

Two things are built from the same schema and the same dataset class:

* **OVO-Bench** (``ovo_bench``) -- the online video understanding benchmark
  StreamReady is evaluated on besides ProReady-QA. Its backward and real-time
  tasks are multiple choice; its forward REC/SSR/CRR tasks are open-ended. Every
  question carries its own ask time, and the official protocol cuts the video
  there, so ``truncate_at_question_time`` is on: a model that saw later frames
  would be reading the future. Annotations are produced by
  ``tools/convert_benchmarks.py``.

* **The training corpus** (``video_qa_train``) -- the 62k video-QA pairs of
  paper supplementary Table 11. These are plain video-QA with no ask time, so
  truncation is off and the whole clip is observable. The corpora themselves are
  not redistributed here; see the README for sources and the schema to convert
  them to.

Frames for both come from ``data/extract_frames.py``.
"""

import os
import warnings

import lavis.common.utils as utils
from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.streaming_qa_datasets import (
    StreamingQADataset,
    StreamingQAEvalDataset,
)


class StreamingQABuilder(BaseDatasetBuilder):
    """Shared build logic; subclasses only supply a config path."""

    train_dataset_cls = StreamingQADataset
    eval_dataset_cls = StreamingQAEvalDataset

    # Whether the stream is cut at the question's ask time. On for online
    # benchmarks, off for plain video-QA training data.
    truncate_at_question_time = True

    def _download_data(self):
        """No-op: annotations are produced locally by convert_benchmarks.py and
        frames by data/extract_frames.py. The inherited version requires a `url`
        per annotation entry."""
        return

    def build(self):
        self.build_processors()

        build_info = self.config.build_info
        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue
            is_train = split == "train"

            vis_processor = (
                self.vis_processors["train"] if is_train else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"] if is_train else self.text_processors["eval"]
            )
            prompt = self.config.get("text_processor").get(
                "train" if is_train else "eval"
            ).get("prompt", "")

            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]
            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = utils.get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)

            missing = [p for p in abs_ann_paths if not os.path.exists(p)]
            if missing:
                warnings.warn(
                    f"{self.__class__.__name__}: annotation file(s) not found: "
                    f"{missing}. Build them with tools/convert_benchmarks.py."
                )
                continue

            vis_path = vis_info.storage
            if not os.path.isabs(vis_path):
                vis_path = utils.get_cache_path(vis_path)
            if not os.path.exists(vis_path):
                warnings.warn(f"storage path {vis_path} does not exist.")

            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=abs_ann_paths,
                vis_root=vis_path,
                num_frames=self.config.num_frames,
                prompt=prompt,
                cached_vit_features_root=self._resolve_cached_vit_features_root(),
                truncate_at_question_time=self.config.get(
                    "truncate_at_question_time", self.truncate_at_question_time
                ),
                skip_missing_videos=self.config.get("skip_missing_videos", True),
                readiness_cue=self.config.get("readiness_cue", ""),
                observe_full_video=self.config.get("observe_full_video", False),
            )

        return datasets


@registry.register_builder("ovo_bench")
class OVOBenchBuilder(StreamingQABuilder):
    """OVO-Bench: online video understanding, one ask time per question."""

    DATASET_CONFIG_DICT = {"default": "configs/datasets/ovo_bench/defaults.yaml"}


@registry.register_builder("video_qa_train")
class VideoQATrainBuilder(StreamingQABuilder):
    """The Table 11 training corpus, converted to the unified schema.

    No ask time, so nothing is truncated and the whole clip is observable. This
    is the dataset ``train.py`` consumes; both ProReady-QA and OVO-Bench are
    evaluation-only.
    """

    DATASET_CONFIG_DICT = {"default": "configs/datasets/video_qa_train/defaults.yaml"}
    truncate_at_question_time = False
