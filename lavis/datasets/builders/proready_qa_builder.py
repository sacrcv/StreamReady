"""
Builder for ProReady-QA (StreamReady paper Sec. 4).

Follows the repo's three-layer idiom: dataset YAML -> registered builder ->
dataset class. Unlike the plain video-QA builders there is no ``ans2label``:
ProReady-QA answers are free-form text scored by the readiness-aware VQA task,
not picked from a closed label set.
"""

import os
import warnings

import lavis.common.utils as utils
from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.proready_qa_datasets import ProReadyQAEvalDataset


@registry.register_builder("proready_qa")
class ProReadyQABuilder(BaseDatasetBuilder):
    """ProReady-QA is evaluation-only: there is no train split to build.

    The benchmark exists to measure *when* a model answers, so training on it
    would defeat its purpose. StreamReady is trained on the separate corpus of
    supplementary Table 11 (the ``video_qa_train`` builder) and evaluated here.
    """

    train_dataset_cls = None
    eval_dataset_cls = ProReadyQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/proready_qa/defaults.yaml",
    }

    def _download_data(self):
        """No-op: nothing about ProReady-QA is fetched at build time.

        ProReady-QA annotations are downloaded from the Hugging Face Hub (see the
        README) and frames are extracted locally with ``data/extract_frames.py``.
        The inherited implementation expects a ``url`` per annotation entry and
        fails without one.
        """
        return

    def build(self):
        self.build_processors()

        build_info = self.config.build_info
        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)

        datasets = dict()
        for split in ann_info.keys():
            if split == "train":
                raise ValueError(
                    "ProReady-QA is an evaluation-only benchmark, but a 'train' "
                    "split was configured. Train on `video_qa_train` instead."
                )
            if split not in ["val", "test"]:
                continue

            vis_processor = self.vis_processors["eval"]
            text_processor = self.text_processors["eval"]
            prompt = self.config.get("text_processor").get("eval").get("prompt", "")

            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]
            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = utils.get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)

            vis_path = vis_info.storage
            if not os.path.isabs(vis_path):
                vis_path = utils.get_cache_path(vis_path)
            if not os.path.exists(vis_path):
                warnings.warn("storage path {} does not exist.".format(vis_path))

            datasets[split] = self.eval_dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=abs_ann_paths,
                vis_root=vis_path,
                num_frames=self.config.num_frames,
                prompt=prompt,
                cached_vit_features_root=self._resolve_cached_vit_features_root(),
                skip_missing_videos=self.config.get("skip_missing_videos", True),
                readiness_cue=self.config.get("readiness_cue", ""),
                observe_full_video=self.config.get("observe_full_video", True),
            )

        return datasets
