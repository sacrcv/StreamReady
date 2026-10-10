"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

Shared frame loading for StreamReady video datasets.

Every StreamReady/MA-LMM video dataset reads pre-extracted JPEG frames from
``<vis_root>/<video_id>/frame%06d.jpg`` (1-indexed, extracted at 10 fps by
``data/extract_frames.py``). This module factors that out and adds the
option of reading precomputed ViT features instead of pixels.

The ``use_cached_vit_features`` path exists because StreamReady feeds up to 100
frames per clip through a frozen EVA-CLIP ViT-G/14. Since the ViT never
receives gradients, its output can be computed once offline (see
``tools/extract_vit_features.py``) and reused across epochs, which is a
large speedup. Caching changes the shape of the ``image`` field a dataset
returns, and the model branches on that:

  pixels : ``[C, T, H, W]`` -> batched to ``[B, C, T, H, W]`` (``dim() == 5``)
  cached : ``[T, N_tokens, C]`` -> batched to ``[B, T, N, C]`` (``dim() == 4``)
"""

import json
import logging
import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


def sample_segment_indices(frame_length, num_frames, randomize):
    """Pick ``num_frames`` frame indices spanning ``[0, frame_length)``.

    This is MA-LMM's sampling scheme, kept identical so results stay
    comparable:

    - ``randomize=True`` (train): split the clip into ``num_frames`` equal
      segments and draw one random index from each. Acts as temporal jitter.
    - ``randomize=False`` (eval): deterministic equally-spaced indices.
    """
    if randomize:
        segment_list = np.linspace(0, frame_length, num_frames + 1, dtype=int)
        selected = []
        for start, end in zip(segment_list[:-1], segment_list[1:]):
            if start == end:
                selected.append(int(start))
            else:
                selected.append(int(np.random.randint(start, end)))
        return selected
    return (
        np.rint(np.linspace(0, max(frame_length - 1, 0), num_frames))
        .astype(int)
        .tolist()
    )


def sample_window_indices(start_frame, end_frame, num_frames, randomize):
    """Like :func:`sample_segment_indices` but over an explicit frame window.

    For annotations that label a sub-clip rather than a whole video. ProReady-QA
    hands the model the full stream and lets readiness decide when to answer, so
    it uses :func:`sample_segment_indices`; this is here for evidence-window-local
    sampling, which ``observe_full_video: False`` and the evidence probes rely on.
    """
    start_frame = int(max(start_frame, 0))
    end_frame = int(max(end_frame, start_frame))
    if randomize:
        bounds = np.linspace(start_frame, end_frame + 1, num_frames + 1, dtype=int)
        selected = []
        for start, end in zip(bounds[:-1], bounds[1:]):
            if start == end:
                selected.append(int(start))
            else:
                selected.append(int(np.random.randint(start, end)))
        return selected
    return (
        np.rint(np.linspace(start_frame, end_frame, num_frames)).astype(int).tolist()
    )


class FrameLoaderMixin:
    """Loads a clip either as pixels or as cached ViT features.

    Datasets mixing this in must set:
      ``self.vis_root``                 frame root directory
      ``self.vis_processor``            video transform (pixel path only)
      ``self.cached_vit_features_root`` ``None``, or the feature cache root

    and then call :meth:`load_clip` with a video id and frame indices.
    """

    # Datasets that forget to pass the kwarg still behave (pixel path).
    cached_vit_features_root = None

    _cache_provenance_logged = False

    @property
    def uses_cached_vit_features(self):
        return self.cached_vit_features_root is not None

    def _log_cache_provenance(self):
        """Report once which encoder built the cache being read.

        tools/extract_vit_features.py writes ``_cache_meta.json`` alongside the
        tensors. A feature-width mismatch would fail loudly in the Q-Former's
        cross-attention, but an ``image_size`` mismatch would not -- it only
        changes the token count, which cross-attention accepts -- so surfacing
        the provenance is the difference between a diagnosable run and a
        silently wrong one.
        """
        if self._cache_provenance_logged:
            return
        type(self)._cache_provenance_logged = True
        meta_path = os.path.join(self.cached_vit_features_root, "_cache_meta.json")
        if not os.path.isfile(meta_path):
            logging.warning(
                "Feature cache at %s has no _cache_meta.json, so the encoder "
                "and resolution that produced it are unknown. Verify they match "
                "model.backbone_model and image_size, or re-extract with "
                "tools/extract_vit_features.py.",
                self.cached_vit_features_root,
            )
            return
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            return
        logging.info(
            "Vision feature cache: %s at %spx (%s, dim %s)",
            meta.get("backbone_model"), meta.get("image_size"),
            meta.get("save_dtype"), meta.get("feature_dim"),
        )

    def has_clip(self, video_id) -> bool:
        """Is this video loadable at all? Cheap enough to call at startup.

        Deliberately does *not* count frames: a 3600-frame directory per video
        makes that slow, and the question here is only presence. Checks the
        cache file when reading cached features, otherwise stops at the first
        image file it sees.
        """
        if self.uses_cached_vit_features:
            return os.path.isfile(
                os.path.join(self.cached_vit_features_root, f"{video_id}.pt")
            )
        d = os.path.join(self.vis_root, str(video_id))
        if not os.path.isdir(d):
            return False
        try:
            with os.scandir(d) as it:
                for entry in it:
                    if entry.name.lower().endswith((".jpg", ".jpeg", ".png")):
                        return True
        except OSError:
            return False
        return False

    def load_clip(self, video_id, frame_indices):
        """Return one clip for ``video_id`` at ``frame_indices``.

        Shape is ``[T, N_tokens, C]`` when reading cached features and
        ``[C, T, H, W]`` when reading pixels.
        """
        if self.uses_cached_vit_features:
            return self._load_cached_features(video_id, frame_indices)
        return self._load_pixels(video_id, frame_indices)

    def _load_cached_features(self, video_id, frame_indices):
        path = os.path.join(self.cached_vit_features_root, f"{video_id}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Cached ViT features not found for video {video_id!r} at {path}. "
                "Generate them with tools/extract_vit_features.py, or unset "
                "`cached_vit_features_root` / `model.use_cached_vit_features` "
                "to run the ViT online."
            )
        self._log_cache_provenance()
        # ``[N_video_frames, N_tokens, C]`` on CPU, usually fp16.
        cache = torch.load(path, map_location="cpu")
        # Frame counts can differ by one or two from the annotation's
        # `frame_length` (extraction rounding), so clamp rather than fail.
        n_cached = cache.size(0)
        clamped = [min(max(int(i), 0), n_cached - 1) for i in frame_indices]
        return cache[clamped]  # [T, N_tokens, C]

    # Frame file naming. ``data/extract_frames.py`` writes ``frame%06d.jpg``, but
    # frame dumps from other sources (notably the VStream-QA release that
    # ProReady-QA is built on) are often bare ``%06d.jpg``. Both are accepted so
    # an existing extraction can be pointed at without renaming it. The pattern is
    # detected once per video directory and then reused.
    _FRAME_PATTERNS = ("frame{:06d}.jpg", "{:06d}.jpg", "frame{:06d}.png", "{:06d}.png")

    def _frame_pattern(self, video_dir):
        cache = getattr(self, "_frame_pattern_cache", None)
        if cache is None:
            cache = self._frame_pattern_cache = {}
        if video_dir in cache:
            return cache[video_dir]

        for pattern in self._FRAME_PATTERNS:
            # Frames are 1-indexed on disk.
            if os.path.isfile(os.path.join(video_dir, pattern.format(1))):
                cache[video_dir] = pattern
                return pattern

        raise FileNotFoundError(
            f"No frames found in {video_dir!r}. Expected 1-indexed files named one "
            f"of {list(self._FRAME_PATTERNS)}. Extract them with "
            f"data/extract_frames.py, or point `vis_root` at an existing dump."
        )

    def _load_pixels(self, video_id, frame_indices):
        video_dir = os.path.join(self.vis_root, video_id)
        pattern = self._frame_pattern(video_dir)
        frame_list = []
        for frame_index in frame_indices:
            frame_path = os.path.join(video_dir, pattern.format(int(frame_index) + 1))
            frame = Image.open(frame_path).convert("RGB")
            frame_list.append(pil_to_tensor(frame).to(torch.float32))
        video = torch.stack(frame_list, dim=1)  # [C, T, H, W]
        return self.vis_processor(video)
