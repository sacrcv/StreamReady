"""Pre-extract the frozen Qwen2-VL vision tower's features for a dataset.

We cache one ``[N_frames, N_tokens, hidden]`` fp16 tensor per video to disk.
At training time the dataset reads the cached tensor and indexes the
selected frame indices, skipping the entire ViT forward.

Caveat: this extraction uses the *eval* visual transform (resize +
center-style behavior) without random crop / flip. That removes
training-time pixel augmentation. A frozen tower hardly benefits from
that augmentation anyway; we keep the per-epoch frame-index sampling
randomness, which is the dominant source of clip-level variation.

Usage (single GPU)::

    python tools/extract_vit_features.py \
        --frames-root data/proready_qa/frames \
        --output-root data/proready_qa/vit_features

Run multiple shards in parallel (e.g. 4 GPUs, one shard each)::

    CUDA_VISIBLE_DEVICES=0 python tools/extract_vit_features.py \
        --frames-root data/proready_qa/frames \
        --output-root data/proready_qa/vit_features \
        --num-shards 4 --shard-id 0 &
    CUDA_VISIBLE_DEVICES=1 python tools/extract_vit_features.py \
        --frames-root data/proready_qa/frames \
        --output-root data/proready_qa/vit_features \
        --num-shards 4 --shard-id 1 &
    ...

The script is idempotent: existing ``<vid>.pt`` files are skipped.

Performance note: frame decoding (PIL JPEG -> tensor) and the video
transform are CPU work. Each shard runs a ``DataLoader`` with
``--num-workers`` worker processes that pre-decode upcoming videos so the
GPU stays busy with ViT forwards. Increase ``--num-workers`` if you have
spare CPU cores.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

# Make the repo importable when run from anywhere.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lavis.models.streamready.backbone import Qwen2VLBackbone  # noqa: E402
from lavis.processors.blip_processors import Blip2VideoEvalProcessor  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--frames-root", type=str, required=True,
                   help="Directory with one subdirectory per video, each containing frame*.jpg.")
    p.add_argument("--output-root", type=str, required=True,
                   help="Where to save <vid>.pt cached feature tensors.")
    p.add_argument("--backbone-model", type=str, default="llm/Qwen2-VL-7B-Instruct",
                   help="Qwen2-VL checkpoint whose vision tower to cache. Must "
                        "match model.backbone_model, or the cache will not "
                        "describe the encoder training runs with.")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--backbone-precision", type=str, default="fp16",
                   choices=["fp16", "fp32"])
    p.add_argument("--batch-size", type=int, default=64,
                   help="Number of frames per ViT forward.")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save-dtype", type=str, default="fp16",
                   choices=["fp16", "fp32"])
    p.add_argument("--frame-glob", type=str, default="frame*.jpg",
                   help="Glob inside each video directory matching frames.")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers used for parallel frame decode/transform.")
    p.add_argument("--prefetch-factor", type=int, default=2,
                   help="DataLoader prefetch_factor (videos per worker queued ahead).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset & collate
# ---------------------------------------------------------------------------

def list_video_dirs(frames_root: Path, num_shards: int, shard_id: int) -> List[Path]:
    all_dirs = sorted(d for d in frames_root.iterdir() if d.is_dir())
    return [d for i, d in enumerate(all_dirs) if i % num_shards == shard_id]


class _VideoFrameDataset(Dataset):
    """One item = one whole video, already decoded + transformed.

    Returned dict has keys:
      - ``video_id`` (str): basename of the video directory.
      - ``frames`` (torch.Tensor): ``[N_frames, C, H, W]`` float32, normalised.
      - ``empty`` (bool): True when the video had zero frames or decoding failed.
    """

    def __init__(self, video_dirs: List[Path], frame_glob: str, processor):
        self.video_dirs = video_dirs
        self.frame_glob = frame_glob
        self.processor = processor

    def __len__(self) -> int:
        return len(self.video_dirs)

    def __getitem__(self, idx: int):
        vdir = self.video_dirs[idx]
        video_id = vdir.name
        frame_paths = sorted(vdir.glob(self.frame_glob))
        if not frame_paths:
            return {"video_id": video_id, "frames": None, "empty": True, "error": "no frames"}

        try:
            decoded = [pil_to_tensor(Image.open(p).convert("RGB")).to(torch.float32)
                       for p in frame_paths]
            video = torch.stack(decoded, dim=1)        # [C, N, H, W]
            video = self.processor(video)              # [C, N, H', W'] normalised
            # [C, N, H, W] -> [N, C, H, W] for the per-frame ViT.
            frames_tensor = video.permute(1, 0, 2, 3).contiguous()
        except Exception as exc:  # pragma: no cover - best-effort robustness
            return {"video_id": video_id, "frames": None, "empty": True, "error": str(exc)}

        return {"video_id": video_id, "frames": frames_tensor, "empty": False, "error": None}


def _collate(batch):
    """We use ``batch_size=1`` so 'collate' just unwraps the singleton list."""
    return batch[0]


# ---------------------------------------------------------------------------
# ViT forward
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_frames(backbone, frames_tensor, device, batch_size: int) -> torch.Tensor:
    """``[N, C, H, W]`` frames -> ``[N, N_tokens, vision_width]``.

    Chunked because the whole clip at once would not fit; the tower treats each
    frame as an independent attention window, so chunking changes nothing.
    """
    frames_tensor = frames_tensor.to(device, non_blocking=True)
    out_chunks = []
    for start in range(0, frames_tensor.size(0), batch_size):
        chunk = frames_tensor[start:start + batch_size]
        # encode_frames takes [B, C, T, H, W]; one clip of T frames as B=1.
        feats = backbone.encode_frames(chunk.permute(1, 0, 2, 3).unsqueeze(0))
        out_chunks.append(feats[0].float())
    return torch.cat(out_chunks, dim=0)  # [N_frames, N_tokens, C]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CACHE_META = "_cache_meta.json"


def write_cache_meta(output_root: Path, args, feature_dim: int) -> None:
    """Record which encoder produced this cache.

    Without it a cache is an anonymous pile of tensors. A width mismatch would
    at least fail loudly in the Q-Former's cross-attention, but an
    ``image_size`` mismatch would not: it only changes the token count, which
    cross-attention accepts, so training would silently run on features from a
    different resolution than the config asks for.

    Shards race to write this; the contents are identical, so last writer wins.
    """
    meta = {
        "backbone_model": args.backbone_model,
        "image_size": args.image_size,
        "backbone_precision": args.backbone_precision,
        "save_dtype": args.save_dtype,
        "feature_dim": int(feature_dim),
    }
    existing_path = output_root / CACHE_META
    if existing_path.is_file():
        try:
            with open(existing_path) as f:
                existing = json.load(f)
        except (OSError, ValueError):
            existing = None
        if existing and existing != meta:
            differing = {k: (existing.get(k), meta[k])
                         for k in meta if existing.get(k) != meta[k]}
            sys.exit(
                f"ERROR: {output_root} already holds a cache built with "
                f"different settings: {differing} (existing, requested). "
                "Mixing them would give training inconsistent features. "
                "Extract to a fresh --output-root, or delete the existing one."
            )
    with open(existing_path, "w") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    args = parse_args()
    frames_root = Path(args.frames_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    save_dtype = torch.float16 if args.save_dtype == "fp16" else torch.float32

    print(f"Loading the Qwen2-VL vision tower from {args.backbone_model} "
          f"(image_size={args.image_size}, precision={args.backbone_precision}) "
          f"on {device} ...")
    backbone = Qwen2VLBackbone(
        args.backbone_model,
        torch_dtype=torch.float16 if args.backbone_precision == "fp16"
        else torch.float32,
    )
    # Only the vision half is needed; dropping the decoder frees ~15 GB.
    del backbone.language_model
    del backbone.lm_head
    backbone = backbone.to(device).eval()

    write_cache_meta(output_root, args, backbone.vision_width)

    processor = Blip2VideoEvalProcessor(image_size=args.image_size)

    all_video_dirs = list_video_dirs(frames_root, args.num_shards, args.shard_id)
    # Skip videos whose features are already on disk so the DataLoader doesn't
    # spin up workers for them.
    pending_video_dirs = [
        vdir for vdir in all_video_dirs
        if not (output_root / f"{vdir.name}.pt").exists()
    ]
    n_skipped_existing = len(all_video_dirs) - len(pending_video_dirs)
    print(f"[shard {args.shard_id}/{args.num_shards}] "
          f"{len(all_video_dirs)} videos in shard; "
          f"{n_skipped_existing} already cached; "
          f"{len(pending_video_dirs)} to process.")

    if not pending_video_dirs:
        print(f"[shard {args.shard_id}/{args.num_shards}] nothing to do.")
        return

    dataset = _VideoFrameDataset(pending_video_dirs, args.frame_glob, processor)
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(**loader_kwargs)

    t_start = time.time()
    n_done = 0
    n_failed = 0
    for i, item in enumerate(loader):
        video_id = item["video_id"]
        out_path = output_root / f"{video_id}.pt"

        if item["empty"]:
            print(f"  [warn] {video_id}: {item['error']}", flush=True)
            n_failed += 1
            continue

        try:
            feats = _encode_frames(
                backbone, item["frames"], device, args.batch_size,
            )
        except Exception as exc:  # pragma: no cover - best-effort robustness
            print(f"  [err] {video_id}: {exc}", flush=True)
            n_failed += 1
            continue

        torch.save(feats.to(save_dtype).cpu(), out_path)
        n_done += 1

        if (n_done + n_failed) % 25 == 0 or (i + 1) == len(pending_video_dirs):
            elapsed = time.time() - t_start
            print(f"  [{n_done + n_failed}/{len(pending_video_dirs)}] "
                  f"{n_done} done, {n_failed} failed  "
                  f"elapsed={elapsed:6.1f}s "
                  f"avg={elapsed / max(1, n_done):5.2f}s/video", flush=True)

    print(f"[shard {args.shard_id}/{args.num_shards}] finished. "
          f"done={n_done}  failed={n_failed}  pre-existing={n_skipped_existing}  "
          f"total_in_shard={len(all_video_dirs)}")


if __name__ == "__main__":
    main()
