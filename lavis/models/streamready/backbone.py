"""
Qwen2-VL backbone -- the frozen host StreamReady plugs into.

The paper builds on a Qwen-2-VL backbone (Sec. 5). This wraps it as two frozen
halves with the StreamReady module in between::

    frames -> [vision tower] -> StreamReadyModule -> [language model] -> answer

Both halves come from one ``Qwen2VLForConditionalGeneration`` checkpoint, so the
vision tower and the decoder are the matched pair Qwen trained together. Nothing
here is trainable; everything is restored from the Qwen2-VL checkpoint rather
than from a StreamReady checkpoint.

Two details of Qwen2-VL's vision tower drive the code below.

**It takes packed patches, not images.** The tower's forward signature is
``(hidden_states, grid_thw)`` where ``hidden_states`` is a flat
``[total_patches, C * temporal_patch_size * patch_size^2]`` tensor. The patch
order is not row-major: ``rot_pos_emb`` assigns positions assuming
``spatial_merge_size`` x ``spatial_merge_size`` block-major order, and
``PatchMerger`` fuses each group of four *adjacent* tokens. Packing row-major
would produce the right tensor shapes while scrambling both the positions and
which patches get merged, so :func:`pack_frames` reproduces the official
``Qwen2VLImageProcessor`` ordering exactly.

"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration


def pack_frames(
    frames: torch.Tensor,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack ``[N, C, H, W]`` frames into Qwen2-VL's flat patch layout.

    Mirrors ``Qwen2VLImageProcessor._preprocess``: reshape to
    ``(n, tps, C, gh/m, m, ps, gw/m, m, ps)`` then transpose to
    ``(n, gh/m, gw/m, m, m, C, tps, ps, ps)``, which orders patches
    block-major so ``rot_pos_emb`` and ``PatchMerger`` agree with the sequence.

    Each frame becomes one temporal patch, repeated ``temporal_patch_size``
    times, so one returned grid entry is one frame.

    Returns:
        ``(flat_patches [N*gh*gw, C*tps*ps*ps], grid_thw [N, 3])``
    """
    n, c, h, w = frames.shape
    ps, tps, m = patch_size, temporal_patch_size, merge_size
    if h % (ps * m) or w % (ps * m):
        raise ValueError(
            f"frame size {h}x{w} must be divisible by patch_size*merge_size="
            f"{ps * m}; Qwen2-VL merges {m}x{m} patch blocks"
        )
    gh, gw = h // ps, w // ps

    x = frames.unsqueeze(1).expand(-1, tps, -1, -1, -1)
    x = x.reshape(n, tps, c, gh // m, m, ps, gw // m, m, ps)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    x = x.reshape(n * gh * gw, c * tps * ps * ps)

    grid = torch.tensor([[1, gh, gw]], dtype=torch.long,
                        device=frames.device).repeat(n, 1)
    return x, grid


class Qwen2VLBackbone(nn.Module):
    """Frozen Qwen2-VL vision tower + language model.

    Args:
        model_path: local directory or hub id of a Qwen2-VL checkpoint.
        torch_dtype: weights dtype, fp16 by default as in the paper's setup.
        attn_implementation: ``sdpa`` works without flash-attn installed.
    """

    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype = torch.float16,
        attn_implementation: str = "sdpa",
    ):
        super().__init__()
        self.model_path = model_path

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        # Keep the two halves as separate attributes: StreamReady sits between
        # them, so they are never called back to back.
        self.visual = model.visual
        self.language_model = model.model
        self.lm_head = model.lm_head
        self.config = model.config
        self._vision_cfg = model.config.vision_config

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for p in self.parameters():
            p.requires_grad = False
        self.visual.eval()
        self.language_model.eval()
        self.lm_head.eval()

        logging.info(
            "Qwen2-VL backbone: vision %d->%d, decoder width %d (both frozen)",
            self._vision_cfg.embed_dim, self.vision_width, self.llm_width,
        )

    def train(self, mode: bool = True):
        """Stay in eval mode: both halves are frozen for all of training."""
        super().train(mode)
        self.visual.eval()
        self.language_model.eval()
        self.lm_head.eval()
        return self

    # ------------------------------------------------------------------ widths
    @property
    def vision_width(self) -> int:
        """Width of the tokens the vision tower emits (post patch-merger)."""
        return int(self._vision_cfg.hidden_size)

    @property
    def llm_width(self) -> int:
        return int(self.config.hidden_size)

    @property
    def tokens_per_frame_at(self) -> Dict[int, int]:
        """Only informational; see :meth:`tokens_per_frame`."""
        return {}

    def tokens_per_frame(self, image_size: int) -> int:
        v = self._vision_cfg
        side = image_size // v.patch_size // v.spatial_merge_size
        return side * side

    # ------------------------------------------------------------------ vision
    @torch.no_grad()
    def encode_frames(self, pixels: torch.Tensor) -> torch.Tensor:
        """``[B, C, T, H, W]`` pixels -> ``[B, T, N_tokens, vision_width]``.

        Run under ``no_grad``: the tower is frozen and StreamReady's gradients
        start at its output.
        """
        if pixels.dim() != 5:
            raise ValueError(f"expected [B, C, T, H, W] pixels, got {pixels.dim()}D")
        b, c, t, h, w = pixels.shape
        frames = pixels.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        v = self._vision_cfg
        flat, grid = pack_frames(
            frames, v.patch_size, v.temporal_patch_size, v.spatial_merge_size
        )
        feats = self.visual(flat.to(self.visual.get_dtype()), grid_thw=grid)
        return feats.view(b, t, -1, feats.shape[-1])

    # ---------------------------------------------------------------- language
    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.embed_tokens(input_ids)

    def lm_forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """Decoder forward on pre-built embeddings.

        Passing ``inputs_embeds`` bypasses Qwen2-VL's own image/video token
        substitution entirely, which is what makes StreamReady's tokens the only
        visual evidence the decoder sees. ``position_ids`` is left to the model,
        which builds the 3D mRoPE positions with all three axes identical -- the
        text-only case, correct here because the sequence carries no native image
        tokens.
        """
        out = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = self.lm_head(out.last_hidden_state).float()
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def lm_generate(self, inputs_embeds, attention_mask, **kwargs):
        """Greedy/beam generation from pre-built embeddings.

        ``Qwen2VLForConditionalGeneration.generate`` expects to manage its own
        visual tokens, so generation is driven from the decoder plus ``lm_head``
        wrapped as a plain causal LM instead.
        """
        from transformers import GenerationConfig

        gen_cfg = GenerationConfig(**kwargs)
        return _generate_from_embeds(
            self.language_model, self.lm_head, inputs_embeds, attention_mask, gen_cfg,
        )


@torch.no_grad()
def _generate_from_embeds(decoder, lm_head, inputs_embeds, attention_mask, cfg):
    """Minimal autoregressive decode over ``inputs_embeds``.

    Beam search is deliberately not implemented: every task this repo scores is
    either multiple choice, a count, Yes/No, or a short free-form answer, all of
    which greedy decoding handles. Sampling is supported because the ablations
    use it.
    """
    device = inputs_embeds.device
    b = inputs_embeds.shape[0]
    max_new = int(getattr(cfg, "max_new_tokens", None) or 32)
    do_sample = bool(getattr(cfg, "do_sample", False))
    temperature = float(getattr(cfg, "temperature", 1.0) or 1.0)
    top_p = float(getattr(cfg, "top_p", 1.0) or 1.0)
    eos_id = getattr(cfg, "eos_token_id", None)
    pad_id = getattr(cfg, "pad_token_id", None)

    embeds, mask = inputs_embeds, attention_mask
    past = None
    done = torch.zeros(b, dtype=torch.bool, device=device)
    tokens = []

    for _ in range(max_new):
        out = decoder(
            inputs_embeds=embeds,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        logits = lm_head(out.last_hidden_state[:, -1]).float()

        if do_sample:
            logits = logits / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            if 0.0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumulative = sorted_probs.cumsum(dim=-1)
                keep = cumulative - sorted_probs < top_p
                keep[:, 0] = True
                sorted_probs = sorted_probs * keep
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                picked = torch.multinomial(sorted_probs, 1)
                nxt = sorted_idx.gather(-1, picked).squeeze(-1)
            else:
                nxt = torch.multinomial(probs, 1).squeeze(-1)
        else:
            nxt = logits.argmax(dim=-1)

        if eos_id is not None:
            eos_ids = eos_id if isinstance(eos_id, (list, tuple)) else [eos_id]
            done = done | torch.isin(nxt, torch.tensor(eos_ids, device=device))
            if pad_id is not None:
                nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
        tokens.append(nxt)
        if bool(done.all()):
            break

        embeds = decoder.embed_tokens(nxt.unsqueeze(1))
        mask = torch.cat(
            [mask, (~done).long().unsqueeze(1)], dim=1
        )

    if not tokens:
        return torch.zeros(b, 0, dtype=torch.long, device=device)
    return torch.stack(tokens, dim=1)
