"""
Contextual Memory Bank -- StreamReady paper Sec. 3.2 ("Contextual Memory Bank")
and Sec. 3.3 ("Contextual Reasoning").

Many streaming questions depend on earlier turns rather than on new pixels. The
bank keeps one entry per answered question: the question embedding ``q_i`` and the
representation ``a_i`` that was actually used to produce that answer. When a new
question arrives, entries whose question embedding is similar to the current one
are selected by a soft gate and their answer representations are fused into the
long-term visual feature ``z_l`` through a light cross-attention layer.

"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualMemoryBank(nn.Module):
    """Per-stream bank of ``(question embedding, answer representation)`` turns.

    Entries are stored detached: contextual memory is history, and gradients for
    the current turn should not flow back into previous turns' graphs.
    """

    def __init__(
        self,
        width: int,
        capacity: int = 32,
        num_heads: int = 12,
        topk: int = 4,
        sim_floor: float = 0.2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.width = int(width)
        self.capacity = int(capacity)
        self.topk = int(topk)
        self.sim_floor = float(sim_floor)

        self.norm_q = nn.LayerNorm(width)
        self.norm_kv = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            width, num_heads, dropout=dropout, batch_first=True
        )
        # Zero-init output projection so the whole module is the identity at
        # construction. The Q-Former arrives with pretrained BLIP-2 weights and we
        # do not want an untrained fusion path perturbing it on step 0.
        nn.init.zeros_(self.attn.out_proj.weight)
        if self.attn.out_proj.bias is not None:
            nn.init.zeros_(self.attn.out_proj.bias)

        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        """Drop all turns. Call between videos, not between turns."""
        self.q_mem: Optional[torch.Tensor] = None      # [B, M, C]
        self.a_mem: Optional[torch.Tensor] = None      # [B, M, C]
        self.mem_mask: Optional[torch.Tensor] = None   # [B, M] bool

    def _alloc(self, b: int, device, dtype):
        self.q_mem = torch.zeros(b, self.capacity, self.width, device=device, dtype=dtype)
        self.a_mem = torch.zeros(b, self.capacity, self.width, device=device, dtype=dtype)
        self.mem_mask = torch.zeros(b, self.capacity, dtype=torch.bool, device=device)

    @torch.no_grad()
    def write(self, q_embed: torch.Tensor, a_repr: torch.Tensor):
        """Record one answered turn.

        Args:
            q_embed: ``[B, C]`` pooled question embedding.
            a_repr:  ``[B, C]`` pooled representation that produced the answer.

        At capacity the oldest turn is dropped (FIFO), keeping the bank bounded
        the same way the visual tree is.
        """
        b, c = q_embed.shape
        if self.q_mem is None:
            self._alloc(b, q_embed.device, q_embed.dtype)

        q_embed = q_embed.detach().to(self.q_mem.dtype)
        a_repr = a_repr.detach().to(self.a_mem.dtype)

        full = self.mem_mask.all(dim=1)
        if bool(full.any()):
            self.q_mem[full] = torch.roll(self.q_mem[full], -1, dims=1)
            self.a_mem[full] = torch.roll(self.a_mem[full], -1, dims=1)
            self.q_mem[full, -1] = q_embed[full]
            self.a_mem[full, -1] = a_repr[full]
        rest = ~full
        if bool(rest.any()):
            idx = self.mem_mask[rest].long().sum(dim=1)
            self.q_mem[rest, idx] = q_embed[rest]
            self.a_mem[rest, idx] = a_repr[rest]
            m = self.mem_mask[rest]
            m[torch.arange(m.shape[0], device=m.device), idx] = True
            self.mem_mask[rest] = m

    # ------------------------------------------------------------------ fusion
    def forward(self, z_l: torch.Tensor, q_embed: torch.Tensor) -> torch.Tensor:
        """Fuse relevant past answers into ``z_l``.

        Args:
            z_l:     ``[B, T, C]`` long-term reasoning representation.
            q_embed: ``[B, C]`` pooled embedding of the current question.

        Returns:
            ``[B, T, C]``, unchanged when the bank is empty (first turn of a video).
        """
        if self.q_mem is None or not bool(self.mem_mask.any()):
            return z_l

        # --- relevance gate over stored questions ---
        # Run the gate in fp32 with autocast disabled. Under autocast an einsum is
        # promoted back to fp16 regardless of a .float() on its inputs, and then a
        # fp32 masking sentinel overflows when filled into a half tensor.
        with torch.cuda.amp.autocast(enabled=False):
            qn = F.normalize(q_embed.float(), dim=-1)                   # [B, C]
            kn = F.normalize(self.q_mem.float(), dim=-1)                # [B, M, C]
            sim = torch.einsum("bc,bmc->bm", qn, kn).float()            # [B, M]

        neg = torch.finfo(sim.dtype).min
        sim = sim.masked_fill(~self.mem_mask, neg)

        # Hard relevance filter, then keep at most topk survivors.
        keep = self.mem_mask & (sim >= self.sim_floor)
        k = min(self.topk, sim.shape[1])
        top = sim.topk(k, dim=1)
        sel = torch.zeros_like(keep).scatter_(1, top.indices, True) & keep  # [B, M]

        if not bool(sel.any()):
            # Nothing in history is relevant to this question -- leave z_l alone
            # rather than attending to noise.
            return z_l

        # Soft weights across the surviving entries only.
        w = torch.softmax(sim.masked_fill(~sel, neg), dim=1)
        w = torch.where(sel, w, torch.zeros_like(w))                    # [B, M]

        kv = self.a_mem * w[:, :, None].to(self.a_mem.dtype)            # [B, M, C]

        out = self.attn(
            self.norm_q(z_l),
            self.norm_kv(kv),
            self.norm_kv(kv),
            key_padding_mask=~sel,
            need_weights=False,
        )[0]
        return z_l + out
