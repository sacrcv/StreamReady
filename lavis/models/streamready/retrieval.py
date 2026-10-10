"""
Query-aware coarse-to-fine retrieval -- StreamReady paper Eqs. 4-5, Sec. 3.3.

When a question arrives the model does not reason over all of long-term memory. It
first scores the prototypes in M_V3 and keeps the Top-K most promising high-level
regions (Eq. 4), then gathers only the centroids hanging off those prototypes into
a candidate pool and keeps the Top-m of those (Eq. 5)::

    M_V3_t = Top-K( (W_s q)^T M_V3 )
    C_0    = union over s in M_V3_t of M_V2^(s)
    M_V2_t = Top-m over c in C_0 of ( (W_c q)^T c )

The paper is explicit about an asymmetry between the two stages, and it is load
bearing, so it is implemented literally:

* Prototype scores **are** softmax-normalised, "to stabilize routing and sharpen
  focus on plausible evidence".
* Centroid scores are **not** normalised, because "sharp ranking is essential for
  isolating specific evidence rather than broad regions".

So the prototype path returns a soft routing weight that scales its slots, while
the centroid path is a hard Top-m with raw scores.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class QueryProjection(nn.Module):
    """The two query projections of Eqs. 4-5, ``W_s`` and ``W_c``.

    Separate matrices on purpose: the prototype stage is a coarse semantic map
    lookup while the centroid stage isolates specific evidence, and the paper
    scores them differently.
    """

    def __init__(self, query_width: int, memory_width: int):
        super().__init__()
        self.w_proto = nn.Linear(query_width, memory_width, bias=False)
        self.w_cent = nn.Linear(query_width, memory_width, bias=False)

    def forward(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.w_proto(q), self.w_cent(q)


def _score(query: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Dot-product scores between ``[B, C]`` queries and ``[B, K, P, C]`` slots.

    Slots are pooled over their token axis first so a slot gets one score
    regardless of how many tokens it holds. Forced to fp32 with autocast off:
    scores are masked with a dtype sentinel and then softmaxed, and an fp16
    sentinel would both overflow and lose ranking precision.
    """
    with torch.cuda.amp.autocast(enabled=False):
        pooled = slots.mean(dim=2).float()                       # [B, K, C]
        return torch.einsum("bc,bkc->bk", query.float(), pooled).float()


def coarse_to_fine_retrieve(
    q_proto: torch.Tensor,
    q_cent: torch.Tensor,
    prototypes: torch.Tensor,
    proto_mask: torch.Tensor,
    centroids: torch.Tensor,
    centroid_mask: torch.Tensor,
    cent_proto: torch.Tensor,
    topk_prototypes: int = 8,
    topk_centroids: int = 24,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Retrieve evidence slots for one question.

    Args:
        q_proto, q_cent: projected question vectors, ``[B, C]`` each.
        prototypes:   ``[B, U, P, C]``   proto_mask:    ``[B, U]``
        centroids:    ``[B, J, P, C]``   centroid_mask: ``[B, J]``
        cent_proto:   ``[B, J]`` prototype index per centroid, ``-1`` if unassigned.
        topk_prototypes: K in Eq. 4 (paper default 8).
        topk_centroids:  m in Eq. 5 (paper default 24).

    Returns:
        ``(slots, mask)`` where ``slots`` is ``[B, (K+m)*P, C]`` ready to be
        concatenated as cross-attention memory, and ``mask`` is ``[B, (K+m)*P]``
        marking which of those token positions are real. Padded positions are
        zeroed so they are inert even if a caller ignores the mask.
    """
    b, u, p, c = prototypes.shape
    j = centroids.shape[1]
    device = prototypes.device
    K = min(topk_prototypes, u)
    m = min(topk_centroids, j)

    neg = torch.finfo(torch.float32).min

    # ---- Eq. 4: score prototypes, keep Top-K, softmax-normalise the scores ----
    ps = _score(q_proto, prototypes).float()
    ps = ps.masked_fill(~proto_mask, neg)
    ptop = ps.topk(K, dim=1)
    pidx, pval = ptop.indices, ptop.values                        # [B, K]
    psel_mask = pval > neg / 2                                    # real prototypes

    # Normalise across the selected set only; dead slots get zero weight.
    pw = torch.softmax(pval.masked_fill(~psel_mask, neg), dim=1)
    pw = torch.where(psel_mask, pw, torch.zeros_like(pw))         # [B, K]

    proto_slots = torch.gather(
        prototypes, 1, pidx[:, :, None, None].expand(b, K, p, c)
    )                                                             # [B, K, P, C]
    # Routing weight scales its prototype's tokens (the "semantic map lookup").
    proto_slots = proto_slots * pw[:, :, None, None].to(proto_slots.dtype)

    # ---- Eq. 5: candidate pool C_0, then an UN-normalised hard Top-m ----
    # A centroid is a candidate iff it is live and its prototype was selected.
    belongs = (cent_proto[:, :, None] == pidx[:, None, :]) & psel_mask[:, None, :]
    in_pool = belongs.any(dim=2) & centroid_mask                  # [B, J]

    cs = _score(q_cent, centroids).float()
    cs = cs.masked_fill(~in_pool, neg)

    # Fall back to all live centroids if the pool came out empty (possible very
    # early in a stream, before any prototype exists).
    empty = ~in_pool.any(dim=1)
    if bool(empty.any()):
        alt = _score(q_cent, centroids).float().masked_fill(~centroid_mask, neg)
        cs = torch.where(empty[:, None], alt, cs)

    ctop = cs.topk(m, dim=1)
    cidx, cval = ctop.indices, ctop.values
    csel_mask = cval > neg / 2

    cent_slots = torch.gather(
        centroids, 1, cidx[:, :, None, None].expand(b, m, p, c)
    )                                                             # [B, m, P, C]

    # ---- assemble ----
    slots = torch.cat([proto_slots, cent_slots], dim=1)           # [B, K+m, P, C]
    slot_ok = torch.cat([psel_mask, csel_mask], dim=1)            # [B, K+m]
    slots = slots * slot_ok[:, :, None, None].to(slots.dtype)

    slots = slots.reshape(b, (K + m) * p, c)
    mask = slot_ok[:, :, None].expand(b, K + m, p).reshape(b, (K + m) * p)
    return slots, mask
