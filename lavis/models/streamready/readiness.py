"""
Readiness mechanism -- StreamReady paper Sec. 3.4.

The reasoning modules decide *what* to answer; this decides *when*. A learnable
``<RDY>`` token is appended to the long-term reasoning representation ``z_l``
inside the long-term branch, so it is exposed to exactly the evidence that drives
answer generation. A lightweight head reads that token and emits a readiness score
``R_pred`` in [0, 1] at every timestep. At inference the LLM is triggered only once
``R_pred`` crosses a threshold (paper default 0.35); until then the model keeps
observing.


Supervision (Eqs. 7-8) needs no ground-truth evidence timestamps. Training has
full-video access, so pseudo-labels come from the similarity between ``z_l`` and
centroid-level memory M_V2 over time: high-similarity timesteps form a
pseudo-positive region P, low-similarity ones a pseudo-negative region N, and a
pairwise contrastive loss pushes readiness higher on P than on N. M_V2 is the
right level because it keeps fine-grained cues without M_V1's redundancy or
M_V3's heavy abstraction.

Critically, ``L_rdy`` updates **only** the readiness head and the ``<RDY>`` token.
Gradients are stopped from the rest of the reasoning module so the reasoning
pathways learn what to answer from the standard cross-entropy loss while readiness
independently learns when. That isolation is not automatic: the ``<RDY>`` token is
read out of the long branch's output, so adding ``L_rdy`` to the total loss and
calling backward would push the timing gradient through the entire reasoning
stack. It is enforced in ``StreamReady._readiness_surrogate``, and
``train_stage`` additionally offers the two objectives as separate stages.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReadinessHead(nn.Module):
    """Small MLP mapping the ``<RDY>`` token to a readiness logit.

    Returns a *logit*; callers apply the sigmoid. Keeping it in logit space lets
    the contrastive loss of Eq. 7 operate on differences without saturating.
    """

    def __init__(self, width: int, hidden: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden = hidden or width
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )
        # Start near R_pred = 0.5 and let the timing loss shape it, rather than
        # biasing the model toward answering or waiting before it has learned.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, rdy_token: torch.Tensor) -> torch.Tensor:
        """``[B, C]`` or ``[B, T, C]`` -> logits with the trailing axis dropped."""
        return self.net(rdy_token).squeeze(-1)


@torch.no_grad()
def pseudo_evidence_regions(
    z_l_seq: torch.Tensor,
    centroids: torch.Tensor,
    centroid_mask: torch.Tensor,
    positive_quantile: float = 0.7,
    negative_quantile: float = 0.3,
    project: Optional[torch.nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weak pseudo-supervision for readiness (paper Sec. 3.4).

    Measures, at each timestep, how strongly the reasoning representation aligns
    with centroid-level memory, then splits timesteps into a pseudo-positive
    region P (high alignment) and pseudo-negative region N (low alignment) by
    per-stream quantiles. Quantiles rather than a fixed cut because the
    similarity scale varies across videos.

    Note this is computed from the *reasoning output* ``z_l``, not from a raw
    query-memory match, so the signal reflects the outcome of reasoning. The paper
    validates these pseudo-labels post hoc against ProReady-QA's annotated
    windows (mean temporal IoU 0.87 over 100 samples).

    Args:
        z_l_seq:       ``[B, T, C_q]`` long-term representation per timestep.
        centroids:     ``[B, J, P, C_v]``
        centroid_mask: ``[B, J]``
        project: maps ``C_q -> C_v`` so the two can be compared. ``z_l`` lives in
            Q-Former width while memory lives in visual-encoder width, so a map is
            required; the caller passes the retrieval centroid projection ``W_c``,
            which is already the operator trained to score centroids against a
            query. Omit only when the two widths already agree.

    Returns:
        ``(pos, neg)`` boolean masks of shape ``[B, T]``.
    """
    if project is not None:
        z_l_seq = project(z_l_seq)
    b, t, c = z_l_seq.shape
    # fp32 with autocast off: autocast would make the einsum fp16, and
    # torch.quantile below requires a float or double tensor.
    with torch.cuda.amp.autocast(enabled=False):
        pooled = centroids.mean(dim=2).float()                      # [B, J, C]
        zn = F.normalize(z_l_seq.float(), dim=-1)
        cn = F.normalize(pooled, dim=-1)

        sim = torch.einsum("btc,bjc->btj", zn, cn).float()          # [B, T, J]
        sim = sim.masked_fill(~centroid_mask[:, None, :], float("-inf"))
        # Best-matching centroid per timestep: evidence is "present" if *some*
        # centroid aligns well, not if the average does.
        score = sim.max(dim=2).values                               # [B, T]
        score = torch.nan_to_num(score, neginf=0.0, posinf=0.0, nan=0.0)

    hi = torch.quantile(score, positive_quantile, dim=1, keepdim=True)
    lo = torch.quantile(score, negative_quantile, dim=1, keepdim=True)
    return score >= hi, score <= lo


def readiness_loss(
    readiness_logits: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    lambda_reg: float = 0.1,
    max_pairs: int = 64,
) -> torch.Tensor:
    """``L_rdy`` of Eq. 8: pairwise contrastive term plus temporal coherence.

    Eq. 7:  ``L_ctr = -log sigmoid( R(t+) - R(t-) )`` for ``t+`` in P, ``t-`` in N.
    Eq. 8:  ``L_rdy = L_ctr + lambda_reg * || grad_t R(t) ||_1``

    The regulariser matters: the paper notes the contrastive term alone yields
    noisy, unstable readiness curves, and the L1 penalty on the temporal gradient
    smooths them without forcing a shape.

    Args:
        readiness_logits: ``[B, T]`` pre-sigmoid scores.
        pos_mask, neg_mask: ``[B, T]`` booleans from :func:`pseudo_evidence_regions`.
        lambda_reg: paper supplementary Sec. D, 0.1.
        max_pairs: cap on sampled (t+, t-) pairs per stream, for cost.

    Returns:
        Scalar loss. Zero (with grad) for streams that lack either region, so a
        degenerate batch cannot produce a NaN.
    """
    b, t = readiness_logits.shape
    device = readiness_logits.device
    terms = []

    for i in range(b):
        p = torch.nonzero(pos_mask[i], as_tuple=False).flatten()
        n = torch.nonzero(neg_mask[i], as_tuple=False).flatten()
        if p.numel() == 0 or n.numel() == 0:
            continue
        # Sample a bounded set of cross-region pairs.
        np_ = min(p.numel(), max_pairs)
        nn_ = min(n.numel(), max_pairs)
        pi = p[torch.randint(p.numel(), (np_,), device=device)]
        ni = n[torch.randint(n.numel(), (nn_,), device=device)]
        k = min(np_, nn_)
        diff = readiness_logits[i, pi[:k]] - readiness_logits[i, ni[:k]]
        terms.append(F.logsigmoid(diff).mean().neg())

    if terms:
        l_ctr = torch.stack(terms).mean()
    else:
        l_ctr = readiness_logits.sum() * 0.0

    if t > 1:
        # L1 norm of the temporal gradient of the readiness *score*.
        r = torch.sigmoid(readiness_logits)
        l_reg = (r[:, 1:] - r[:, :-1]).abs().mean()
    else:
        l_reg = readiness_logits.sum() * 0.0

    return l_ctr + lambda_reg * l_reg
