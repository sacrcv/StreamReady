"""
Visual Memory Tree -- StreamReady paper Sec. 3.2.

Three levels of decreasing temporal resolution, maintained online as frames arrive:

    M_V1  FIFO buffer of the most recent raw frame embeddings (short-term detail)
    M_V2  centroid set, capacity J, seeded by K-means and updated by EMA  (Eq. 1)
    M_V3  prototype set, capacity U, abstracted from M_V2 by EMA          (Eq. 2)

The point of the hierarchy is a *fixed* memory footprint: once all three levels are
at capacity the cost of retrieval stops growing with video length, which is what
keeps latency flat on hour-long streams (paper Sec. 5.3).

Each level also records which prototype every centroid belongs to, because
retrieval is coarse-to-fine: prototypes are scored first, then only the centroids
hanging off the selected prototypes are considered (paper Eq. 5).


"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim, eps=1e-6)


def kmeans(
    x: torch.Tensor,
    k: int,
    iters: int = 10,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Cosine K-means over ``x`` of shape ``[N, D]``, returning ``[k, D]``.

    Used to seed M_V2 the first time the frame buffer fills, and to realign M_V3
    when its prototypes drift apart. Deliberately small and dependency-free: k is
    at most a few hundred and this runs a handful of times per video, so the cost
    is negligible next to the ViT.

    **Seeding is deterministic.** This function used to pick its first centre
    with ``torch.randint`` off the global RNG, which made the whole memory tree
    non-reproducible: two identical evaluation runs produced different centroids
    once the frame buffer filled, and because the per-stream loop in
    :meth:`VisualMemoryTree.update` draws sequentially, a video's memory also
    depended on which batch position it landed in. ARS is a timing metric, so
    scores that move between runs of the same checkpoint are not usable.

    The fix is to drop the RNG rather than seed it: the first centre is the point
    whose direction is farthest from the data's mean direction, and k-means++'s
    greedy far-point rule picks the rest. That is a standard deterministic
    initialisation, it uses the same cosine geometry as the rest of the function,
    and it makes a video's memory a pure function of its frames -- independent of
    batch composition, shuffling, DDP sharding and whatever ran before it. The
    paper specifies "K-means clustering" without pinning the seeding, so this is
    as faithful as the random version and reproducible.

    Args:
        generator: opt in to the old stochastic first-centre selection, for
            anyone who wants to measure seeding variance. Leave ``None`` for
            reproducible runs.
    """
    n = x.shape[0]
    if n == 0:
        return x.new_zeros((0, x.shape[-1]))
    if n <= k:
        return x.clone()

    xn = _l2(x)

    # k-means++-lite: one starting centre, the rest greedily far from the chosen
    # set.
    if generator is None:
        mean_dir = _l2(xn.mean(dim=0, keepdim=True))
        first = int(torch.argmin((xn @ mean_dir.t()).flatten()))
    else:
        first = int(torch.randint(n, (1,), generator=generator, device=x.device)[0])
    centres = [x[first]]

    for _ in range(k - 1):
        c = _l2(torch.stack(centres, 0))
        sim = xn @ c.t()                       # [N, chosen]
        far = (1.0 - sim.max(dim=1).values)    # distance to nearest chosen
        centres.append(x[int(torch.argmax(far))])
    c = torch.stack(centres, 0)

    for _ in range(iters):
        assign = (xn @ _l2(c).t()).argmax(dim=1)             # [N]
        new_c = c.clone()
        for j in range(k):
            m = assign == j
            if bool(m.any()):
                new_c[j] = x[m].mean(0)
        if torch.allclose(new_c, c, atol=1e-5):
            c = new_c
            break
        c = new_c
    return c


class VisualMemoryTree(nn.Module):
    """Stateful three-level visual memory for one batch of streams.

    State is allocated lazily on the first :meth:`update` and must be cleared
    between videos with :meth:`reset`. Every tensor carries a leading batch axis
    so a batch of independent streams advances in lockstep.

    Shapes, with ``B`` streams, ``P`` tokens per slot and width ``C``:

    ===============  ====================  ========================================
    attribute        shape                 meaning
    ===============  ====================  ========================================
    ``frames``       ``[B, L1, N, C]``     raw frame embeddings (M_V1)
    ``frame_count``  ``[B]``               how many slots of M_V1 are populated
    ``centroids``    ``[B, J, P, C]``      centroid values (M_V2)
    ``centroid_mask````[B, J]``            which centroid slots are live
    ``cent_proto``   ``[B, J]``            prototype index each centroid belongs to
    ``prototypes``   ``[B, U, P, C]``      prototype values (M_V3)
    ``proto_mask``   ``[B, U]``            which prototype slots are live
    ===============  ====================  ========================================
    """

    def __init__(
        self,
        frame_capacity: int = 24,
        num_centroids: int = 96,
        num_prototypes: int = 12,
        ema_decay: float = 0.985,
        slot_tokens: int = 1,
        tau_base: float = 0.6,
        tau_adapt: float = 0.5,
        tau_min: float = 0.2,
        tau_max: float = 0.95,
        drift_tolerance: float = 0.25,
        kmeans_iters: int = 10,
    ):
        super().__init__()
        self.frame_capacity = int(frame_capacity)
        self.num_centroids = int(num_centroids)
        self.num_prototypes = int(num_prototypes)
        # Paper supplementary Sec. D: alpha = 0.985 for the EMA decay in Eqs. 1-2.
        # Note the equations write the update as (1-alpha)*old + alpha*new, so a
        # value this close to 1 means each slot tracks new evidence quickly.
        self.ema_decay = float(ema_decay)
        self.slot_tokens = int(slot_tokens)
        self.tau_base = float(tau_base)
        self.tau_adapt = float(tau_adapt)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.drift_tolerance = float(drift_tolerance)
        self.kmeans_iters = int(kmeans_iters)

        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        """Drop all memory. Call between videos."""
        self.frames: Optional[torch.Tensor] = None
        self.frame_count: Optional[torch.Tensor] = None
        self.centroids: Optional[torch.Tensor] = None
        self.centroid_mask: Optional[torch.Tensor] = None
        self.cent_proto: Optional[torch.Tensor] = None
        self.prototypes: Optional[torch.Tensor] = None
        self.proto_mask: Optional[torch.Tensor] = None
        self._sim_ema: Optional[torch.Tensor] = None
        self._new_cluster_rate: Optional[torch.Tensor] = None
        self._seeded: Optional[torch.Tensor] = None

    def _alloc(self, b: int, n: int, c: int, device, dtype):
        p = self.slot_tokens
        self.frames = torch.zeros(b, self.frame_capacity, n, c, device=device, dtype=dtype)
        self.frame_count = torch.zeros(b, dtype=torch.long, device=device)
        self.centroids = torch.zeros(b, self.num_centroids, p, c, device=device, dtype=dtype)
        self.centroid_mask = torch.zeros(b, self.num_centroids, dtype=torch.bool, device=device)
        self.cent_proto = torch.full((b, self.num_centroids), -1, dtype=torch.long, device=device)
        self.prototypes = torch.zeros(b, self.num_prototypes, p, c, device=device, dtype=dtype)
        self.proto_mask = torch.zeros(b, self.num_prototypes, dtype=torch.bool, device=device)
        self._sim_ema = torch.full((b,), self.tau_base, device=device, dtype=torch.float32)
        self._new_cluster_rate = torch.zeros(b, device=device, dtype=torch.float32)
        self._seeded = torch.zeros(b, dtype=torch.bool, device=device)

    # ------------------------------------------------------------- descriptors
    def _descriptor(self, frame: torch.Tensor) -> torch.Tensor:
        """Reduce a frame ``[B, N, C]`` to a slot-shaped descriptor ``[B, P, C]``."""
        if self.slot_tokens == 1:
            return frame.mean(dim=1, keepdim=True)
        # Adaptive pool along the token axis so P slots cover the frame evenly.
        x = frame.transpose(1, 2)                                  # [B, C, N]
        x = F.adaptive_avg_pool1d(x.float(), self.slot_tokens)     # [B, C, P]
        return x.transpose(1, 2).to(frame.dtype)                   # [B, P, C]

    @staticmethod
    def _flat(x: torch.Tensor) -> torch.Tensor:
        """``[B, K, P, C]`` -> ``[B, K, P*C]`` for similarity computations."""
        b, k = x.shape[0], x.shape[1]
        return x.reshape(b, k, -1)

    # ------------------------------------------------------------------ update
    @torch.no_grad()
    def update(self, frame: torch.Tensor):
        """Admit one timestep's frame embedding ``[B, N, C]`` into the memory.

        Runs under ``no_grad``: the memory is a running summary of *detached*
        evidence, exactly as MA-LMM-style banks are maintained. Gradients reach
        the Q-Formers through the current frame and the retrieved slots, not
        through the EMA updates.
        """
        b, n, c = frame.shape
        if self.frames is None:
            self._alloc(b, n, c, frame.device, frame.dtype)

        frame = frame.detach()
        full = self.frame_count >= self.frame_capacity

        # --- M_V1: FIFO admit, capturing whatever is evicted ---
        evicted = torch.zeros(b, n, c, device=frame.device, dtype=frame.dtype)
        has_evicted = full.clone()
        if bool(full.any()):
            evicted[full] = self.frames[full, 0]
            self.frames[full] = torch.roll(self.frames[full], shifts=-1, dims=1)
            self.frames[full, -1] = frame[full]
        not_full = ~full
        if bool(not_full.any()):
            idx = self.frame_count[not_full]
            self.frames[not_full, idx] = frame[not_full]
            self.frame_count[not_full] = idx + 1

        if not bool(has_evicted.any()):
            return

        # --- first eviction seeds M_V2 by K-means over the full buffer (Sec. 3.2) ---
        to_seed = has_evicted & (~self._seeded)
        for i in torch.nonzero(to_seed, as_tuple=False).flatten().tolist():
            desc = self._descriptor(self.frames[i])                  # [L1, P, C]
            flat = desc.reshape(desc.shape[0], -1).float()
            k = min(self.num_centroids, flat.shape[0])
            cent = kmeans(flat, k, iters=self.kmeans_iters)
            self.centroids[i, :k] = cent.reshape(k, self.slot_tokens, c).to(self.centroids.dtype)
            self.centroid_mask[i, :k] = True
            self._seeded[i] = True
            self._rebuild_prototypes(i)

        # --- subsequent evictions update M_V2 by EMA (Eq. 1) ---
        rest = torch.nonzero(has_evicted, as_tuple=False).flatten().tolist()
        for i in rest:
            self._admit_centroid(i, self._descriptor(evicted[i : i + 1])[0])

    def _tau(self, i: int) -> float:
        """Adaptive merge threshold for stream ``i`` (see module docstring)."""
        s = float(self._sim_ema[i])
        tau = self.tau_base - self.tau_adapt * (s - self.tau_base)
        return float(min(max(tau, self.tau_min), self.tau_max))

    @torch.no_grad()
    def _admit_centroid(self, i: int, desc: torch.Tensor):
        """Merge ``desc`` ``[P, C]`` into stream ``i``'s centroid set, or add it."""
        live = self.centroid_mask[i]
        flat_desc = desc.reshape(1, -1).float()

        best_sim, best_j = -1.0, -1
        if bool(live.any()):
            cents = self.centroids[i][live].reshape(int(live.sum()), -1).float()
            sims = (_l2(flat_desc) @ _l2(cents).t()).flatten()
            k = int(torch.argmax(sims))
            best_sim = float(sims[k])
            best_j = int(torch.nonzero(live, as_tuple=False).flatten()[k])

        # Track how well recent frames match memory; drives the adaptive threshold.
        self._sim_ema[i] = 0.9 * self._sim_ema[i] + 0.1 * max(best_sim, 0.0)
        tau = self._tau(i)

        a = self.ema_decay
        if best_j >= 0 and best_sim >= tau:
            # Eq. 1, merge branch.
            self.centroids[i, best_j] = (
                (1.0 - a) * self.centroids[i, best_j].float() + a * desc.float()
            ).to(self.centroids.dtype)
            self._new_cluster_rate[i] *= 0.9
        else:
            free = torch.nonzero(~live, as_tuple=False).flatten()
            if free.numel() > 0:
                # Eq. 1, new-centroid branch.
                j = int(free[0])
                self.centroids[i, j] = desc.to(self.centroids.dtype)
                self.centroid_mask[i, j] = True
                self._assign_prototype(i, j)
            elif best_j >= 0:
                # At capacity J: fold into the nearest centroid instead of growing.
                # Keeps the footprint fixed, which is the whole point of the tree.
                self.centroids[i, best_j] = (
                    (1.0 - a) * self.centroids[i, best_j].float() + a * desc.float()
                ).to(self.centroids.dtype)
            self._new_cluster_rate[i] = 0.9 * self._new_cluster_rate[i] + 0.1

        # M_V3 is refreshed when M_V2 is saturated or drifting (paper Sec. 3.2).
        saturated = bool(self.centroid_mask[i].all())
        drifting = float(self._new_cluster_rate[i]) > self.drift_tolerance
        if saturated or drifting:
            self._update_prototypes(i, realign=drifting)

    # -------------------------------------------------------------- prototypes
    @torch.no_grad()
    def _rebuild_prototypes(self, i: int):
        """Cluster stream ``i``'s live centroids into M_V3 from scratch."""
        live = self.centroid_mask[i]
        if not bool(live.any()):
            return
        cents = self.centroids[i][live]
        flat = cents.reshape(cents.shape[0], -1).float()
        u = min(self.num_prototypes, flat.shape[0])
        protos = kmeans(flat, u, iters=self.kmeans_iters)
        self.prototypes[i, :u] = protos.reshape(
            u, self.slot_tokens, self.centroids.shape[-1]
        ).to(self.prototypes.dtype)
        self.proto_mask[i] = False
        self.proto_mask[i, :u] = True
        # Re-assign every live centroid to its nearest prototype.
        assign = (_l2(flat) @ _l2(protos).t()).argmax(dim=1)
        self.cent_proto[i] = -1
        self.cent_proto[i, torch.nonzero(live, as_tuple=False).flatten()] = assign

    @torch.no_grad()
    def _assign_prototype(self, i: int, j: int):
        """Attach a freshly created centroid to its nearest existing prototype."""
        live = self.proto_mask[i]
        if not bool(live.any()):
            self._rebuild_prototypes(i)
            return
        protos = self.prototypes[i][live].reshape(int(live.sum()), -1).float()
        cent = self.centroids[i, j].reshape(1, -1).float()
        k = int(torch.argmax((_l2(cent) @ _l2(protos).t()).flatten()))
        self.cent_proto[i, j] = int(torch.nonzero(live, as_tuple=False).flatten()[k])

    @torch.no_grad()
    def _update_prototypes(self, i: int, realign: bool = False):
        """EMA-refresh M_V3 from its member centroids (Eq. 2).

        ``realign`` triggers the paper's "lightweight mini-K-means" path, used when
        centroids have become heterogeneous and the current grouping no longer
        reflects the data.
        """
        if realign:
            self._rebuild_prototypes(i)
            self._new_cluster_rate[i] = 0.0
            return

        a = self.ema_decay
        for u in torch.nonzero(self.proto_mask[i], as_tuple=False).flatten().tolist():
            members = (self.cent_proto[i] == u) & self.centroid_mask[i]
            if not bool(members.any()):
                continue
            mean_c = self.centroids[i][members].float().mean(dim=0)
            self.prototypes[i, u] = (
                (1.0 - a) * self.prototypes[i, u].float() + a * mean_c
            ).to(self.prototypes.dtype)

    # ----------------------------------------------------------------- readers
    def short_term(self) -> Optional[torch.Tensor]:
        """Live contents of M_V1 as ``[B, L*N, C]``, or ``None`` if empty.

        Streams shorter than the buffer are padded with zeros up to the longest
        stream in the batch; the padding is zero-valued so it contributes nothing
        through cross-attention.

        Returns a **clone**, not a view. The caller feeds this into the Q-Former,
        so autograd keeps a reference to it until backward -- while the next
        :meth:`update` rewrites the same buffer in place to advance the FIFO. A
        view would have its version counter bumped underneath the graph and
        backward would fail. Cloning costs nothing in gradient terms because the
        buffer holds detached frames by construction.
        """
        if self.frames is None:
            return None
        L = int(self.frame_count.max())
        if L == 0:
            return None
        b, _, n, c = self.frames.shape
        return self.frames[:, :L].reshape(b, L * n, c).clone()

    def has_long_term(self) -> bool:
        return self.centroid_mask is not None and bool(self.centroid_mask.any())
