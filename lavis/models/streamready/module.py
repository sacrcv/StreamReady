"""
StreamReady as a self-contained, backbone-agnostic module.

Everything StreamReady contributes lives here: the Visual Memory Tree, the
coarse-to-fine retrieval, the dual-branch Q-Former with its Eq. 6 fusion, the
contextual memory bank, and the readiness mechanism. Everything it is *hosted
by* -- the frozen vision tower and the frozen language decoder -- lives in
``backbone.py``.

The contract is deliberately narrow, which is what makes the block pluggable::

    module = StreamReadyModule(vision_width=..., output_width=...)

    # one independent question per batch row (training, single-turn benchmarks)
    out = module(frame_feats, questions)
    #  frame_feats  [B, T, N_tokens, vision_width]  from any frozen encoder
    #  out["prefix_embeds"]  [B, Q, output_width]    ready for any decoder
    #  out["readiness"]      [B, T']                 readiness logit per timestep

    # one continuous stream answering a dialogue in order (multi-turn inference)
    out = module.stream_dialogue(frame_feats, questions, ask_steps, threshold)
    #  out["prefix_embeds"]  [n_turns, Q, output_width]
    #  out["trigger_step"]   [n_turns]               when each turn answered

Nothing in here knows what produced ``frame_feats`` or what consumes
``prefix_embeds``. Hosting it on a different VLM means writing a backbone that
reports its two widths and exposes an embedding-level decoder entry point; the
module is unchanged.

Width note. A frozen encoder's token width rarely matches the width BLIP-2's
Q-Former cross-attention was pretrained at (1408, EVA-CLIP ViT-G). Qwen2-VL emits
3584. Setting ``qformer_encoder_width`` inserts a trainable projection onto that
pretrained width, so the Q-Former's cross-attention keys and values keep their
BLIP-2 initialisation -- which is what the paper means by initialising the
dual-branch Q-Former from BLIP-2 weights. Setting it to ``None`` instead widens
the cross-attention to the encoder and gives up that initialisation.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from lavis.models.streamready.contextual_memory import ContextualMemoryBank
from lavis.models.streamready.fusion import (
    LongTermFusionLayers,
    ShortLayerOutputCapture,
    apply_long_term_fusion,
    set_long_term_fusion_queries,
)
from lavis.models.streamready.readiness import ReadinessHead
from lavis.models.streamready.retrieval import QueryProjection, coarse_to_fine_retrieve
from lavis.models.streamready.visual_memory import VisualMemoryTree

# The width BLIP-2 / InstructBLIP pretrained the Q-Former cross-attention at.
BLIP2_ENCODER_WIDTH = 1408


class StreamReadyModule(nn.Module):
    """Reasoning + readiness, between a frozen encoder and a frozen decoder."""

    def __init__(
        self,
        vision_width: int,
        output_width: int,
        num_query_token: int = 32,
        qformer_text_input: bool = True,
        qformer_encoder_width: Optional[int] = BLIP2_ENCODER_WIDTH,
        # --- visual memory tree (paper Sec. 3.2) ---
        frame_capacity: int = 24,
        num_centroids: int = 96,
        num_prototypes: int = 12,
        ema_decay: float = 0.985,
        slot_tokens: int = 1,
        # --- query-aware retrieval (paper Eqs. 4-5) ---
        topk_prototypes: int = 8,
        topk_centroids: int = 24,
        # --- Eq. 6 fusion ---
        long_term_fusion_mode: str = "per_layer",
        fusion_num_heads: Optional[int] = None,
        # --- contextual memory (paper Sec. 3.2/3.3) ---
        use_contextual_memory: bool = True,
        context_capacity: int = 32,
        context_topk: int = 4,
        context_sim_floor: float = 0.2,
        # --- readiness (paper Sec. 3.4) ---
        use_readiness: bool = True,
        readiness_hidden: Optional[int] = None,
        max_txt_len: int = 128,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        # Imported here rather than at module scope: blip2 pulls in the LAVIS
        # model registry, and importing it from a module the registry itself
        # loads would be circular.
        from lavis.models.blip2_models.blip2 import Blip2Base

        self.vision_width = vision_width
        self.output_width = output_width
        self.num_query_token = num_query_token
        self.qformer_text_input = qformer_text_input
        self.topk_prototypes = topk_prototypes
        self.topk_centroids = topk_centroids
        self.use_readiness = use_readiness
        self.max_txt_len = max_txt_len

        self.tokenizer = Blip2Base.init_tokenizer(truncation_side="left")

        # ---- optional projection onto the Q-Former's pretrained width ----
        if qformer_encoder_width and qformer_encoder_width != vision_width:
            self.vision_proj = nn.Linear(vision_width, qformer_encoder_width)
            encoder_width = qformer_encoder_width
            logging.info(
                "StreamReady: projecting encoder tokens %d -> %d so the "
                "Q-Former keeps its BLIP-2 cross-attention init",
                vision_width, encoder_width,
            )
        else:
            self.vision_proj = None
            encoder_width = vision_width
        self.encoder_width = encoder_width

        # ---- dual-branch Q-Former (paper Sec. 3.3) ----
        # Two independent branches, both initialised from BLIP-2/InstructBLIP.
        # Neither carries a query memory bank: the only memory in StreamReady is
        # the Visual Memory Tree, which lives outside the Q-Formers.
        self.short_qformer, self.short_query_tokens = Blip2Base.init_Qformer(
            num_query_token, encoder_width,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.long_qformer, self.long_query_tokens = Blip2Base.init_Qformer(
            num_query_token, encoder_width,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.long_term_fusion_mode = long_term_fusion_mode
        fusion_heads = fusion_num_heads or self.long_qformer.config.num_attention_heads
        if long_term_fusion_mode.startswith("per_layer"):
            # Eq. 6: z_s enters the long branch through two extra submodules per
            # cross-attending block, not as extra tokens in its memory.
            self.long_qformer.bert = apply_long_term_fusion(
                self.long_qformer.bert, num_heads=fusion_heads
            )
            self.post_fusion = None
        elif long_term_fusion_mode == "post":
            # Cheaper approximation: fuse once on the long branch's output.
            self.post_fusion = LongTermFusionLayers(
                embed_dim=self.long_qformer.config.hidden_size, num_heads=fusion_heads
            )
        else:
            raise ValueError(
                f"long_term_fusion_mode must be one of 'per_layer', "
                f"'per_layer_matched', 'post'; got {long_term_fusion_mode!r}"
            )

        for qformer in (self.short_qformer, self.long_qformer):
            if not qformer_text_input:
                qformer.bert.embeddings.word_embeddings = None
                qformer.bert.embeddings.position_embeddings = None
                for layer in qformer.bert.encoder.layer:
                    layer.output = None
                    layer.intermediate = None
            else:
                qformer.resize_token_embeddings(len(self.tokenizer))
            qformer.cls = None

        qformer_width = self.short_qformer.config.hidden_size
        self.qformer_width = qformer_width

        # ---- visual memory tree ----
        self.memory = VisualMemoryTree(
            frame_capacity=frame_capacity,
            num_centroids=num_centroids,
            num_prototypes=num_prototypes,
            ema_decay=ema_decay,
            slot_tokens=slot_tokens,
        )
        # Retrieval scores a question against memory slots, which live in
        # whatever width the Q-Former cross-attends at.
        self.query_proj = QueryProjection(qformer_width, encoder_width)

        # ---- contextual memory ----
        if use_contextual_memory:
            self.context = ContextualMemoryBank(
                width=qformer_width,
                capacity=context_capacity,
                num_heads=self.short_qformer.config.num_attention_heads,
                topk=context_topk,
                sim_floor=context_sim_floor,
            )
        else:
            self.context = None

        # ---- readiness mechanism (paper Sec. 3.4) ----
        if use_readiness:
            # The <RDY> token is appended to z_l inside the long-term branch, so
            # it co-evolves with query-aligned evidence (the best of the three
            # placements ablated in the paper).
            self.rdy_token = nn.Parameter(torch.zeros(1, 1, qformer_width))
            self.rdy_token.data.normal_(mean=0.0, std=0.02)
            self.readiness_head = ReadinessHead(qformer_width, hidden=readiness_hidden)
        else:
            self.rdy_token = None
            self.readiness_head = None

        # ---- projection into the host decoder ----
        self.output_proj = nn.Linear(qformer_width, output_width)

    # ------------------------------------------------------------- parameters
    def readiness_parameters(self) -> List[nn.Parameter]:
        """The parameters ``L_rdy`` is allowed to update (paper Sec. 3.4)."""
        params = list(self.readiness_head.parameters()) if self.use_readiness else []
        if self.rdy_token is not None:
            params.append(self.rdy_token)
        return params

    def reasoning_parameters(self) -> List[nn.Parameter]:
        """Everything else trainable: what cross-entropy is allowed to update."""
        readiness = {id(p) for p in self.readiness_parameters()}
        return [p for p in self.parameters() if id(p) not in readiness]

    # ---------------------------------------------------------------- helpers
    def _param_dtype(self) -> torch.dtype:
        """The dtype this module computes in."""
        return self.output_proj.weight.dtype

    def tokenize_question(self, text_input: List[str], device):
        return self.tokenizer(
            text_input,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

    def _question_vector(self, q_toks, b: int, device) -> torch.Tensor:
        """Pooled question embedding in Q-Former width, for retrieval and gating."""
        if q_toks is None:
            return torch.zeros(b, self.qformer_width, device=device)
        emb = self.short_qformer.bert.embeddings.word_embeddings(q_toks.input_ids)
        mask = q_toks.attention_mask.unsqueeze(-1).to(emb.dtype)
        return (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def _run_branch(
        self,
        qformer,
        query_tokens: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor],
        q_toks,
        extra_query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one Q-Former branch over ``memory`` ``[B, L, C]``.

        ``extra_query`` lets the long-term branch carry the ``<RDY>`` token
        alongside its own learnable queries.
        """
        b = memory.shape[0]
        queries = query_tokens.expand(b, -1, -1)
        if extra_query is not None:
            queries = torch.cat([queries, extra_query], dim=1)

        atts = torch.ones(queries.shape[:-1], dtype=torch.long, device=memory.device)
        if memory_mask is None:
            memory_mask = torch.ones(
                memory.shape[:-1], dtype=torch.long, device=memory.device
            )
        memory_mask = memory_mask.long()

        if self.qformer_text_input and q_toks is not None:
            attn = torch.cat([atts, q_toks.attention_mask], dim=1)
            out = qformer.bert(
                q_toks.input_ids,
                query_embeds=queries,
                attention_mask=attn,
                encoder_hidden_states=memory,
                encoder_attention_mask=memory_mask,
                return_dict=True,
            )
        else:
            out = qformer.bert(
                query_embeds=queries,
                attention_mask=atts,
                encoder_hidden_states=memory,
                encoder_attention_mask=memory_mask,
                return_dict=True,
            )
        return out.last_hidden_state[:, : queries.shape[1], :]

    # ------------------------------------------------------------------- core
    def forward(
        self,
        frame_feats: torch.Tensor,
        text_input: List[str],
        collect_readiness: bool = True,
        trigger_threshold: Optional[float] = None,
        early_exit: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Stream over ``frame_feats`` ``[B, T, N, vision_width]``, one question
        per batch row.

        This is the single-question path, used for training and for benchmarks
        whose questions are independent. Multi-turn dialogues go through
        :meth:`stream_dialogue` instead, which keeps one continuous stream and
        accumulates the contextual memory bank across turns.

        Walks the video one timestep at a time so the memory tree, retrieval and
        readiness all see the same causal prefix an online model would.

        Args:
            trigger_threshold: when given, enables readiness gating (paper Sec.
                3.4): the returned ``z_l`` is frozen at the timestep each stream
                first crosses the threshold, rather than being the
                whole-video representation. This is what makes the answer
                conditioned on exactly the evidence observed at the moment the
                model declared itself ready, so the timestamp ARS scores is the
                one the answer actually came from. Leave ``None`` during
                training: Sec. 3.4 notes training has full-video access, and
                gating on an untrained readiness head would be circular.
            early_exit: stop the loop once *every* stream has fired. Equivalent
                in output, cheaper, and closer to true online behaviour -- but it
                truncates the returned readiness curve, so leave it off when
                sweeping the threshold or inspecting the curve.

        Returns:
            ``z_l``            ``[B, Q, C_q]``  representation handed to the decoder
            ``prefix_embeds``  ``[B, Q, C_out]`` the same, projected for the decoder
            ``z_l_seq``        ``[B, T', C_q]`` pooled z_l per timestep
            ``readiness``      ``[B, T']``      readiness logit per timestep, or None
            ``trigger_step``   ``[B]``          timestep each stream fired at, or None
        """
        if frame_feats.dim() != 4:
            raise ValueError(
                f"expected [B, T, N, C] frame features, got {frame_feats.dim()}D"
            )
        b, t = frame_feats.shape[0], frame_feats.shape[1]
        device = frame_feats.device

        # A frozen encoder typically runs in fp16 while this module is trained in
        # fp32, so the features are cast to the module's own dtype at the
        # boundary rather than forcing either side to match the other.
        frame_feats = frame_feats.to(self._param_dtype())

        # Gradients start here: the encoder is frozen and detached upstream, but
        # the projection onto the Q-Former's width is trained.
        if self.vision_proj is not None:
            frame_feats = self.vision_proj(frame_feats)

        q_toks = (
            self.tokenize_question(text_input, device)
            if self.qformer_text_input else None
        )

        self.memory.reset()
        if self.context is not None:
            # Each row is an independent question here, so there is no prior turn
            # to reuse: the bank starts empty and stays empty. See the README note
            # on what that means for the contextual fusion weights.
            self.context.reset()

        # Retrieval scoring uses the question once; it does not change while the
        # stream advances.
        q_vec = self._question_vector(q_toks, b, device)
        q_proto, q_cent = self.query_proj(q_vec)

        z_l_seq, rdy_logits, z_l = [], [], None

        # Readiness gating state. Each stream is captured independently because
        # rows of a batch fire at different timesteps; continuing the loop for an
        # already-captured row costs compute but changes nothing, which is why
        # `early_exit` only breaks once every row has fired.
        gating = trigger_threshold is not None
        captured_z_l = None
        captured_step = torch.full((b,), -1, dtype=torch.long, device=device)
        step = 0

        for step in range(t):
            self.memory.update(frame_feats[:, step])

            short_mem = self.memory.short_term()
            if short_mem is None:
                continue

            # Eqs. 3-6. Shared with stream_dialogue so the two paths cannot drift.
            z_l, rdy_logit = self._reason_once(
                short_mem, q_toks, q_proto, q_cent, b=b
            )
            if rdy_logit is not None and collect_readiness:
                rdy_logits.append(rdy_logit)

            z_l_seq.append(z_l.mean(dim=1))

            # --- readiness gating: freeze z_l at the step each stream fires ---
            if gating and rdy_logit is not None:
                if captured_z_l is None:
                    captured_z_l = torch.zeros_like(z_l)
                fired = torch.sigmoid(rdy_logit.float()) >= trigger_threshold
                newly = fired & (captured_step < 0)
                if bool(newly.any()):
                    # torch.where rather than indexed assignment: keeps this
                    # usable under autograd if anyone enables gating with grad.
                    captured_z_l = torch.where(newly[:, None, None], z_l, captured_z_l)
                    captured_step = torch.where(
                        newly, torch.full_like(captured_step, step), captured_step
                    )
                if early_exit and bool((captured_step >= 0).all()):
                    break

        if z_l is None:
            raise RuntimeError(
                "the streaming pass produced no timesteps; check num_frames > 0"
            )

        trigger_step = None
        if gating:
            # A stream that never crossed the threshold declined to answer within
            # the video. Fall back to everything it observed, and report the last
            # step -- which ARS scores as a late answer rather than silently as
            # on-time.
            never = captured_step < 0
            if captured_z_l is None:
                captured_z_l = z_l
            else:
                captured_z_l = torch.where(never[:, None, None], z_l, captured_z_l)
            captured_step = torch.where(
                never, torch.full_like(captured_step, step), captured_step
            )
            z_l = captured_z_l
            trigger_step = captured_step

        # --- contextual reasoning (paper Sec. 3.3) ---
        # Applied to whichever z_l is being handed on, so a gated run fuses
        # context into the trigger-step representation rather than a later one.
        if self.context is not None:
            z_l = self.context(z_l, q_vec)

        return {
            "z_l": z_l,
            "prefix_embeds": self.output_proj(z_l),
            "z_l_seq": torch.stack(z_l_seq, dim=1) if z_l_seq else None,
            "readiness": torch.stack(rdy_logits, dim=1) if rdy_logits else None,
            "trigger_step": trigger_step,
            "q_vec": q_vec,
        }

    # ------------------------------------------------------- multi-turn stream
    def stream_dialogue(
        self,
        frame_feats: torch.Tensor,
        questions: List[str],
        ask_steps: List[int],
        trigger_threshold: float,
    ) -> Dict[str, torch.Tensor]:
        """One continuous streaming pass answering a whole dialogue in order.

        This is the paper's actual inference setting (Sec. 3.1, 3.3): the video
        streams once, questions arrive at their own timestamps, and each answered
        turn writes its representation into the Contextual Memory Bank so later
        turns can reuse it.

        Two things follow from doing it this way rather than one pass per turn.

        * **The memory tree is built once.** Per-turn items rebuild the whole
          visual memory for every question, which is both wasteful and not what
          "streaming" means -- a question asked at minute 40 should see the memory
          the stream has already accumulated, not a replay.
        * **M_C holds the model's own representations.** Eq. for ``a_i`` in
          Sec. 3.2 is the fused representation that was passed to the LLM, so it
          can only come from having actually answered the earlier turn. Priming
          the bank from the dataset's ground-truth answer text instead would both
          store the wrong object and leak the answers.

        Turns are serviced strictly in order: a later question cannot be answered
        before an earlier one, which is what makes the bank's contents
        well-defined. A turn that is still not ready at the final frame is forced
        to answer there, which ARS scores as late.

        Args:
            frame_feats: ``[1, T, N, vision_width]``. Batching dialogues is not
                supported -- turn counts and ask times differ per dialogue, and
                the bookkeeping to mask that is not worth the risk. Use
                ``batch_size_eval: 1``.
            questions:   ``n_turns`` question strings, in dialogue order.
            ask_steps:   ``n_turns`` frame indices, the step each question is
                asked at. Must be non-decreasing.
            trigger_threshold: readiness score above which the model answers.

        Returns:
            ``prefix_embeds`` ``[n_turns, Q, C_out]`` decoder input per turn
            ``trigger_step``  ``[n_turns]``           frame index each turn answered at
            ``readiness``     list of 1-D tensors, the logit trace per turn
        """
        if frame_feats.dim() != 4:
            raise ValueError(
                f"expected [1, T, N, C] frame features, got {frame_feats.dim()}D"
            )
        if frame_feats.shape[0] != 1:
            raise ValueError(
                f"stream_dialogue handles one dialogue at a time, got batch "
                f"{frame_feats.shape[0]}. Set batch_size_eval: 1."
            )
        n_turns = len(questions)
        if n_turns == 0:
            raise ValueError("stream_dialogue needs at least one question")
        if len(ask_steps) != n_turns:
            raise ValueError(
                f"{n_turns} questions but {len(ask_steps)} ask steps"
            )

        device = frame_feats.device
        t = frame_feats.shape[1]

        frame_feats = frame_feats.to(self._param_dtype())
        if self.vision_proj is not None:
            frame_feats = self.vision_proj(frame_feats)

        # One stream, so the memory is reset once rather than per turn.
        self.memory.reset()
        if self.context is not None:
            self.context.reset()

        # Precompute each turn's question tokens, pooled embedding and retrieval
        # queries. None of these change while the stream advances.
        turn_toks, turn_qvec, turn_queries = [], [], []
        for q in questions:
            toks = self.tokenize_question([q], device) if self.qformer_text_input else None
            qv = self._question_vector(toks, 1, device)
            turn_toks.append(toks)
            turn_qvec.append(qv)
            turn_queries.append(self.query_proj(qv))

        captured: List[Optional[torch.Tensor]] = [None] * n_turns
        trigger = [t - 1] * n_turns
        curves: List[List[torch.Tensor]] = [[] for _ in range(n_turns)]

        turn = 0
        for step in range(t):
            self.memory.update(frame_feats[:, step])
            if turn >= n_turns:
                break

            short_mem = self.memory.short_term()
            if short_mem is None:
                continue

            # Service every question whose ask time has passed, earliest first.
            # Answering one may expose the next, so this is a loop, not an `if`.
            while turn < n_turns and step >= int(ask_steps[turn]):
                q_toks = turn_toks[turn]
                q_proto, q_cent = turn_queries[turn]

                z_l, rdy_logit = self._reason_once(
                    short_mem, q_toks, q_proto, q_cent, b=1
                )
                if rdy_logit is not None:
                    curves[turn].append(rdy_logit.detach().flatten())

                ready = (
                    rdy_logit is not None
                    and float(torch.sigmoid(rdy_logit.float()).flatten()[0])
                    >= trigger_threshold
                )
                # The last frame forces an answer: the stream is over, so there is
                # no more evidence coming.
                if not (ready or step == t - 1):
                    break

                # Sec. 3.3/3.4: the representation handed to the LLM is z_l fused
                # with the relevant contextual memory, and *that* is what gets
                # stored as this turn's answer representation.
                fused = self.context(z_l, turn_qvec[turn]) if self.context is not None else z_l
                captured[turn] = fused
                trigger[turn] = step
                if self.context is not None:
                    self.context.write(turn_qvec[turn], fused.mean(dim=1))
                turn += 1

        # Any turn never serviced (its ask time fell past the end of the sampled
        # stream) answers from whatever the stream ended with.
        if any(c is None for c in captured):
            short_mem = self.memory.short_term()
            for i, c in enumerate(captured):
                if c is not None:
                    continue
                q_proto, q_cent = turn_queries[i]
                z_l, _ = self._reason_once(
                    short_mem, turn_toks[i], q_proto, q_cent, b=1
                )
                fused = (
                    self.context(z_l, turn_qvec[i])
                    if self.context is not None else z_l
                )
                captured[i] = fused
                trigger[i] = t - 1
                if self.context is not None:
                    self.context.write(turn_qvec[i], fused.mean(dim=1))

        z_l_all = torch.cat(captured, dim=0)                     # [n_turns, Q, C]
        return {
            "z_l": z_l_all,
            "prefix_embeds": self.output_proj(z_l_all),
            "trigger_step": torch.tensor(trigger, dtype=torch.long, device=device),
            "readiness": [
                torch.stack(c) if c else torch.zeros(0, device=device)
                for c in curves
            ],
        }

    def _reason_once(
        self, short_mem, q_toks, q_proto, q_cent, b: int
    ):
        """One question's worth of reasoning against the current memory state.

        Factored out of the streaming loops so the single-question and
        multi-turn paths cannot drift apart: both run Eq. 3, Eqs. 4-5 and Eq. 6
        through exactly this code.

        Returns ``(z_l, readiness_logit_or_None)``.
        """
        # --- Eq. 3: short-term reasoning over M_V1 ---
        if self.long_term_fusion_mode == "per_layer_matched":
            with ShortLayerOutputCapture(
                self.short_qformer.bert, self.num_query_token
            ) as cap:
                z_s = self._run_branch(
                    self.short_qformer, self.short_query_tokens,
                    short_mem, None, q_toks,
                )
            short_for_long = dict(cap.outputs)
        else:
            z_s = self._run_branch(
                self.short_qformer, self.short_query_tokens,
                short_mem, None, q_toks,
            )
            short_for_long = z_s

        # --- Eqs. 4-5: coarse-to-fine retrieval over M_V2/M_V3 ---
        if self.memory.has_long_term():
            slots, slot_mask = coarse_to_fine_retrieve(
                q_proto, q_cent,
                self.memory.prototypes, self.memory.proto_mask,
                self.memory.centroids, self.memory.centroid_mask,
                self.memory.cent_proto,
                self.topk_prototypes, self.topk_centroids,
            )
        else:
            slots, slot_mask = short_mem, None

        # --- Eq. 6: long-term reasoning, fused with z_s, carrying <RDY> ---
        extra = None
        if self.use_readiness:
            extra = self.rdy_token.expand(b, -1, -1).to(z_s.dtype)

        if self.post_fusion is None:
            set_long_term_fusion_queries(self.long_qformer.bert, short_for_long)
        try:
            out = self._run_branch(
                self.long_qformer, self.long_query_tokens, slots, slot_mask,
                q_toks, extra_query=extra,
            )
        finally:
            if self.post_fusion is None:
                set_long_term_fusion_queries(self.long_qformer.bert, None)

        z_l = out[:, : self.num_query_token, :]
        rdy_logit = None
        if self.use_readiness:
            rdy_logit = self.readiness_head(out[:, self.num_query_token, :])
        if self.post_fusion is not None:
            z_l = self.post_fusion(z_l, z_s)
        return z_l, rdy_logit
