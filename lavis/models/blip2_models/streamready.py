"""
StreamReady -- readiness-aware streaming video understanding.

Paper: "StreamReady: Learning What to Answer and When in Long Streaming Videos",
CVPR 2026.

This file is the composition layer. It holds three things and little else:

* ``self.backbone`` -- a frozen Qwen2-VL vision tower and language decoder
  (``lavis/models/streamready/backbone.py``). The paper's backbone, used as the
  matched pair Qwen trained together.
* ``self.core`` -- the StreamReady module: Visual Memory Tree, coarse-to-fine
  retrieval, dual-branch Q-Former with the Eq. 6 fusion, contextual memory and
  the readiness mechanism (``lavis/models/streamready/module.py``). This is
  everything the paper contributes, and everything that trains.
* the two training objectives, and the gradient isolation between them.

The dataflow for one forward pass over ``T`` timesteps::

    frames --> Qwen2-VL vision tower (frozen, no_grad)      [B, T, N, 3584]
           --> StreamReadyModule
                 1. admit the frame into the Visual Memory Tree       (Sec. 3.2)
                    M_V1 raw frames -> M_V2 EMA centroids -> M_V3 prototypes
                 2. short branch Q_s reasons over M_V1 -> z_s             (Eq. 3)
                 3. Top-K prototypes, then Top-m of their centroids   (Eqs. 4-5)
                 4. long branch Q_l reasons over the retrieved slots
                    conditioned on z_s, with <RDY> riding along           (Eq. 6)
                 5. contextual memory fuses relevant past answers      (Sec. 3.3)
                 6. the readiness head reads <RDY> -> R_pred(t)        (Sec. 3.4)
           --> Qwen2-VL decoder (frozen)                       answer tokens

Eq. 6 is ``z_l = Q_l(Concat[retrieved], z_s)``: ``z_s`` is a *second argument* to
the long branch, exactly as in HierarQ, not extra tokens concatenated onto the
long query sequence. See ``lavis/models/streamready/fusion.py``. Unlike HierarQ
there is no query memory bank; the Visual Memory Tree is the only visual-side
memory.

Trainable: both Q-Former branches, the vision projection, the retrieval
projections, the contextual fusion, the ``<RDY>`` token, the readiness head and
the output projection into the decoder. Frozen: the whole Qwen2-VL backbone
(paper supplementary Sec. D).

Training is **separated** (Sec. 3.4). Reasoning is supervised by cross-entropy on
the answer; readiness by the Eq. 7-8 contrastive loss over pseudo evidence
timestamps, which updates only the ``<RDY>`` token and the readiness head.
``train_stage`` selects ``reasoning``, ``readiness`` or ``joint``; even in
``joint`` the timing gradient is isolated, see :meth:`_readiness_surrogate`.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base
from lavis.models.streamready.backbone import Qwen2VLBackbone
from lavis.models.streamready.module import BLIP2_ENCODER_WIDTH, StreamReadyModule
from lavis.models.streamready.readiness import pseudo_evidence_regions, readiness_loss


@registry.register_model("streamready")
class StreamReady(Blip2Base):
    """StreamReady on a frozen Qwen2-VL backbone."""

    PRETRAINED_MODEL_CONFIG_DICT = {
        "qwen2_vl": "configs/models/streamready/streamready_qwen2_vl.yaml",
    }

    def __init__(
        self,
        # --- frozen backbone ---
        backbone_model="llm/Qwen2-VL-7B-Instruct",
        image_size=224,
        backbone_precision="fp16",
        attn_implementation="sdpa",
        # --- Q-Former ---
        num_query_token=32,
        qformer_text_input=True,
        qformer_encoder_width=BLIP2_ENCODER_WIDTH,
        # --- visual memory tree (paper Sec. 3.2) ---
        frame_capacity=24,
        num_centroids=96,
        num_prototypes=12,
        ema_decay=0.985,
        slot_tokens=1,
        # --- query-aware retrieval (paper Eqs. 4-5) ---
        topk_prototypes=8,
        topk_centroids=24,
        # --- short -> long branch fusion (paper Eq. 6) ---
        long_term_fusion_mode="per_layer",
        fusion_num_heads=None,
        # --- contextual memory (paper Sec. 3.2/3.3) ---
        use_contextual_memory=True,
        context_capacity=32,
        context_topk=4,
        context_sim_floor=0.2,
        # --- readiness (paper Sec. 3.4) ---
        use_readiness=True,
        readiness_threshold=0.35,
        readiness_hidden=None,
        lambda_reg=0.1,
        readiness_loss_weight=1.0,
        train_stage="joint",
        # --- misc ---
        num_frames=100,
        max_txt_len=128,
        max_output_txt_len=256,
        prompt="",
        gradient_checkpointing=False,
    ):
        super().__init__()

        if train_stage not in ("joint", "reasoning", "readiness"):
            raise ValueError(
                f"train_stage must be joint, reasoning or readiness, "
                f"got {train_stage!r}"
            )
        self.train_stage = train_stage

        self.num_frames = num_frames
        self.image_size = image_size
        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        self.use_readiness = use_readiness
        self.readiness_threshold = readiness_threshold
        self.lambda_reg = lambda_reg
        self.readiness_loss_weight = readiness_loss_weight
        self.use_contextual_memory = use_contextual_memory

        # ---------------- frozen Qwen2-VL backbone ----------------
        self.backbone = Qwen2VLBackbone(
            backbone_model,
            torch_dtype=torch.float16 if backbone_precision == "fp16" else torch.float32,
            attn_implementation=attn_implementation,
        )

        # ---------------- StreamReady (everything trainable) ----------------
        self.core = StreamReadyModule(
            vision_width=self.backbone.vision_width,
            output_width=self.backbone.llm_width,
            num_query_token=num_query_token,
            qformer_text_input=qformer_text_input,
            qformer_encoder_width=qformer_encoder_width,
            frame_capacity=frame_capacity,
            num_centroids=num_centroids,
            num_prototypes=num_prototypes,
            ema_decay=ema_decay,
            slot_tokens=slot_tokens,
            topk_prototypes=topk_prototypes,
            topk_centroids=topk_centroids,
            long_term_fusion_mode=long_term_fusion_mode,
            fusion_num_heads=fusion_num_heads,
            use_contextual_memory=use_contextual_memory,
            context_capacity=context_capacity,
            context_topk=context_topk,
            context_sim_floor=context_sim_floor,
            use_readiness=use_readiness,
            readiness_hidden=readiness_hidden,
            max_txt_len=max_txt_len,
            gradient_checkpointing=gradient_checkpointing,
        )

        # DDP note. Each Q-Former branch is called once per timestep inside a
        # single forward, which DDP handles via find_unused_parameters=True (set
        # by the runner). static_graph=True would be cheaper, but the graph is
        # genuinely iteration-dependent: the contextual fusion is skipped on a
        # dialogue's first turn and whenever nothing clears the relevance floor,
        # and the number of streaming steps varies with the question timestamp.
        self.ddp_static_graph = False

        self._apply_train_stage()
        self._lemmatizer = None

    # ---------------------------------------------------------------- plumbing
    @property
    def llm_tokenizer(self):
        """The backbone's tokenizer, under the name the tasks expect."""
        return self.backbone.tokenizer

    @staticmethod
    def externally_restored_prefixes() -> Tuple[str, ...]:
        """Parameter prefixes a checkpoint does not need to carry.

        The entire Qwen2-VL backbone is frozen for all of training and reloaded
        from its own checkpoint, so writing it into every StreamReady checkpoint
        only costs disk. Everything under ``core.`` belongs in the checkpoint
        regardless of whether the current ``train_stage`` has it frozen -- see
        ``RunnerBase.trainable_state_dict``.
        """
        return ("backbone.",)

    def readiness_parameters(self) -> List[nn.Parameter]:
        """The parameters ``L_rdy`` is allowed to update (paper Sec. 3.4)."""
        return self.core.readiness_parameters()

    def reasoning_parameters(self) -> List[nn.Parameter]:
        """Everything else trainable: what cross-entropy is allowed to update."""
        return self.core.reasoning_parameters()

    def _apply_train_stage(self) -> None:
        """Freeze whichever half of the model this stage does not train.

        Paper Sec. 3.4 trains the two objectives separately. ``reasoning`` and
        ``readiness`` realise that as two stages over the same architecture;
        ``joint`` runs both losses in one pass and relies on
        :meth:`_readiness_surrogate` to keep ``L_rdy`` off the reasoning
        modules.
        """
        if self.train_stage == "reasoning":
            frozen = self.readiness_parameters()
            label = "readiness head + <RDY> token"
        elif self.train_stage == "readiness":
            frozen = self.reasoning_parameters()
            label = "reasoning pathway"
        else:
            logging.info(
                "StreamReady train_stage=joint: both losses in one pass, with "
                "L_rdy gradient-isolated to the readiness head and <RDY> token"
            )
            return

        n = 0
        for p in frozen:
            if p.requires_grad:
                p.requires_grad = False
                n += 1
        logging.info(
            "StreamReady train_stage=%s: froze %d %s tensor(s)",
            self.train_stage, n, label,
        )

    # -------------------------------------------------------------------- core
    def stream(
        self,
        image: torch.Tensor,
        text_input: List[str],
        collect_readiness: bool = True,
        trigger_threshold: Optional[float] = None,
        early_exit: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Encode frames with the frozen tower, then run the StreamReady module.

        ``image`` is ``[B, C, T, H, W]`` pixels, or ``[B, T, N, C]`` features
        already cached from the same frozen tower
        (``tools/extract_vit_features.py``).

        ``trigger_threshold`` enables readiness gating; see
        :meth:`StreamReadyModule.forward`. Training leaves it unset.
        """
        if image.dim() == 5:
            feats = self.backbone.encode_frames(image)
        elif image.dim() == 4:
            feats = image
        else:
            raise ValueError(
                f"expected 5D pixels or 4D cached features, got {image.dim()}D"
            )
        return self.core(
            feats.detach(), text_input,
            collect_readiness=collect_readiness,
            trigger_threshold=trigger_threshold,
            early_exit=early_exit,
        )

    # ------------------------------------------------------- training objectives
    def _readiness_surrogate(self, l_rdy: torch.Tensor) -> torch.Tensor:
        """Turn ``L_rdy`` into a term that updates only the readiness parameters.

        Paper Sec. 3.4 trains readiness separately from reasoning: ``L_rdy``
        updates the ``<RDY>`` token and the readiness head, while the reasoning
        pathway learns *what* to answer from cross-entropy alone. Simply adding
        ``L_rdy`` to the total loss does not achieve that -- the ``<RDY>`` token
        is read out of the long branch's output, so a plain backward pushes the
        timing gradient through the whole long Q-Former, the Eq. 6 fusion, the
        short branch and the retrieval projections.

        So the gradient is taken explicitly with respect to the readiness
        parameters only, and replayed as a linear surrogate. Backward through
        ``sum(g_i . p_i)`` deposits exactly ``g_i`` on ``p_i`` and touches
        nothing else. The ``- detach + detach`` pair makes the returned tensor's
        *value* equal ``L_rdy``, so the logged loss stays meaningful while the
        gradient is the isolated one.

        Note the token still influences ``z_l`` in the forward direction, and so
        is still shaped by cross-entropy: that is the architecture of Sec. 3.4,
        which deliberately places the token inside the reasoning branch. What is
        prevented here is the reverse, readiness rewriting the reasoning modules.

        Cost: one extra backward traversal of the long branch per step. Use
        ``train_stage: readiness`` to train timing on its own instead.
        """
        params = [p for p in self.readiness_parameters() if p.requires_grad]
        grads = torch.autograd.grad(
            l_rdy, params, retain_graph=True, allow_unused=True,
        ) if params else ()

        surrogate = None
        for g, p in zip(grads, params):
            if g is None:
                continue
            term = (g.detach() * p).sum()
            surrogate = term if surrogate is None else surrogate + term

        if surrogate is None:
            # Nothing to update (all readiness params frozen). Keep the value for
            # logging, drop the gradient.
            return l_rdy.detach()
        return surrogate - surrogate.detach() + l_rdy.detach()

    def _readiness_term(self, out: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """``L_rdy`` from Eqs. 7-8, gradient-isolated to the readiness params."""
        if not self.use_readiness:
            return None
        if out["readiness"] is None or out["z_l_seq"] is None:
            return None

        # Pseudo-labels come from detached reasoning output: the readiness
        # pathway must not shape what counts as evidence.
        pos, neg = pseudo_evidence_regions(
            out["z_l_seq"].detach(),
            self.core.memory.centroids,
            self.core.memory.centroid_mask,
            # z_l is in Q-Former width, memory in the Q-Former's cross-attention
            # width. W_c is the projection already used to score centroids
            # against a query, so it is the natural operator here too.
            project=self.core.query_proj.w_cent,
        )
        l_rdy = readiness_loss(
            out["readiness"], pos, neg, lambda_reg=self.lambda_reg
        )
        return self.readiness_loss_weight * l_rdy

    # ----------------------------------------------------------------- forward
    def forward(self, samples):
        out = self.stream(
            samples["image"], samples["text_input"],
            collect_readiness=self.use_readiness,
        )

        losses = {}

        # --- reasoning: cross-entropy on the answer (paper Sec. 3.3) ---
        # Skipped entirely in the readiness stage, where the frozen reasoning
        # pathway makes the decoder forward pure overhead.
        if self.train_stage in ("joint", "reasoning"):
            losses["loss"] = self._reasoning_loss(out, samples)

        # --- readiness: contrastive timing loss (paper Eqs. 7-8) ---
        if self.train_stage in ("joint", "readiness"):
            l_rdy = self._readiness_term(out)
            if l_rdy is not None:
                term = self._readiness_surrogate(l_rdy)
                losses["loss_readiness"] = l_rdy.detach()
                losses["loss"] = (
                    losses["loss"] + term if "loss" in losses else term
                )

        if "loss" not in losses:
            raise RuntimeError(
                f"train_stage={self.train_stage!r} produced no loss. The "
                "readiness stage needs use_readiness=True."
            )
        return losses

    def _reasoning_loss(self, out, samples) -> torch.Tensor:
        """Video-text cross-entropy through the frozen decoder."""
        text_input = samples["text_input"]
        text_output = samples["text_output"]
        prefix = out["prefix_embeds"]
        device = prefix.device

        tok = self.backbone.tokenizer
        atts_prefix = torch.ones(prefix.shape[:-1], dtype=torch.long, device=device)

        tok.padding_side = "right"
        tok.truncation_side = "left"
        text_tokens = tok(
            text_input, return_tensors="pt", padding="longest",
            truncation=True, max_length=self.max_txt_len,
        ).to(device)

        tok.truncation_side = "right"
        out_tokens = tok(
            [t + tok.eos_token for t in text_output],
            return_tensors="pt", padding="longest",
            truncation=True, max_length=self.max_output_txt_len,
        ).to(device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_tokens.input_ids, text_tokens.attention_mask,
            out_tokens.input_ids, out_tokens.attention_mask,
        )

        targets = llm_tokens["input_ids"].masked_fill(
            llm_tokens["input_ids"] == tok.pad_token_id, -100
        )
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100
        empty_targets = torch.ones(
            atts_prefix.size(), dtype=torch.long, device=device
        ).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        text_embeds = self.backbone.embed_tokens(llm_tokens["input_ids"])
        inputs_embeds = torch.cat([prefix.to(text_embeds.dtype), text_embeds], dim=1)
        attention_mask = torch.cat([atts_prefix, llm_tokens["attention_mask"]], dim=1)

        _, loss = self.backbone.lm_forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=targets,
        )
        return loss

    # -------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=64,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        """Answer each question, reporting *when* the model became ready.

        The returned ``answer_times`` are what ARS consumes. A trigger at
        timestep ``k`` is mapped back to wall-clock seconds through the
        ``frame_times`` the dataset provides; without it we fall back to the
        fraction of the stream observed.
        """
        # Readiness gating happens *inside* the streaming pass (paper Sec. 3.4):
        # z_l is frozen at the timestep each stream declares itself ready, so the
        # answer below is conditioned on exactly the evidence observed at that
        # moment. Reading the curve afterwards and generating from the
        # whole-video representation would report an online timestamp for an
        # answer that had already seen the rest of the video.
        out = self.stream(
            samples["image"], samples["text_input"],
            collect_readiness=True,
            trigger_threshold=self.readiness_threshold if self.use_readiness else None,
        )
        prefix = out["prefix_embeds"]
        device = prefix.device
        b = prefix.shape[0]

        # Map each trigger timestep back to wall-clock seconds.
        answer_times: List[float] = []
        if out["trigger_step"] is not None:
            frame_times = samples.get("frame_times")
            steps = out["trigger_step"].tolist()
            n_steps = out["z_l_seq"].shape[1] if out["z_l_seq"] is not None else 1
            for i, k in enumerate(steps):
                if frame_times is not None:
                    ft = frame_times[i]
                    # trigger_step is an absolute index into the sampled frames,
                    # so this needs no offset correction.
                    answer_times.append(float(ft[min(int(k), ft.shape[0] - 1)]))
                else:
                    answer_times.append(float(int(k) + 1) / max(n_steps, 1))
        else:
            answer_times = [float("nan")] * b

        tok = self.backbone.tokenizer
        tok.padding_side = "left"
        llm_tokens = tok(
            samples["text_input"], return_tensors="pt", padding="longest"
        ).to(device)

        text_embeds = self.backbone.embed_tokens(llm_tokens.input_ids)
        inputs_embeds = torch.cat([prefix.to(text_embeds.dtype), text_embeds], dim=1)
        atts_prefix = torch.ones(prefix.shape[:-1], dtype=torch.long, device=device)
        attention_mask = torch.cat([atts_prefix, llm_tokens.attention_mask], dim=1)

        gen = self.backbone.lm_generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            max_new_tokens=max_length,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        text = tok.batch_decode(gen, skip_special_tokens=True)
        return [t.strip() for t in text], answer_times

    @torch.no_grad()
    def generate_dialogue(self, samples, max_length=30):
        """Answer a whole dialogue from one continuous streaming pass.

        This is the paper's inference setting for multi-turn data (Sec. 3.1):
        the video streams once, each question is serviced from its own ask time,
        and every answered turn writes its representation into the contextual
        memory bank for the turns that follow.

        ``samples`` carries per-turn lists, one inner list per dialogue, as built
        by ``ProReadyQADataset.collater``. Dialogues are processed one at a time
        because turn counts and ask times differ; see
        :meth:`StreamReadyModule.stream_dialogue`.

        Returns ``(answers, answer_times)``, both lists-of-lists shaped like the
        incoming per-turn fields.
        """
        images = samples["image"]
        all_answers, all_times = [], []

        for d in range(images.shape[0]):
            questions = samples["text_input"][d]
            ask_steps = samples["ask_steps"][d]

            feats = self.backbone.encode_frames(images[d : d + 1])
            out = self.core.stream_dialogue(
                feats.detach(), questions, ask_steps,
                trigger_threshold=self.readiness_threshold,
            )

            # Every turn's prefix is ready now, so the decoder runs once for the
            # whole dialogue. Deferring generation like this is exactly
            # equivalent: the bank stores the representation handed to the
            # decoder, never the decoded text.
            answers = self._decode(out["prefix_embeds"], questions, max_length)

            ft = samples["frame_times"][d]
            steps = out["trigger_step"].tolist()
            times = [float(ft[min(int(k), ft.shape[0] - 1)]) for k in steps]

            all_answers.append(answers)
            all_times.append(times)

        return all_answers, all_times

    @torch.no_grad()
    def _decode(self, prefix, questions: List[str], max_length: int) -> List[str]:
        """Run the frozen decoder on a batch of prefix embeddings."""
        device = prefix.device
        tok = self.backbone.tokenizer
        tok.padding_side = "left"
        llm_tokens = tok(questions, return_tensors="pt", padding="longest").to(device)

        text_embeds = self.backbone.embed_tokens(llm_tokens.input_ids)
        inputs_embeds = torch.cat([prefix.to(text_embeds.dtype), text_embeds], dim=1)
        atts_prefix = torch.ones(prefix.shape[:-1], dtype=torch.long, device=device)
        attention_mask = torch.cat([atts_prefix, llm_tokens.attention_mask], dim=1)

        gen = self.backbone.lm_generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_length,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        return [t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)]

    @torch.no_grad()
    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs,
    ):
        # Dialogue-structured batches (per-turn lists) go through the continuous
        # multi-turn pass; flat single-question batches through the gated
        # single-question pass.
        if "ask_steps" in samples:
            return self.generate_dialogue(samples, max_length=max_len)
        answers, answer_times = self.generate(
            samples, num_beams=num_beams, max_length=max_len,
            min_length=min_len, length_penalty=length_penalty,
        )
        return answers, answer_times

    # ------------------------------------------------------- checkpoint loading
    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens["input_ids"].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:],
                ])
            )
            llm_tokens["attention_mask"].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:],
                ])
            )
        llm_tokens["input_ids"] = torch.stack(llm_tokens["input_ids"])
        llm_tokens["attention_mask"] = torch.stack(llm_tokens["attention_mask"])
        return llm_tokens, input_part_targets_len

    def _expand_state_dict_for_dual_branch(self, state_dict):
        """Clone single-Q-Former checkpoint weights into both branches.

        A BLIP-2 / InstructBLIP checkpoint carries one ``Qformer.*`` tower and
        one ``query_tokens``. StreamReady has two independent branches, so the
        weights are duplicated into each (no sharing afterwards) -- this is what
        "initialize the dual-branch Q-Former using pretrained weights" means in
        practice. The branches live under ``core.`` because they belong to the
        pluggable module rather than to this composition layer.
        """
        new_state = dict(state_dict)
        for key, value in list(state_dict.items()):
            if key.startswith("Qformer."):
                tail = key[len("Qformer.") :]
                for prefix in ("core.short_qformer.", "core.long_qformer."):
                    if prefix + tail not in new_state:
                        new_state[prefix + tail] = value.clone()
            elif key == "query_tokens":
                for target in ("core.short_query_tokens", "core.long_query_tokens"):
                    if target not in new_state:
                        new_state[target] = value.clone()
        return new_state

    def _drop_size_mismatched_keys(self, state_dict):
        """``load_state_dict(strict=False)`` still raises on shape mismatch.

        Changing ``num_query_token`` (or any dim affecting parameter shapes)
        would otherwise make a pretrained checkpoint unloadable; drop those keys
        and keep the fresh init, logging exactly what was dropped.
        """
        own_state = self.state_dict()
        kept, dropped = {}, []
        for key, value in state_dict.items():
            if key in own_state and own_state[key].shape != value.shape:
                dropped.append(
                    f"{key}: ckpt {tuple(value.shape)} vs "
                    f"model {tuple(own_state[key].shape)}"
                )
                continue
            kept[key] = value
        if dropped:
            logging.warning(
                "Dropping %d shape-mismatched checkpoint keys (kept the model's "
                "init for these): %s", len(dropped), "; ".join(dropped),
            )
        return kept

    def _load_any(self, url_or_filename):
        from lavis.common.dist_utils import download_cached_file
        from lavis.common.utils import is_url

        if is_url(url_or_filename):
            cached = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached, map_location="cpu")
        else:
            assert os.path.isfile(url_or_filename), "checkpoint url or path is invalid"
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        state_dict = self._expand_state_dict_for_dual_branch(state_dict)
        state_dict = self._drop_size_mismatched_keys(state_dict)
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info("load checkpoint from %s" % url_or_filename)
        return msg

    def load_from_pretrained(self, url_or_filename):
        return self._load_any(url_or_filename)

    def load_checkpoint(self, url_or_filename):
        msg = self._load_any(url_or_filename)
        logging.info("Missing keys {}".format(msg.missing_keys))
        return msg

    @classmethod
    def from_config(cls, cfg):
        model = cls(
            backbone_model=cfg.get("backbone_model", "llm/Qwen2-VL-7B-Instruct"),
            image_size=cfg.get("image_size", 224),
            backbone_precision=cfg.get("backbone_precision", "fp16"),
            attn_implementation=cfg.get("attn_implementation", "sdpa"),
            num_query_token=cfg.get("num_query_token", 32),
            qformer_text_input=cfg.get("qformer_text_input", True),
            qformer_encoder_width=cfg.get(
                "qformer_encoder_width", BLIP2_ENCODER_WIDTH
            ),
            frame_capacity=cfg.get("frame_capacity", 24),
            num_centroids=cfg.get("num_centroids", 96),
            num_prototypes=cfg.get("num_prototypes", 12),
            ema_decay=cfg.get("ema_decay", 0.985),
            slot_tokens=cfg.get("slot_tokens", 1),
            topk_prototypes=cfg.get("topk_prototypes", 8),
            topk_centroids=cfg.get("topk_centroids", 24),
            long_term_fusion_mode=cfg.get("long_term_fusion_mode", "per_layer"),
            fusion_num_heads=cfg.get("fusion_num_heads", None),
            use_contextual_memory=cfg.get("use_contextual_memory", True),
            context_capacity=cfg.get("context_capacity", 32),
            context_topk=cfg.get("context_topk", 4),
            context_sim_floor=cfg.get("context_sim_floor", 0.2),
            use_readiness=cfg.get("use_readiness", True),
            readiness_threshold=cfg.get("readiness_threshold", 0.35),
            readiness_hidden=cfg.get("readiness_hidden", None),
            lambda_reg=cfg.get("lambda_reg", 0.1),
            readiness_loss_weight=cfg.get("readiness_loss_weight", 1.0),
            train_stage=cfg.get("train_stage", "joint"),
            num_frames=cfg.get("num_frames", 100),
            max_txt_len=cfg.get("max_txt_len", 128),
            max_output_txt_len=cfg.get("max_output_txt_len", 256),
            prompt=cfg.get("prompt", ""),
            gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        )

        if cfg.get("load_pretrained", True) and cfg.get("pretrained", None):
            model.load_from_pretrained(url_or_filename=cfg.get("pretrained"))
        model.load_checkpoint_from_config(cfg)
        return model
