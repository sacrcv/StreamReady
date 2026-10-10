"""
Short-term -> long-term branch fusion.

This is how the long-term branch consumes the short-term representation in
Eq. 6, ``z_l = Q_l(Concat[M_V3t, M_V2t, q_i], z_s)``. Note ``z_s`` is a second
argument there, not another entry in the concatenated memory: it does not enter
through the cross-attention over retrieved slots, it is fused into the long
branch's own query stream.

The mechanism is the one introduced in HierarQ, where a scene-level Q-Former
extends an entity-level one with two extra submodules per block, applied after
the usual self-attention and visual cross-attention:

  (3) an extra self-attention over the queries learned so far, and
  (4) a cross-attention with Q from the long-term queries and K, V from the
      short-term queries.

"""

import logging
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from transformers.modeling_utils import apply_chunking_to_forward

from lavis.models.blip2_models.Qformer import BertLayer


class _FromModule:
    """Sentinel: "take the short-term queries from module state"."""

    def __repr__(self):  # pragma: no cover - debugging aid only
        return "<from module state>"


_FROM_MODULE = _FromModule()


class _SelfAttentionBlock(nn.Module):
    """Standard transformer-encoder self-attention block.

    Forward: LN -> MHA(self) -> Add -> LN -> FFN -> Add.
    Zero-init residuals so the block is exact identity at construction.
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_ratio: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * ffn_ratio, embed_dim),
        )
        self._zero_init_residuals()

    def _zero_init_residuals(self) -> None:
        nn.init.zeros_(self.self_attn.out_proj.weight)
        if self.self_attn.out_proj.bias is not None:
            nn.init.zeros_(self.self_attn.out_proj.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        if self.ffn[-1].bias is not None:
            nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm(x)
        attn_out, _ = self.self_attn(
            h, h, h, key_padding_mask=key_padding_mask, need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class _CrossAttentionBlock(nn.Module):
    """Standard transformer-decoder cross-attention block.

    Forward: LN -> MHA(query=x, key/value=ctx) -> Add -> LN -> FFN -> Add.
    Zero-init residuals, same reason as above.
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_ratio: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        ffn_hidden = embed_dim * ffn_ratio
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, embed_dim),
        )
        self._zero_init_residuals()

    def _zero_init_residuals(self) -> None:
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        if self.cross_attn.out_proj.bias is not None:
            nn.init.zeros_(self.cross_attn.out_proj.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        if self.ffn[-1].bias is not None:
            nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor,
                ctx_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:   query tokens, ``[B, Nq, C]``
            ctx: key/value tokens, ``[B, Nk, C]``
            ctx_key_padding_mask: bool ``[B, Nk]`` where True means "ignore".
        """
        q = self.norm_q(x)
        kv = self.norm_kv(ctx)
        attn_out, _ = self.cross_attn(
            query=q, key=kv, value=kv,
            key_padding_mask=ctx_key_padding_mask, need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class LongTermFusionLayers(nn.Module):
    """The long-term branch's two extra submodules.

    ``fuse(z_l, z_s)``: an extra self-attention over the long-term queries,
    then a cross-attention with Q from those queries and K, V from the
    short-term queries. Identity at construction.
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 ffn_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.self_attn = _SelfAttentionBlock(
            embed_dim, num_heads, ffn_ratio=ffn_ratio, dropout=dropout)
        self.cross_attn = _CrossAttentionBlock(
            embed_dim, num_heads, ffn_ratio=ffn_ratio, dropout=dropout)

    def forward(self, long_q: torch.Tensor, short_q: torch.Tensor,
                short_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.self_attn(long_q)
        x = self.cross_attn(x, short_q, ctx_key_padding_mask=short_key_padding_mask)
        return x


class StreamReadyLongBertLayer(BertLayer):
    """A ``BertLayer`` extended with the two long-term fusion submodules.

    Applied after the usual self-attention and the visual cross-attention over
    retrieved memory. Only the query stream (the first ``query_length``
    positions) is fused; the text stream passes through untouched, matching how
    the stock layer keeps the two streams separate.

    Stays a drop-in ``BertLayer`` when ``short_queries`` is ``None``, so the
    same module can be called without fusion.
    """

    # Set by set_long_term_fusion_queries(); the LongTermFusionLayers module is
    # registered as a submodule named ``long_term_fusion``.
    short_queries = None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        query_length=0,
        short_queries=_FROM_MODULE,
    ):
        # ``short_queries`` normally comes from module state, published by
        # set_long_term_fusion_queries(). Under gradient checkpointing that
        # state is already cleared by the time the block is recomputed, so
        # BertEncoder passes the tensor explicitly instead; an explicit
        # ``None`` therefore has to keep meaning "no fusion", which is why the
        # default is a sentinel rather than ``None``.
        if short_queries is _FROM_MODULE:
            short_queries = self.short_queries
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]

        if query_length > 0:
            query_attention_output = attention_output[:, :query_length, :]

            if self.has_cross_attention:
                assert encoder_hidden_states is not None, \
                    "encoder_hidden_states must be given for cross-attention layers"
                cross_attention_outputs = self.crossattention(
                    query_attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                query_attention_output = cross_attention_outputs[0]
                outputs = outputs + cross_attention_outputs[1:-1]

                # ---- the two extra submodules ----
                if short_queries is not None:
                    query_attention_output = self.long_term_fusion(
                        query_attention_output, short_queries
                    )

            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                query_attention_output,
            )
            if attention_output.shape[1] > query_length:
                layer_output_text = apply_chunking_to_forward(
                    self.feed_forward_chunk,
                    self.chunk_size_feed_forward,
                    self.seq_len_dim,
                    attention_output[:, query_length:, :],
                )
                layer_output = torch.cat([layer_output, layer_output_text], dim=1)
        else:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
        outputs = (layer_output,) + outputs
        outputs = outputs + (present_key_value,)
        return outputs


class ShortLayerOutputCapture:
    """Record each short-branch layer's query output, keyed by depth.

    Used by the depth-matched fusion mode, where long-branch block *i* is fused
    with the short branch's block *i* output rather than with its final output.

    Information only ever flows short -> long, so capturing the short branch's
    per-layer outputs and replaying them into the long branch is numerically
    identical to running the two in lockstep, and far simpler.
    """

    def __init__(self, bert_model, query_length):
        self.bert_model = bert_model
        self.query_length = query_length
        self.outputs: Dict[int, torch.Tensor] = {}
        self._handles = []

    def _make_hook(self, layer_idx):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.outputs[layer_idx] = hidden[:, : self.query_length, :]
        return hook

    def __enter__(self):
        for idx, layer in enumerate(self.bert_model.encoder.layer):
            self._handles.append(layer.register_forward_hook(self._make_hook(idx)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


def apply_long_term_fusion(model, num_heads=12, ffn_ratio=4, dropout=0.0,
                           cross_attention_only=True):
    """Give the long branch's ``BertLayer``s their two extra submodules.

    Class reassignment rather than subclass construction, so the pretrained
    weights already loaded into each layer are preserved untouched.

    By default only blocks that already cross-attend are patched: those are the
    ones that have just read visual evidence, which is the point in the block
    where fusing short-term context is meaningful.
    """
    hidden = model.config.hidden_size
    patched = 0
    for idx, layer in enumerate(model.encoder.layer):
        if cross_attention_only and not getattr(layer, "has_cross_attention", False):
            continue
        layer.__class__ = StreamReadyLongBertLayer
        layer.add_module(
            "long_term_fusion",
            LongTermFusionLayers(embed_dim=hidden, num_heads=num_heads,
                                 ffn_ratio=ffn_ratio, dropout=dropout),
        )
        layer.long_term_fusion_layer_idx = idx
        layer.short_queries = None
        patched += 1
    logging.info(
        "StreamReady: attached long-term fusion to %d Q-Former block(s) "
        "(cross_attention_only=%s)", patched, cross_attention_only,
    )
    return model


def set_long_term_fusion_queries(
    model, short_queries: Optional[Union[torch.Tensor, Dict[int, torch.Tensor]]]
):
    """Publish the short-term queries onto the long branch before a call.

    Accepts:
      * a tensor -- every fused block sees the same ``z_s``;
      * a ``{depth: tensor}`` dict -- depth-matched fusion, falling back to the
        deepest available entry for blocks beyond what was captured;
      * ``None`` -- clears the published queries, so the layers behave as stock
        ``BertLayer``s again.

    Always clear after the call; a stale tensor would silently fuse the previous
    timestep's context into the next one.
    """
    if isinstance(short_queries, dict) and short_queries:
        deepest = max(short_queries)
    else:
        deepest = None

    for module in model.modules():
        if not isinstance(module, StreamReadyLongBertLayer):
            continue
        if short_queries is None:
            module.short_queries = None
        elif isinstance(short_queries, dict):
            idx = getattr(module, "long_term_fusion_layer_idx", None)
            module.short_queries = short_queries.get(idx, short_queries[deepest])
        else:
            module.short_queries = short_queries
