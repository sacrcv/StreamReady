"""StreamReady.

The framework splits cleanly in two, which is what lets the StreamReady block be
hosted by a different VLM without touching it:

* :mod:`backbone`          -- the frozen host: Qwen2-VL's vision tower and decoder
* :mod:`module`            -- StreamReady itself, everything that trains

and inside the module:

* :mod:`visual_memory`     -- the three-level Visual Memory Tree (paper Sec. 3.2)
* :mod:`contextual_memory` -- the Contextual Memory Bank (paper Sec. 3.2)
* :mod:`retrieval`         -- coarse-to-fine query-aware retrieval (paper Eqs. 4-5)
* :mod:`fusion`            -- how the long-term branch consumes z_s (paper Eq. 6)
* :mod:`readiness`         -- the <RDY> token, readiness head and timing loss
                              (paper Sec. 3.4, Eqs. 7-8)

``module`` is imported lazily: it needs the vendored Q-Former, which lives under
the LAVIS model registry that imports this package.
"""
from lavis.models.streamready.backbone import Qwen2VLBackbone, pack_frames
from lavis.models.streamready.contextual_memory import ContextualMemoryBank
from lavis.models.streamready.fusion import (
    LongTermFusionLayers,
    ShortLayerOutputCapture,
    apply_long_term_fusion,
    set_long_term_fusion_queries,
)
from lavis.models.streamready.readiness import ReadinessHead, readiness_loss
from lavis.models.streamready.retrieval import coarse_to_fine_retrieve
from lavis.models.streamready.visual_memory import VisualMemoryTree

__all__ = [
    # host
    "Qwen2VLBackbone",
    "pack_frames",
    # StreamReady internals
    "VisualMemoryTree",
    "ContextualMemoryBank",
    "coarse_to_fine_retrieve",
    "LongTermFusionLayers",
    "ShortLayerOutputCapture",
    "apply_long_term_fusion",
    "set_long_term_fusion_queries",
    "ReadinessHead",
    "readiness_loss",
]


def __getattr__(name):
    """Lazy access to ``StreamReadyModule`` to avoid a registry import cycle."""
    if name == "StreamReadyModule":
        from lavis.models.streamready.module import StreamReadyModule

        return StreamReadyModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
