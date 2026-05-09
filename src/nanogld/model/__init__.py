"""nanoGLDV1 model package — public API."""

from nanogld.model.aecf import AECFMask, aecf_entropy_reg
from nanogld.model.attention import MultiHeadAttention
from nanogld.model.cfa_projector import CFAProjector
from nanogld.model.decomposition import SeriesDecomposition
from nanogld.model.encoder import HybridEncoder
from nanogld.model.film import FiLMConditioner
from nanogld.model.heads import MultiTaskHead
from nanogld.model.model import nanoGLDV1
from nanogld.model.news_fuser import NewsFuser
from nanogld.model.patch_embed import PatchEmbed
from nanogld.model.regime_encoder import REGIME_VECTOR_DIM, RegimeEncoder, compute_regime_vec
from nanogld.model.revin import RevIN
from nanogld.model.rms_norm import RMSNorm
from nanogld.model.rope import apply_partial_rope, precompute_rope_cache
from nanogld.model.slstm import sLSTM, sLSTMCell
from nanogld.model.slstm_block import sLSTMBlock
from nanogld.model.swiglu import SwiGLU, swiglu_hidden_dim
from nanogld.model.transformer_block import TransformerBlock
from nanogld.model.vsn import GRN, VSN

__all__ = [
    "AECFMask",
    "CFAProjector",
    "FiLMConditioner",
    "GRN",
    "HybridEncoder",
    "MultiHeadAttention",
    "MultiTaskHead",
    "NewsFuser",
    "PatchEmbed",
    "REGIME_VECTOR_DIM",
    "RMSNorm",
    "RegimeEncoder",
    "RevIN",
    "SeriesDecomposition",
    "SwiGLU",
    "TransformerBlock",
    "VSN",
    "aecf_entropy_reg",
    "apply_partial_rope",
    "compute_regime_vec",
    "nanoGLDV1",
    "precompute_rope_cache",
    "sLSTM",
    "sLSTMBlock",
    "sLSTMCell",
    "swiglu_hidden_dim",
]
