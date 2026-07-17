import math
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass

from transformers.cache_utils import DynamicCache, Cache
from transformers.generation import GenerationMixin
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import (
    auto_docstring,
    logging,
    TransformersKwargs
)

from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2_5OmniRotaryEmbedding,
    Qwen2_5OmniAttention,
    Qwen2_5OmniDecoderLayer,
    Qwen2_5OmniPreTrainedModelForConditionalGeneration,
    Qwen2_5OmniAudioEncoder,
    Qwen2_5OmniVisionEncoder,
    Qwen2_5OmniThinkerTextModel,
    Qwen2_5OmniThinkerCausalLMOutputWithPast,
    Qwen2_5OmniForConditionalGeneration,
    rotate_half,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS
)

from qwen2_5_omni_compress.configuration_qwen2_5_omni_compress import (
    Qwen2_5OmniConfig,
    Qwen2_5OmniCompressTextConfig,
    Qwen2_5OmniCompressThinkerConfig,
    Qwen2_5OmniCompressConfig
)

logger = logging.get_logger(__name__)
# These runtime knobs are echoed at import so the active configuration is
# visible (they materially change the model's behavior). Override via env vars.
softmax_type = os.getenv('SOFTMAX', 'vanilla').lower()
retain_token = os.getenv('RETAIN_TOKEN', 'AV').lower()
mrope_section_str = os.getenv('MROPE_SECTION', '18,18,18,10')
mrope_section_os = [int(section.strip()) for section in mrope_section_str.split(',')]
print("=" * 80, flush=True)
print("EchoingPixels runtime configuration:", flush=True)
print(f"  softmax_type  = {softmax_type}", flush=True)
print(f"  retain_token  = {retain_token}", flush=True)
print(f"  mrope_section = {mrope_section_os}", flush=True)
print("=" * 80, flush=True)


@dataclass
class BaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    labels: Optional[torch.LongTensor] = None

class Qwen2_5OmniRotaryEmbedding_add_time(Qwen2_5OmniRotaryEmbedding):
    def __init__(self, config: Qwen2_5OmniCompressThinkerConfig, device=None):
        super().__init__(
            config=config,
            device=device
            )

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen2_5Omni has different position ids for the grids
        # So we expand the inv_freq to shape (4, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(4, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (4, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def apply_multimodal_rotary_pos_emb_add_time(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    # mrope_section = [16, 16, 16, 16]
    # mrope_section = [16, 20, 20, 8]
    # mrope_section = [20, 20, 20, 4]
    mrope_section = mrope_section_os
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 4] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 4] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen2_5OmniAttention_add_time(Qwen2_5OmniAttention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2_5OmniConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.rotary_emb = Qwen2_5OmniRotaryEmbedding_add_time(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb_add_time(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            # Do NOT pass position_ids to the attention interface. Rotary is already
            # applied above, so FA2 doesn't need them for rotation. FA2's only other use
            # of position_ids is varlen segmentation (it splits the sequence at every
            # discontinuity). After compression our mrope position_ids[0] is non-contiguous
            # (video temporal positions have gaps), which would make FA2 wrongly segment
            # one logical sequence into many and corrupt the attention (NaN -> OOB tokens).
            # Passing None forces FA2 onto the pure-causal `flash_fn` path (the `else`
            # branch in _flash_attention_forward), which is correct here. sdpa/eager
            # ignore position_ids for masking anyway, so this is safe for all backends.
            position_ids=None,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class Qwen2_5OmniDecoderLayer_add_time(Qwen2_5OmniDecoderLayer):
    def __init__(self, config: Qwen2_5OmniCompressTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen2_5OmniAttention_add_time(config, layer_idx)

# -----------------------------------------------------------
# Bidirectional Transformer Encoder
# -----------------------------------------------------------

class BiModalRefinerLayer(nn.Module):
    """
    Bimodal refiner layer processing audio and visual features.
    Fuses audio and visual features and refines them through self-attention.
    """
    def __init__(self, config: Qwen2_5OmniCompressTextConfig, layer_idx: int):
        super().__init__()
        # Flash-Attention easily supports causal masks, and needs lenth info for bidirectional attention
        noncausal_cfg = copy.deepcopy(config)
        if noncausal_cfg._attn_implementation == "flash_attention_2":
            noncausal_cfg._attn_implementation = "sdpa"
        self.hidden_size = noncausal_cfg.hidden_size

        if noncausal_cfg.use_sliding_window and noncausal_cfg._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{noncausal_cfg._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = Qwen2_5OmniAttention_add_time(noncausal_cfg, layer_idx)
        self.self_attn.is_causal = False

        self.mlp = Qwen2MLP(noncausal_cfg)
        self.input_layernorm = Qwen2RMSNorm(noncausal_cfg.hidden_size, eps=noncausal_cfg.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(noncausal_cfg.hidden_size, eps=noncausal_cfg.rms_norm_eps)
        self.attention_type = noncausal_cfg.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class BiModalRefiner(nn.Module):
    """
    Stacked bidirectional refiner layers.
    """
    _no_split_modules = ["BiModalRefinerLayer"]
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [BiModalRefinerLayer(config, i) for i in range(config.av_refiner_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
        ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return hidden_states
    
# -----------------------------------------------------------
# Token selection / compression module
# -----------------------------------------------------------


class TopKTokenCompressor(nn.Module):
    def __init__(self, hidden_size: int, compression_ratio: float = 0.2, tau: float = 1.0,
        image_token_id: int = 151655, video_token_id: int = 151656, audio_token_id: int = 151646):
        """
        Differentiable top-k token compression via Gumbel-Softmax with a straight-through estimator (STE).

        Args:
            score_head (nn.Module): Linear layer or MLP that scores each token.
            ln (nn.Module): LayerNorm applied before scoring.
            compression_ratio (float): Retention ratio for AV tokens.
            tau (float): Gumbel-Softmax temperature. Smaller values make the selection closer to hard selection.
        """
        super().__init__()
        self.ln = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1)
        )

        self.compression_ratio = compression_ratio
        self.tau = tau
        # special token IDs
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id

    def forward(
        self,
        hidden_states: torch.Tensor,                   # (B, L, D)
        inputs_embeds: torch.Tensor,                     # (B, L, D)
        input_ids: torch.LongTensor,                   # (B, L) - MODIFIED: this argument is now essential
        position_ids: torch.LongTensor,               # (4, B, L)
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # cos/sin: 2*(4,B,L,Hd)
        attention_mask: torch.Tensor,       # (B, L)
    ):
        B, L, D = hidden_states.shape
        device = hidden_states.device
        Hd = position_embeddings[0].size(-1)          # rotary dim

        # ------------- 1. Build modality-specific masks and determine compression / retention ranges -------------
        # MODIFICATION START
        # Generate video and audio masks dynamically from input_ids, replacing the former av_mask
        is_video_token = (input_ids == self.video_token_id) | (input_ids == self.image_token_id)
        is_audio_token = (input_ids == self.audio_token_id)
        av_token = is_video_token | is_audio_token

        # Compression targets: video tokens only, or audio+video tokens
        if retain_token == 'v':
            compression_candidate_mask = is_video_token
        elif retain_token == 'av':
            compression_candidate_mask = is_video_token | is_audio_token
        else:
            raise NotImplementedError(f'retain_token {retain_token} not supported')
        # Always retained: text tokens
        always_keep_mask = ~av_token
        # MODIFICATION END

        # ------------- 2. Score only valid (non-pad) video positions -------------
        valid_mask = attention_mask.bool()          # (B, L), marks non-padding positions

        # MODIFIED: renamed av_valid to video_valid
        # video_valid marks all valid video tokens eligible for compression
        video_valid_mask = compression_candidate_mask & valid_mask  # (B, L)

        scores = hidden_states.new_full((B, L), -torch.inf)
        # Compute scores only for video tokens
        video_scores = self.score_head(self.ln(hidden_states)).squeeze(-1)
        # Fill the computed scores into the video-token positions
        scores = torch.where(video_valid_mask, video_scores, scores)

        # ------------- 3. Per-sample top-k video selection (with Gumbel-Softmax and STE) -------------
        keep_indices_list = []
        keep_h, keep_e, keep_pid, keep_cos, keep_sin = [], [], [], [], []

        for b in range(B):
            # Scores of valid video tokens for the current sample
            sample_video_scores = scores[b][video_valid_mask[b]] # (video_num,)
            video_num = sample_video_scores.shape[0]
            av_num = av_token[b].sum().item()

            if video_num == 0:
                # No video tokens: nothing to compress
                topk_idx_in_video = torch.tensor([], device=device, dtype=torch.long)
                video_selection_ste = torch.tensor([], device=device, dtype=hidden_states.dtype)
            else:
                k = max(1, math.ceil(av_num * self.compression_ratio))
                k = min(av_num, k) # ensure k does not exceed the actual number of video tokens

                # --- Gumbel-Softmax STE start ---
                if softmax_type == 'gumbel':
                    y_soft = F.gumbel_softmax(sample_video_scores.unsqueeze(0), tau=self.tau, hard=False).squeeze(0)
                elif softmax_type == 'vanilla':
                    y_soft = F.softmax(sample_video_scores / self.tau, dim=-1)
                else:
                    raise NotImplementedError

                _, topk_idx_in_video = torch.topk(y_soft, k)
                y_hard = torch.zeros_like(sample_video_scores, device=device).scatter_(
                    0, topk_idx_in_video, 1.0
                )
                video_selection_ste = y_hard - y_soft.detach() + y_soft
                # --- Gumbel-Softmax STE end ---

            # --- Apply the differentiable mask to hidden_states and inputs_embeds ---
            full_selection_vector = hidden_states.new_zeros(L)

            # MODIFIED: text tokens are always retained
            # always_keep_mask already includes text
            current_always_keep_mask = always_keep_mask[b] & valid_mask[b]
            full_selection_vector[current_always_keep_mask] = 1.0

            # Place the differentiable video selection back into the full-sequence mask
            if video_num > 0:
                full_selection_vector[video_valid_mask[b]] = video_selection_ste

            # Weight hidden_states and inputs_embeds with the differentiable mask
            h_weighted = hidden_states[b] * full_selection_vector.unsqueeze(-1)

            # --- Forward pass: still use the hard selection to compress the sequence ---
            # Determine, from y_hard, the indices in the original sequence (L) of the retained video tokens
            if video_num > 0:
                original_video_indices = torch.nonzero(video_valid_mask[b], as_tuple=False).squeeze(-1)
                selected_video_indices = original_video_indices[topk_idx_in_video]
            else:
                selected_video_indices = torch.tensor([], device=device, dtype=torch.long)

            # MODIFIED: merge always-retained (text) indices with the selected video-token indices
            always_keep_indices = torch.nonzero(current_always_keep_mask, as_tuple=False).squeeze(-1)
            keep_idx = torch.cat([always_keep_indices, selected_video_indices]).sort().values

            # ------- Gather -------
            # Gather from the weighted tensors to preserve the gradient flow
            keep_indices_list.append(keep_idx)
            keep_h.append(h_weighted.index_select(0, keep_idx))
            keep_e.append(inputs_embeds[b].index_select(0, keep_idx))
            keep_pid.append(position_ids[:, b].index_select(1, keep_idx))
            keep_cos.append(position_embeddings[0][:, b].index_select(1, keep_idx))
            keep_sin.append(position_embeddings[1][:, b].index_select(1, keep_idx))

        # ------------- 4. Reassemble batched data (padding and batching) -------------
        max_len = max(x.size(0) for x in keep_h) if keep_h else 0
        
        new_hid  = hidden_states.new_zeros((B, max_len, D))
        new_emb  = inputs_embeds.new_zeros((B, max_len, D))
        new_attn = hidden_states.new_zeros((B, max_len), dtype=attention_mask.dtype)
        new_pid  = position_ids.new_zeros((4, B, max_len))
        new_cos  = position_embeddings[0].new_zeros((4, B, max_len, Hd))
        new_sin  = position_embeddings[1].new_zeros((4, B, max_len, Hd))

        for b in range(B):
            l = keep_h[b].size(0)
            if l > 0:
                new_hid[b, :l]       = keep_h[b]
                new_emb[b, :l]       = keep_e[b]
                new_attn[b, :l]      = 1
                new_pid[:, b, :l]    = keep_pid[b]
                new_cos[:, b, :l, :] = keep_cos[b]
                new_sin[:, b, :l, :] = keep_sin[b]

        new_pembs = (new_cos.contiguous(), new_sin.contiguous())
        
        return (
            new_hid.contiguous(),
            new_emb.contiguous(),
            new_pid.contiguous(),
            new_pembs,
            new_attn.contiguous(),
            keep_indices_list,
        )


# -----------------------------------------------------------
# Compress Thinker Text Model
# -----------------------------------------------------------
class Qwen2_5OmniCompressThinkerTextModel(Qwen2_5OmniThinkerTextModel):
    config_class = Qwen2_5OmniCompressTextConfig
    _no_split_modules = ["Qwen2_5OmniDecoderLayer_add_time"]
    """
    Steps:
      (i)  Keep all parent members (embeddings / rotary etc.).
      (ii) Insert `BiDirectionalEncoder` before the causal decoder layers;
      (iii) Call `TopKTokenCompressor` to compress the AV tokens;
      (iv) Feed the compressed sequence into the original `self.layers` (causal).
    """
    def __init__(self,
                 config: Qwen2_5OmniCompressTextConfig,
                 image_token_id,
                 video_token_id,
                 audio_token_id
                 ):
        super().__init__(config)

        # new components
        self.rotary_emb = Qwen2_5OmniRotaryEmbedding_add_time(config=config)
        self.layers = nn.ModuleList(
            [Qwen2_5OmniDecoderLayer_add_time(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        if config.av_refiner_layers > 0:
            self.bidir_encoder = BiModalRefiner(config)
        else:
            self.bidir_encoder = None
        self.compressor = TopKTokenCompressor(config.hidden_size, config.compression_ratio, image_token_id=image_token_id, video_token_id=video_token_id, audio_token_id=audio_token_id)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id
        # Initialize weights and apply final processing
        self.post_init()

    def init_bidir_from_decoder(self, share_strategy="one2one"):
        """
        Copy the already-loaded decoder layer weights into bidir_encoder.
        share_strategy:
            1. "one2one":  refiner[i] <- decoder[i]
            2. "last":     all refiner layers copy the last decoder layer
        """
        if not self.bidir_encoder:
            return
        with torch.no_grad():
            n_refiner = len(self.bidir_encoder.layers)
            n_decoder = len(self.layers)

            for i in range(n_refiner):
                if share_strategy == "one2one":
                    src_idx = i if i < n_decoder else n_decoder - 1
                elif share_strategy == "last":
                    src_idx = n_decoder - 1
                else:
                    raise ValueError(f"Unknown share_strategy: {share_strategy}")

                src_layer = self.layers[src_idx]
                tgt_layer = self.bidir_encoder.layers[i]

                # A plain load_state_dict is the simplest approach
                missing, unexpected = tgt_layer.load_state_dict(src_layer.state_dict(), strict=False)
                if len(missing) > 0 or len(unexpected) > 0:
                    logger.warning(
                        f"While copying decoder layer {src_idx} to refiner layer {i}, "
                        f"missing keys: {missing}, unexpected keys: {unexpected}"
                    )
                else:
                    logger.info(f"Refiner layer {i} initialized from decoder layer {src_idx}.")

    # ---------------- override forward ----------------
    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Key change: after self.rotary_emb and before entering the causal decoder, insert
        1) bidir_encoder
        2) compressor
        """
        # --------- 0. Perform the parent class input checks / embedding / positions ----------
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:  # [L]
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `4` is for temporal, height, width and real-time.
        # (4, bs, length)
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        # NOTE: we need to pass text position ids for packing. Qwen2-VL uses 3D positions
        # where each dim indicates visual spatial positions for temporal/height/width grids.
        # There are two scenarios when FA2-like packed masking might be activated.
        # 1. User specifically passed packed `position_ids` and no attention mask.
        #    In this case we expect the useer to create correct position ids for all 3 grids
        #    and prepend text-only position ids to it. The final tensor will be [5, bs, seq-len]
        # 2. User runs forward with no attention mask and no position ids. In this case, position ids
        #    are prepared by the model (`get_rope_index`) as `[5, bs, seq-len]` tensor. Text-only positions are
        #    prepended by us when creating positions so that the mask is constructed correctly. NOTE: failing to pass
        #    text-only positions will cause incorrect mask construction, do not change `prepare_input_for_generation`
        if position_ids.ndim == 3 and position_ids.shape[0] == 5:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }   # [batch_size, num_heads (or broadcast 1), query_seq_len, key_seq_len]
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)  # (4, bs, length, 128)
        hidden_states = inputs_embeds         # (B, L, D)

        if input_ids is not None and input_ids.shape[1] > 1: # prefill stage
            has_mm = (
                (input_ids == self.video_token_id).any() |
                (input_ids == self.audio_token_id).any() |
                (input_ids == self.image_token_id).any()
            )
            # Pure-text training is not yet supported: if any sample in the batch contains audio/video tokens,
            # the compression path is triggered, causing text samples to be re-encoded by the compression module
            if has_mm:
                batch_size, seq_len, _ = inputs_embeds.size()
                # -------- 1. Enter the non-causal bidirectional layers ------------
                if attention_mask is None:
                    attention_mask = inputs_embeds.new_ones((batch_size, seq_len), dtype=torch.long)
                    logger.warning_once("attention_mask is None; defaulting to an all-ones mask.")

                if labels is not None:
                    key_mask = attention_mask.clone()
                    answer_pos = (labels != -100)
                    key_mask[answer_pos] = 0
                else:
                    key_mask = attention_mask.clone()
                # [B, 1, 1, L] broadcasts to [B, head, L_q, L_k]
                # [B, 1, 1, L], passed to bidir_encoder as the key-padding mask

                bidir_mask = key_mask[:, None, None, :].to(dtype=torch.bool)
                if self.bidir_encoder:
                    hidden_states = self.bidir_encoder(
                        hidden_states,
                        attention_mask=bidir_mask,
                        position_ids=text_position_ids,
                        output_attentions=output_attentions,
                        position_embeddings=position_embeddings,
                        **kwargs,
                        )
                    hidden_states = hidden_states * key_mask.unsqueeze(-1).to(hidden_states.device, dtype=hidden_states.dtype) + \
                                    inputs_embeds.to(hidden_states.device, ) * (1 - key_mask).unsqueeze(-1).to(hidden_states.device, dtype=hidden_states.dtype)

                # -------- 2. Compress AV tokens ---------------
                hidden_states, inputs_embeds, position_ids, position_embeddings, attention_mask, keep_indices = self.compressor(
                    hidden_states,
                    inputs_embeds,
                    input_ids,
                    position_ids,
                    position_embeddings,
                    attention_mask,
                    )

                new_labels = None
                if labels is not None:
                    B, _ = labels.shape
                    Lnew = hidden_states.size(1)
                    new_labels = labels.new_full((B, Lnew), -100)
                    for b in range(B):
                        idx = keep_indices[b]
                        new_labels[b, :idx.size(0)] = labels[b, idx]


                # NOTE: skip the length update; use the selected position_ids directly

                # -------- 3. Continue with the parent causal decoder -------
                text_position_ids = position_ids[0]

                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
                )


                if not isinstance(causal_mask_mapping := attention_mask, dict):
                    # Prepare mask arguments
                    mask_kwargs = {
                        "config": self.config,
                        "input_embeds": inputs_embeds,
                        "attention_mask": attention_mask,
                        "cache_position": cache_position,
                        "past_key_values": past_key_values,
                        "position_ids": text_position_ids,
                    }
                    # Create the masks
                    causal_mask_mapping = {
                        "full_attention": create_causal_mask(**mask_kwargs),
                    }
                    # The sliding window alternating layers are not always activated depending on the config
                    if self.has_sliding_layers:
                        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=text_position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if labels is not None:
            # Use new_labels if generated above, otherwise the original labels
            out_labels = new_labels if "new_labels" in locals() else labels
        else:
            out_labels = None

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns, out_labels] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            labels=out_labels,
        )

@auto_docstring(
    custom_intro="""
    The Qwen2.5OmniCompressThinker model which consists of a audio backbone and a language model.
    """
)
class Qwen2_5OmniCompressThinkerForConditionalGeneration(Qwen2_5OmniPreTrainedModelForConditionalGeneration, GenerationMixin):
    config_class = Qwen2_5OmniCompressThinkerConfig
    base_model_prefix = "thinker"
    _tied_weights_keys = ["model.embed_tokens.weight", "lm_head.weight"]
    _no_split_modules = ["Qwen2_5OmniAudioEncoder", "Qwen2_5OmniVisionEncoder"]

    def __init__(self, config: Qwen2_5OmniCompressThinkerConfig):
        super().__init__(config)

        self.audio_tower = Qwen2_5OmniAudioEncoder._from_config(config.audio_config)
        self.visual = Qwen2_5OmniVisionEncoder._from_config(config.vision_config)
        self.vocab_size = config.text_config.vocab_size
        self.model = Qwen2_5OmniCompressThinkerTextModel._from_config(
            config.text_config,
            image_token_id=config.image_token_id,
            video_token_id=config.video_token_id,
            audio_token_id=config.audio_token_id
        )
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.rope_deltas = None
        self.post_init()

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
        audio_seqlens: Optional[torch.LongTensor] = None,
        second_per_grids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            use_audio_in_video (`bool`, *optional*):
                 If set to `True`, use the audio in video.
            audio_seqlens (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
            second_per_grids (`torch.LongTensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        audio_token_id = self.config.audio_token_id
        vision_start_token_id = self.config.vision_start_token_id
        audio_start_token_id = self.config.audio_start_token_id
        position_id_per_seconds = self.config.position_id_per_seconds
        seconds_per_chunk = self.config.seconds_per_chunk

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                4,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_idx, video_idx, audio_idx = 0, 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums, audio_nums = 0, 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                audio_nums = torch.sum(input_ids == audio_start_token_id)
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (
                    (vision_tokens == audio_start_token_id).sum()
                    if use_audio_in_video
                    else (vision_tokens == video_token_id).sum()
                )
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
                multimodal_nums = (
                    image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums
                )
                for _ in range(multimodal_nums):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if audio_token_id in input_tokens and remain_audios > 0:
                        ed_audio = input_tokens.index(audio_token_id, st)
                    else:
                        ed_audio = len(input_tokens) + 1
                    min_ed = min(ed_image, ed_video, ed_audio)
                    if min_ed == ed_audio:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            sequence_rows = torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                            t_row_4d = sequence_rows[0].reshape(1, -1)
                            combined_tensor = torch.cat([sequence_rows, t_row_4d], dim=0)
                            llm_pos_ids_list.append(combined_tensor)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        sequence_rows = torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx
                        t_row_4d = sequence_rows[0].reshape(1, -1)
                        combined_tensor = torch.cat([sequence_rows, t_row_4d], dim=0)
                        llm_pos_ids_list.append(combined_tensor)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                        sequence_rows = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        t_row_4d = sequence_rows[0].reshape(1, -1)
                        combined_tensor = torch.cat([sequence_rows, t_row_4d], dim=0)
                        llm_pos_ids_list.append(combined_tensor)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        sequence_rows = torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx
                        t_row_4d = sequence_rows[0].reshape(1, -1)
                        combined_tensor = torch.cat([sequence_rows, t_row_4d], dim=0)
                        llm_pos_ids_list.append(combined_tensor)

                        st += text_len + bos_len + audio_len + eos_len
                        audio_idx += 1
                        remain_audios -= 1

                    elif min_ed == ed_image:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            sequence_rows = torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                            t_row_4d = sequence_rows[0].reshape(1, -1)
                            combined_tensor = torch.cat([sequence_rows, t_row_4d], dim=0)
                            llm_pos_ids_list.append(combined_tensor)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        sequence_rows = torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx
                        t_row_4d = sequence_rows[0].reshape(1, -1)
                        combined_tensor = torch.cat([sequence_rows, t_row_4d], dim=0)
                        llm_pos_ids_list.append(combined_tensor)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        grid_t = image_grid_thw[image_idx][0]
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]
                        t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).long()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                        sequence_rows = llm_pos_ids
                        t_row_4d = sequence_rows[0].reshape(1, -1)
                        combined_tensor = torch.cat([sequence_rows, t_row_4d], dim=0)
                        llm_pos_ids_list.append(combined_tensor)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        sequence_rows = torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx
                        t_row_4d = sequence_rows[0].reshape(1, -1)
                        combined_tensor = torch.cat([sequence_rows, t_row_4d], dim=0)
                        llm_pos_ids_list.append(combined_tensor)

                        st += text_len + bos_len + image_len + eos_len
                        image_idx += 1
                        remain_images -= 1

                    elif min_ed == ed_video and not use_audio_in_video:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            sequence_rows = torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                            t_row_4d = sequence_rows[0].reshape(1, -1)
                            combined_tensor = torch.cat([sequence_rows, t_row_4d], dim=0)
                            llm_pos_ids_list.append(combined_tensor)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        sequence_rows = torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx
                        t_row_4d = sequence_rows[0].reshape(1, -1)
                        combined_tensor = torch.cat([sequence_rows, t_row_4d], dim=0)
                        llm_pos_ids_list.append(combined_tensor)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]
                        t_index = (
                            torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                        ).long()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                        sequence_rows = llm_pos_ids
                        t_row_4d = sequence_rows[0].reshape(1, -1)
                        combined_tensor = torch.cat([sequence_rows, t_row_4d], dim=0)
                        llm_pos_ids_list.append(combined_tensor)


                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        sequence_rows = torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx
                        t_row_4d = sequence_rows[0].reshape(1, -1)
                        combined_tensor = torch.cat([sequence_rows, t_row_4d], dim=0)
                        llm_pos_ids_list.append(combined_tensor)

                        st += text_len + bos_len + video_len + eos_len
                        video_idx += 1
                        remain_videos -= 1

                    elif min_ed == ed_video and use_audio_in_video:
                        text_len = min_ed - st - 2
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            sequence_rows = torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                            t_row_4d = sequence_rows[0].reshape(1, -1)
                            combined_tensor = torch.cat([sequence_rows, t_row_4d], dim=0)
                            llm_pos_ids_list.append(combined_tensor)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        sequence_rows = torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx
                        t_row_4d = sequence_rows[0].reshape(1, -1)
                        combined_tensor = torch.cat([sequence_rows, t_row_4d], dim=0)
                        llm_pos_ids_list.append(combined_tensor)
                        llm_pos_ids_list.append(combined_tensor)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                        audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]

                        t_index = (
                            torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                        ).long()
                        video_llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )

                        t_ntoken_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                        video_chunk_indexes = self.get_chunked_index(video_llm_pos_ids[0], t_ntoken_per_chunk, st_idx)
                        audio_chunk_indexes = self.get_chunked_index(audio_llm_pos_ids[0], t_ntoken_per_chunk, st_idx)
                        sub_len = 0
                        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                            video_chunk_index = video_chunk_indexes[j] if j < len(video_chunk_indexes) else None
                            audio_chunk_index = audio_chunk_indexes[j] if j < len(audio_chunk_indexes) else None
                            if video_chunk_index is not None:
                                sub_len += video_chunk_index[1] - video_chunk_index[0]

                                sequence_rows = video_llm_pos_ids[:, video_chunk_index[0] : video_chunk_index[1]]
                                t_row_4d = sequence_rows[0].reshape(1, -1)
                                combined_tensor = torch.cat([sequence_rows, t_row_4d], dim=0)
                                llm_pos_ids_list.append(combined_tensor)

                            if audio_chunk_index is not None:
                                sub_len += audio_chunk_index[1] - audio_chunk_index[0]

                                sequence_rows = audio_llm_pos_ids[:, audio_chunk_index[0] : audio_chunk_index[1]]
                                t_row_4d = sequence_rows[0].reshape(1, -1)
                                combined_tensor = torch.cat([sequence_rows, t_row_4d], dim=0)
                                llm_pos_ids_list.append(combined_tensor)

                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        sequence_rows = torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx
                        t_row_4d = sequence_rows[0].reshape(1, -1)
                        combined_tensor = torch.cat([sequence_rows, t_row_4d], dim=0)
                        llm_pos_ids_list.append(combined_tensor)
                        llm_pos_ids_list.append(combined_tensor)

                        st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2

                        audio_idx += 1
                        video_idx += 1
                        remain_videos -= 1
                        remain_audios -= 1

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    sequence_rows = torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    t_row_4d = sequence_rows[0].reshape(1, -1)
                    combined_tensor = torch.cat([sequence_rows, t_row_4d], dim=0)
                    llm_pos_ids_list.append(combined_tensor)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(4, -1)

                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

            return position_ids, mrope_position_deltas
        else:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(4, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

            return position_ids, mrope_position_deltas


    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        return video_embeds

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
    ):
        """
        Encodes audios into continuous embeddings that can be forwarded to the language model.

        Args:
            input_features (`torch.FloatTensor`):
                The tensors corresponding to the input audios.
            feature_attention_mask (`torch.LongTensor`, *optional*):
                Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
            audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
        """
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None

        audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
            audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        )
        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=feature_lens,
            aftercnn_lens=audio_feat_lengths,
        )
        audio_features = audio_outputs.last_hidden_state

        if audio_features.shape[0] != sum(audio_output_lengths.tolist()):
            raise ValueError("length of audio_features should match audio_output_lengths")

        return audio_features

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_audio_in_video: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, Qwen2_5OmniThinkerCausalLMOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        feature_attention_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
            The length of feature shape of each audio in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        use_audio_in_video (`bool`, *optional*):
            Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
        video_second_per_grid (`torch.LongTensor` of shape `(num_videos)`, *optional*):
            Number of seconds per grid for each video, used for temporal feature mapping.

        Example:

        ```python
        >>> from io import BytesIO
        >>> from urllib.request import urlopen
        >>> import librosa
        >>> from qwen_vl_utils import process_vision_info
        >>> from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        >>> thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        >>> processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

        >>> conversations = [
        >>>         {'role': 'system', 'content': 'You are a helpful voice chat bot, and please respond to me in a casual conversation manner using random voice.'},
        >>>         {"role": "user", "content": [
        >>>             {"type": "image", "image_url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
        >>>             {"type": "audio", "audio_url": "https://example.com/sample.mp3"},
        >>>         ]},
        >>> ]

        >>> text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        >>> audios = [ librosa.load(BytesIO(urlopen( conversations[1]['content'][1]['audio_url'] ).read()), sr=self.processor.feature_extractor.sampling_rate) ]
        >>> images, videos = process_vision_info(conversations)
        >>> inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True)

        >>> # Generate
        >>> inputs['use_audio_in_video'] = `True` or `False`
        >>> generation = thinker.generate(**inputs, max_new_tokens=2048)
        >>> generate_ids = generation[:, inputs.input_ids.size(1):]

        >>> response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text , audios , image and video
        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            if input_ids is None:
                audio_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                audio_mask = audio_mask.all(-1)
            else:
                audio_mask = input_ids == self.config.audio_token_id
            audio_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            if input_ids is None:
                image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                image_mask = image_mask.all(-1)
            else:
                image_mask = input_ids == self.config.image_token_id

            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            if input_ids is None:
                video_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                video_mask = video_mask.all(-1)
            else:
                video_mask = input_ids == self.config.video_token_id

            video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(4, -1, -1)

        outputs = self.model(
            # CompressThinkerTextModel needs input_ids to detect the compressible prefill stage
            input_ids=input_ids if input_ids is not None and (input_features is not None or pixel_values is not None or pixel_values_videos is not None) else None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            labels=labels,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if not return_dict:
                compressed_labels = outputs[4] if len(outputs) > 4 else None
            else:
                compressed_labels = getattr(outputs, "labels", None)
            assert compressed_labels is not None, "Compressed labels should not be None"
            labels = compressed_labels
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size
            )

        if not return_dict:
            output = (logits,) + outputs
            return (loss,) + output if loss is not None else output

        return Qwen2_5OmniThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_features=None,
        feature_attention_mask=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        # GenerationMixin.prepare_inputs_for_generation computes cu_seq_lens_q/k
        # and max_length_q/k from position_ids when attn_implementation contains
        # "flash". These cumulative sequence lengths are based on the ORIGINAL
        # (uncompressed) sequence, but our compressor shortens the sequence inside
        # the text model's forward. Passing stale cu_seq_lens forces FA2 onto the
        # flash_varlen_fn path with wrong segment boundaries → out-of-bounds
        # access → NaN → OOB token → device-side assert crash.
        # Removing them makes FA2 fall through to the simple flash_fn (causal)
        # path, which is correct for single-sequence (non-packed) inference.
        for k in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"):
            model_inputs.pop(k, None)

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["input_features"] = None

        return model_inputs
    

class Qwen2_5OmniCompressForConditionalGeneration(Qwen2_5OmniForConditionalGeneration):
    config_class = Qwen2_5OmniCompressConfig

    def __init__(self, config):
        super().__init__(config)

        if "thinker" in self._modules:
            del self._modules["thinker"]

        self.thinker = Qwen2_5OmniCompressThinkerForConditionalGeneration(
            config.thinker_config
        )

        # Re-run post_init; super().__init__() already ran it once, but this also
        # initializes the newly replaced compress thinker.
        self.post_init()

    @torch.no_grad()
    # TODO: raushan, defaults should be saved in generation config
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        speaker: str = "Chelsie",
        use_audio_in_video: bool = False,
        return_audio: Optional[bool] = None,
        thinker_max_new_tokens: int = 1024,
        talker_max_new_tokens: int = 4096,
        talker_do_sample: bool = True,
        talker_top_k: int = 40,
        talker_top_p: float = 0.8,
        talker_temperature: float = 0.9,
        talker_eos_token_id: list[int] = [8292, 8294],
        talker_repetition_penalty: float = 1.05,
        **kwargs,
    ):    
        r"""
        Generate text response and audio from input.

        Args:
            input_ids (`Optional[torch.Tensor]`, *optional*):
                Input ids, should obtain from processor.
            speaker (`str` , defaults to "Chelsie"):
                Which speaker should be used in audio response.
            use_audio_in_video (`bool`, defaults to False):
                Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
            return_audio (`Optional[bool]`, *optional*):
                Whether or not return response in audio format. When `return_audio=None`, this parameter is same as `config.enable_audio_output`.
            kwargs (*optional*):
                - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model.
                - With a *thinker_*, *talker_*, *token2wav_* prefix, they will be input for the `generate` method of the
                thinker, talker and token2wav respectively. It has the priority over the keywords without a prefix.
        Returns:
            When `return_audio=False`:
                - **Text** (`torch.Tensor`): Generated text token sequence.
            When `return_audio=True`:
                - **Text** (`torch.Tensor`): Generated text token sequence.
                - **Audio waveform** (`torch.Tensor`): Generated audio waveform.
        """
        if return_audio and not self.has_talker:
            raise ValueError(
                "Cannot use talker when talker module not initialized. Use `enable_talker` method or set enable_talker in config to enable talker."
            )
        if return_audio is None:
            return_audio = self.has_talker
        if input_ids.shape[0] != 1 and return_audio:
            raise NotImplementedError("Qwen2.5-Omni currently does not support batched inference with audio output")

        shared_kwargs = {"use_audio_in_video": use_audio_in_video}
        thinker_kwargs = {
            "max_new_tokens": thinker_max_new_tokens,
        }
        talker_kwargs = {
            "max_new_tokens": talker_max_new_tokens,
            "do_sample": talker_do_sample,
            "top_k": talker_top_k,
            "top_p": talker_top_p,
            "temperature": talker_temperature,
            "eos_token_id": talker_eos_token_id,
            "repetition_penalty": talker_repetition_penalty,
        }
        token2wav_kwargs = {}

        for key, value in kwargs.items():
            if key.startswith("thinker_"):
                thinker_kwargs[key[len("thinker_") :]] = value
            elif key.startswith("talker_"):
                talker_kwargs[key[len("talker_") :]] = value
            elif key.startswith("token2wav_"):
                token2wav_kwargs[key[len("token2wav_") :]] = value
            # Process special input values
            elif key == "feature_attention_mask":
                thinker_kwargs[key] = value
                talker_kwargs["audio_feature_lengths"] = torch.sum(value, dim=1)
            elif key == "input_features" or key == "attention_mask":
                thinker_kwargs[key] = value
            # Put other key to shared kwargs
            else:
                shared_kwargs[key] = value

        # Merge kwargs
        for key, value in shared_kwargs.items():
            if key not in thinker_kwargs:
                thinker_kwargs[key] = value
            if key not in talker_kwargs:
                talker_kwargs[key] = value
            if key not in token2wav_kwargs:
                token2wav_kwargs[key] = value

        # 1. Generate from thinker module
        generate_audio = return_audio and self.has_talker
        if generate_audio:
            thinker_kwargs["output_hidden_states"] = True
            thinker_kwargs["return_dict_in_generate"] = True

        thinker_result = self.thinker.generate(input_ids=input_ids, **thinker_kwargs)

        if not generate_audio:
            return thinker_result

        return thinker_result.sequences, None