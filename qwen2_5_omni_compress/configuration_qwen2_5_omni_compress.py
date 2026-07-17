from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoderConfig,
    Qwen2_5OmniConfig,
    Qwen2_5OmniTalkerConfig,
    Qwen2_5OmniTextConfig,
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniToken2WavConfig,
    Qwen2_5OmniVisionEncoderConfig,
)


logger = logging.get_logger(__name__)



# Custom compression config — fields are unchanged from the base Qwen2.5-Omni
# config, but a dedicated subclass is required for transformers registration.

class Qwen2_5OmniCompressTextConfig(Qwen2_5OmniTextConfig):

    model_type = "qwen2_5_omni_compress_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen25OmniText`
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen25OmniText`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }


    def __init__(
        self,
        vocab_size=152064,
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=32768,
        max_window_layers=28,
        layer_types=None,
        attention_dropout=0.0,
        av_refiner_layers: int = 1,
        compression_ratio: float = 0.2,
        **kwargs,
    ):
        self.av_refiner_layers = av_refiner_layers
        self.compression_ratio = compression_ratio
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            layer_types=layer_types,
            attention_dropout=attention_dropout,
            **kwargs,
        )

class Qwen2_5OmniCompressThinkerConfig(Qwen2_5OmniThinkerConfig):

    model_type = "qwen2_5_omni_compress_thinker"
    attribute_map = {
        "image_token_id": "image_token_index",
        "video_token_id": "video_token_index",
        "audio_token_id": "audio_token_index",
    }
    sub_configs = {
        "audio_config": Qwen2_5OmniAudioEncoderConfig,
        "vision_config": Qwen2_5OmniVisionEncoderConfig,
        "text_config": Qwen2_5OmniTextConfig,
    }

    def __init__(
        self,
        audio_config=None,
        vision_config=None,
        text_config=None,
        audio_token_index=151646,
        image_token_index=151655,
        video_token_index=151656,
        position_id_per_seconds=25,
        seconds_per_chunk=2,
        audio_start_token_id=151647,
        audio_end_token_id=151648,
        user_token_id=872,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(
            audio_config=audio_config,
            vision_config=vision_config,
            text_config=text_config,
            audio_token_index=audio_token_index,
            image_token_index=image_token_index,
            video_token_index=video_token_index,
            position_id_per_seconds=position_id_per_seconds,
            seconds_per_chunk=seconds_per_chunk,
            audio_start_token_id=audio_start_token_id,
            audio_end_token_id=audio_end_token_id,
            user_token_id=user_token_id,
            initializer_range=initializer_range,
            **kwargs,
        )

class Qwen2_5OmniCompressConfig(PretrainedConfig):
    model_type = "qwen2_5_omni_compress"
    sub_configs = {
        "thinker_config": Qwen2_5OmniThinkerConfig,
        "talker_config": Qwen2_5OmniTalkerConfig,
        "token2wav_config": Qwen2_5OmniToken2WavConfig,
    }

    def __init__(
        self,
        thinker_config=None,
        talker_config=None,
        token2wav_config=None,
        enable_audio_output: bool = True,
        **kwargs,
    ):
        if thinker_config is None:
            thinker_config = {}
            logger.info("thinker_config is None. Initializing thinker model with default values")

        if talker_config is None:
            talker_config = {}
            logger.info("talker_config is None. Initializing talker model with default values")

        if token2wav_config is None:
            token2wav_config = {}
            logger.info("token2wav_config is None. Initializing token2wav model with default values")

        self.thinker_config = Qwen2_5OmniCompressThinkerConfig(**thinker_config)
        self.talker_config = Qwen2_5OmniTalkerConfig(**talker_config)
        self.token2wav_config = Qwen2_5OmniToken2WavConfig(**token2wav_config)
        self.enable_audio_output = enable_audio_output

        super().__init__(**kwargs)

    def get_text_config(self, decoder=False) -> "PretrainedConfig":
        """
        Returns the config that is meant to be used with text IO. On most models, it is the original config instance
        itself. On specific composite models, it is under a set of valid names.

        Args:
            decoder (`Optional[bool]`, *optional*, defaults to `False`):
                If set to `True`, then only search for decoder config names.
        """
        # Overridden for deeply nested config like Qwen2-Omni. We don't have any omni model
        # except for Qwen yet. This has to be generalized if more deeply nested configs are
        # added. NOTE: currently method used only by vLLM
        return self.thinker_config.get_text_config()