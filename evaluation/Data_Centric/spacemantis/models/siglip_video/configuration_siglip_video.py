# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Siglip model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import SiglipTextConfig, SiglipVisionConfig
logger = logging.get_logger(__name__)

class SiglipVideoPerceiverConfig(PretrainedConfig):
    r"""
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the perceiver block.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        resampler_n_latents (`int`, *optional*, defaults to 64):
            Number of latent embeddings to resample ("compress") the input sequence to (usually < 128).
        resampler_depth (`int`, *optional*, defaults to 3):
            Depth of the Perceiver Resampler (Transformer w/ cross attention). Should be shallow (<= 3).
        resampler_n_heads (`int`, *optional*, defaults to 16):
            Number of heads in each Transformer block (for multi-headed self-attention).
        resampler_head_dim (`int`, *optional*, defaults to 96):
            Dimensionality of each head projection in the Transformer block.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            Number of key-value heads in the perceiver attention block.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
    """

    model_type = "siglip_video_perceiver"

    def __init__(
        self,
        hidden_act="silu",
        hidden_size=1152,
        intermediate_size=4304,
        rms_norm_eps=1e-06,
        resampler_n_latents=64,
        resampler_depth=3,
        resampler_n_heads=16,
        resampler_head_dim=96,
        num_key_value_heads=4,
        attention_dropout=0.0,
        max_temporal_clip_size=8,
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.resampler_n_latents = resampler_n_latents
        self.resampler_depth = resampler_depth
        self.resampler_n_heads = resampler_n_heads
        self.num_key_value_heads = num_key_value_heads
        self.resampler_head_dim = resampler_head_dim
        self.attention_dropout = attention_dropout
        self.max_temporal_clip_size = max_temporal_clip_size
        self.layer_norm_eps = layer_norm_eps
        
        if self.num_key_value_heads > self.resampler_n_heads:
            raise ValueError(
                f"num_key_value_heads={self.num_key_value_heads} must be less than or equal to"
                f" resampler_n_heads={self.resampler_n_heads}"
            )
        super().__init__(**kwargs)
    

class SiglipVideoConfig(PretrainedConfig):
    r"""
    [`SiglipConfig`] is the configuration class to store the configuration of a [`SiglipVideoModel`]. It is used to
    instantiate a SiglipVideo model according to the specified arguments, defining the text model and vision model configs.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Siglip
    [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`SiglipTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`SiglipVisionConfig`].
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import SiglipVideoConfig, SiglipVideoModel

    >>> # Initializing a SiglipVideoConfig with ... style configuration
    >>> configuration = SiglipVideoConfig()

    >>> # Initializing a SiglipModel (with random weights) from the google/siglip-base-patch16-224 style configuration
    >>> model = SiglipVideoModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a SiglipConfig from a SiglipTextConfig and a SiglipVisionConfig
    >>> from transformers import SiglipTextConfig, SiglipVisionConfig, SiglipVideoPerceiverConfig

    >>> # Initializing a SiglipText and SiglipVision configuration
    >>> config_text = SiglipTextConfig()
    >>> config_vision = SiglipVisionConfig()
    >>> config_perceiver = SiglipVideoPerceiverConfig()

    >>> config = SiglipVideoConfig.from_text_vision_configs(config_text, config_vision, config_perceiver)
    ```"""

    model_type = "siglip_video"
    sub_configs = {"text_config": SiglipTextConfig, "vision_config": SiglipVisionConfig, "perceiver_config": SiglipVideoPerceiverConfig}

    def __init__(self, text_config=None, vision_config=None, perceiver_config=None, **kwargs):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `SiglipTextConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the `SiglipVisionConfig` with default values.")
        
        if perceiver_config is None:
            perceiver_config = {}
            logger.info("`perceiver_config` is `None`. initializing the `PerceiverConfig` with default values.")

        self.text_config = SiglipTextConfig(**text_config)
        self.vision_config = SiglipVisionConfig(**vision_config)
        self.perceiver_config = SiglipVideoPerceiverConfig(**perceiver_config)

        self.initializer_factor = 1.0

    @classmethod
    def from_configs(cls, text_config: SiglipTextConfig, vision_config: SiglipVisionConfig, perceiver_config: SiglipVideoPerceiverConfig, **kwargs):
        r"""
        Instantiate a [`SiglipConfig`] (or a derived class) from siglip text model configuration and siglip vision
        model configuration.

        Returns:
            [`SiglipConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), perceiver_config=perceiver_config.to_dict(), **kwargs)
