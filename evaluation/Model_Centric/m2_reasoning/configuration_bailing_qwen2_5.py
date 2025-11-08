# coding=utf-8
# Copyright 2024 ANT Group and the HuggingFace Inc. team. All rights reserved.
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

import os
from typing import Union

from configuration_qwen2_5_vit import Qwen2_5_VLVisionConfig
# from .configuration_qwen2_3d import Qwen2_3dConfig
from configuration_qwen2 import Qwen2Config

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import (
    logging,
)
from transformers.models.auto import CONFIG_MAPPING

logger = logging.get_logger(__name__)

def init_mm_special_tokens():
# additional_special_tokens_qwen2 = [
#     "[item]",151665
#     "<html>",151666
#     "</html>",151667
#     "<body>",151668
#     "</body>",151669
#     "<table>",151670
#     "</table>",151671
#     "<tr>",151672
#     "</tr>",151673
#     "<td>",151674
#     "</td>",151675
#     "<think>",151676
#     "</think>",151677
# ]
# https://code.alipay.com/multimodal/antmmf/blob/refact_clean_qwen2_5_vit/bailingmm/models/blip2_models/bailing_mm_interleave.py#L1465
# DEFAULT_IMAGE_PATCH_TOKEN = "<imagePatch>",151678
# DEFAULT_IM_START_TOKEN = "<image>",151679
# DEFAULT_IM_END_TOKEN = "</image>",151680
# DEFAULT_VID_START_TOKEN = "<video>",151681
# DEFAULT_VID_END_TOKEN = "</video>",151682
# DEFAULT_GEN_IMAGE_PATCH_TOKEN = "<gen_imagePatch>",151683
# DEFAULT_GEN_IM_START_TOKEN = "<gen_image>",151684
# DEFAULT_GEN_IM_END_TOKEN = "</gen_image>",151685
# PLACEHOLDER_IMAGE_TOKEN_IN_TEXT = "<imageHere>",151686
# DEFAULT_END_OF_CHUNK_TOKEN = "<end_of_chunk>",151687
# DEFAULT_END_OF_AUDIO_TOKEN = "<end_of_audio>",151688
# DEFAULT_AUDIO_PATCH_TOKEN = "<audioPatch>",151689
# DEFAULT_AU_START_TOKEN = "<audio>",151690
# DEFAULT_AU_END_TOKEN = "</audio>",151691
# DEFAULT_GEN_AUDIO_PATCH_TOKEN = "<gen_audioPatch>",151692
# DEFAULT_GEN_AU_START_TOKEN = "<gen_audio>",151693
# DEFAULT_GEN_AU_END_TOKEN = "</gen_audio>",151694
# PLACEHOLDER_AUDIO_TOKEN_IN_TEXT = "<audioHere>",151695
# DEFAULT_FRAME_PATCH_TOKEN = "<framePatch>",15196

    special_tokens = {
        "[item]": 151665,
        "<html>": 151666,
        "</html>": 151667,
        "<body>": 151668,
        "</body>": 151669,
        "<table>": 151670,
        "</table>": 151671,
        "<tr>": 151672,
        "</tr>": 151673,
        "<td>": 151674,
        "</td>": 151675,
        "<think>": 151676,
        "</think>": 151677,

        "<imagePatch>": 151678,
        "<image>": 151679,
        "</image>": 151680,
        "<video>": 151681,
        "</video>": 151682,
        "<gen_imagePatch>": 151683,
        "<gen_image>": 151684,
        "</gen_image>": 151685,
        "<imageHere>": 151686,
        "<end_of_chunk>": 151687,
        "<end_of_audio>": 151688,
        "<audioPatch>": 151689,
        "<audio>": 151690,
        "</audio>": 151691,
        "<gen_audioPatch>": 151692,
        "<gen_audio>": 151693,
        "</gen_audio>": 151694,
        "<audioHere>": 151695,
        "<framePatch>": 15196,
    }
    return special_tokens

class Bailing_qwen2_5VisionConfig(PretrainedConfig):
    model_type = "bailing_qwen2_5"

    def __init__(
        self,
        vision_type="qwen2_5_vit",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_type = vision_type

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "bailing_qwen2_5":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)

class Bailing_qwen2_5LlmConfig(PretrainedConfig):
    model_type = "bailing_qwen2_5"

    def __init__(
        self,
        llm_type="qwen2_5",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_type = llm_type

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "bailing_qwen2_5":
            config_dict = config_dict["llm_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)

class Bailing_qwen2_5AudioConfig(PretrainedConfig):
    model_type = "bailing_qwen2_5"

    def __init__(
        self,
        audio_encoder_type="sanm",
        audio_input_size=560,
        audio_output_size=512,
        audio_attention_heads=4,
        audio_linear_units=2048,
        audio_num_blocks=50,
        audio_dropout_rate=0.1,
        audio_positional_dropout_rate=0.1,
        audio_attention_dropout_rate=0.1,
        audio_input_layer="pe",
        audio_pos_enc_class="SinusoidalPositionEncoder",
        audio_normalize_before=True,
        audio_kernel_size=11,
        audio_sanm_shfit=0,
        audio_selfattention_layer_type="sanm",
        audio_chunk_size=(12,),
        audio_stride=(8,),
        audio_pad_left=(0,),
        audio_compress=3,
        **kwargs
    ):
        self.audio_encoder_type = audio_encoder_type
        self.audio_input_size = audio_input_size
        self.audio_output_size = audio_output_size
        self.audio_attention_heads = audio_attention_heads
        self.audio_linear_units = audio_linear_units
        self.audio_num_blocks = audio_num_blocks
        self.audio_dropout_rate = audio_dropout_rate
        self.audio_positional_dropout_rate = audio_positional_dropout_rate
        self.audio_attention_dropout_rate = audio_attention_dropout_rate
        self.audio_input_layer = audio_input_layer
        self.audio_pos_enc_class = audio_pos_enc_class
        self.audio_normalize_before = audio_normalize_before
        self.audio_kernel_size = audio_kernel_size
        self.audio_sanm_shfit = audio_sanm_shfit
        self.audio_selfattention_layer_type = audio_selfattention_layer_type
        self.audio_chunk_size = audio_chunk_size
        self.audio_stride = audio_stride
        self.audio_pad_left = audio_pad_left
        self.audio_compress = audio_compress

        if audio_input_layer is None:
            self.audio_input_layer = "pe"
        else:
            self.audio_input_layer = audio_input_layer

        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "bailing_qwen2_5":
            config_dict = config_dict["audio_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)

class Bailing_qwen2_5Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BailingNativeModel`]. It is used to instantiate a
    Bailing model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of

    Configuration objects inherited from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Dict`, *optional*):
            The config for the visual encoder initialization.
        audio_config (`Dict`, *optional*):
            The config for the audio encoder initialization.
        llm_config (`Dict`, *optional*):
            The config for the language model initialization.
    """
    model_type = "bailing_qwen2_5"
    is_composition = True
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vision_config=None,
        audio_config=None,
        llm_config=None,
        use_vit_lora=0,
        use_llm_lora=0,
        mlp_depths=1,
        select_layer=-1,
        num_query_token_image=64,
        num_query_token_video=64,
        num_query_token_audio=256,
        num_decoder_image_token=1024,
        num_decoder_audio_token=512,
        _attn_implementation="flash_attention_2",
        **kwargs,
    ):
        if isinstance(vision_config, dict):
        # if vision_config is not None:
            # print(vision_config)
            if vision_config.get("model_type") == "qwen2_5_vit":
                self.vision_config = Qwen2_5_VLVisionConfig(**vision_config)
                self.vision_type = "qwen2_5_vit"
            else:
                self.vision_config = Qwen2_5_VLVisionConfig(**vision_config)
                self.vision_type = "qwen2_5_vit"

        elif vision_config is None:
            self.vision_config = Qwen2_5_VLVisionConfig()
            self.vision_type = "qwen2_5_vit"

        ## remove audio
        self.audio_config = None
        # if isinstance(audio_config, dict):
        #     self.audio_config = Bailing_qwen2_5AudioConfig(**audio_config)
        # elif audio_config is None:
        #     self.audio_config = Bailing_qwen2_5AudioConfig()

        if isinstance(llm_config, dict):
        # if llm_config is not None:
            # print(llm_config)
            if llm_config.get("model_type") == "qwen2_5_3d":
                self.llm_config = Qwen2Config(**llm_config)
                self.llm_type = "qwen2_5_3d"
            else:
                llm_config["model_type"] = llm_config["model_type"] if "model_type" in llm_config else "qwen2_5"
                # self.llm_config = CONFIG_MAPPING[llm_config["model_type"]](**llm_config)
                self.llm_config = Qwen2Config(**llm_config)
                self.llm_type = "qwen2_5"
        else:
            self.llm_config = Qwen2Config()
            self.llm_type = "qwen2_5_3d"

        self.use_vit_lora = use_vit_lora
        self.use_llm_lora = use_llm_lora

        self.mlp_depths = mlp_depths
        self.select_layer = select_layer
        self.num_query_token_image = num_query_token_image
        self.num_query_token_video = num_query_token_video
        self.num_query_token_audio = num_query_token_audio
        self.num_decoder_image_token = num_decoder_image_token
        self.num_decoder_audio_token = num_decoder_audio_token

        self._attn_implementation = _attn_implementation

        # for key, value in init_mm_special_tokens().items():
        #     kwargs.update({key: value})

        super().__init__(**kwargs)
