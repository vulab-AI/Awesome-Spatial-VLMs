from .configuration_siglip_video import SiglipVideoConfig, SiglipVideoPerceiverConfig
from .modeling_siglip_video import SiglipVideoModel, SiglipVideoPerceiverResampler

from transformers import AutoModel, AutoConfig

AutoConfig.register("siglip_video", SiglipVideoConfig)
AutoModel.register(SiglipVideoConfig, SiglipVideoModel)