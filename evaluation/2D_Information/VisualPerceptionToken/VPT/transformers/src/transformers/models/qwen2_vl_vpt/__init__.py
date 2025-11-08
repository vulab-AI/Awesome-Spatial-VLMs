_import_structure = {
    "configuration_qwen2_vl_vpt": ["VPT_Qwen2VLConfig"],
    "processing_qwen2_vl_vpt": ["VPT_Qwen2VLProcessor"],
    "modeling_qwen2_vl_vpt": ["VPT_Qwen2VLForConditionalGeneration"],
}

from .modeling_qwen2_vl_vpt import (
    VPT_Qwen2VLConfig,
    VPT_Qwen2VLProcessor,
    VPT_Qwen2VLForConditionalGeneration,
)