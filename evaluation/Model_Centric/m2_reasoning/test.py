import json, os
from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor

NAME_OR_PATH = "/mnt/vstor/CSE_CSDS_YXY1421/huggingface_ulab_server/hub/models--inclusionAI--M2-Reasoning/snapshots/bed6ae896e7023762055fa2d9e3e3fcf8508a40b"

# A) 直接读取本地 preprocessor_config.json
pp_cfg_path = os.path.join(NAME_OR_PATH, "preprocessor_config.json")
print("exists preprocessor_config.json:", os.path.exists(pp_cfg_path))
if os.path.exists(pp_cfg_path):
    with open(pp_cfg_path, "r") as f:
        j = json.load(f)
    print("processor_class:", j.get("processor_class"))
    print("image_processor_type:", j.get("image_processor_type"))

# B) 试图加载 AutoProcessor（依赖 __init__.py 导出 + processor_class）
try:
    proc = AutoProcessor.from_pretrained(NAME_OR_PATH, trust_remote_code=True)
    print("AutoProcessor class:", type(proc).__name__)
except Exception as e:
    print("AutoProcessor load failed:", repr(e))

# C) 单独加载 tokenizer / image_processor，验证文件是否完备
try:
    tok = AutoTokenizer.from_pretrained(NAME_OR_PATH, trust_remote_code=True, use_fast=True)
    print("AutoTokenizer ok:", tok.__class__.__name__)
except Exception as e:
    print("AutoTokenizer failed:", repr(e))

try:
    imgp = AutoImageProcessor.from_pretrained(NAME_OR_PATH, trust_remote_code=True)
    print("AutoImageProcessor ok:", imgp.__class__.__name__)
except Exception as e:
    print("AutoImageProcessor failed:", repr(e))