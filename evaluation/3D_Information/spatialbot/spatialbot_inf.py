import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
import numpy as np

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

# set device
assert torch.cuda.is_available(), "CUDA not available!"
device = torch.device("cuda")
print("✅ CUDA available on:", torch.cuda.get_device_name(device))

model_name = 'RussRobin/SpatialBot-3B'
offset_bos = 0

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.to(device).eval()

# Force-load lazy-initialized submodules and move them to CUDA
base = model.get_model() if hasattr(model, "get_model") else model

# Dummy image to trigger init if lazy
dummy_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
_ = model.process_images([dummy_image, dummy_image], model.config)

# Move mm_projector
if hasattr(base, "mm_projector") and base.mm_projector is not None:
    base.mm_projector.to(device=device, dtype=model.dtype)

# Move vision_tower
if hasattr(base, "vision_tower") and base.vision_tower is not None:
    vt = base.vision_tower[0] if isinstance(base.vision_tower, (list, tuple)) else base.vision_tower
    if hasattr(vt, "to"):
        vt.to(device)
    if hasattr(vt, "vision_tower") and hasattr(vt.vision_tower, "to"):
        vt.vision_tower.to(device)

# Move visual encoder if any
for attr in ("image_encoder", "visual_encoder", "vision_model"):
    if hasattr(base, attr) and getattr(base, attr) is not None:
        enc = getattr(base, attr)
        if hasattr(enc, "to"):
            enc.to(device)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def build_input_with_images(
    prompt: str,
    image_count: int,
    tokenizer,
    device,
    *,
    images_per_group: int = 2,
    line_sep: str = "\n",
    group_sep: str = "\n\n",
    offset_bos: int = 0,
) -> torch.Tensor:
    """
    将若干 <image n> 占位符插入到 USER 段落中，并在文本与图片 token 之间正确拼接。
    假设特殊 token 映射为: <image n> -> token_id = -(200 + n)
    例如: <image 1> -> -201, <image 2> -> -202, ...

    参数:
        prompt: 文本提示
        image_count: 总图片张数 (>= 0)
        tokenizer: 分词器，需有 .__call__ 返回含 input_ids
        device: torch device
        images_per_group: 每组多少张图片（默认 2）
        line_sep: 组内占位符之间的分隔符
        group_sep: 组与组之间的分隔符
        offset_bos: 对后半段 token 的起始偏移（用于跳过 BOS 等）

    返回:
        shape = (1, seq_len) 的 LongTensor（已放到 device）
    """

    assert image_count >= 0, "image_count 必须 >= 0"
    assert images_per_group >= 1, "images_per_group 必须 >= 1"

    # 1) 构造占位符文本：<image 1> ... <image image_count>
    placeholders = [f"<image {i}>" for i in range(1, image_count + 1)]

    # 根据 images_per_group 分组并拼接文本
    grouped_text_parts = []
    for start in range(0, image_count, images_per_group):
        group = placeholders[start:start + images_per_group]
        grouped_text_parts.append(line_sep.join(group))
    placeholders_block = group_sep.join(grouped_text_parts) if grouped_text_parts else ""

    # 2) 拼装完整对话文本（仅在 USER 段里放占位符）
    #    注意：如果没有图片，placeholders_block 为空字符串，也能正常工作
    header = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    )
    user_prefix = "USER: "
    assistant_prefix = " ASSISTANT:"

    # 3) 为了在文本与图片 token 之间精确拼接，我们把占位符整体作为 split 标记
    #    文本结构: [header + user_prefix] + placeholders_block + [\n + prompt + assistant_prefix]
    before = header + user_prefix
    after = (("\n" if placeholders_block else "") + f"{prompt}{assistant_prefix}")

    # 按占位符块切分（会得到 [before, after] 两段）
    # 若没有图片，占位符块为空，则我们直接把图片 token 插在 before 和 after 之间即可
    text_chunks = [tokenizer(before).input_ids, tokenizer(after).input_ids]

    # 4) 生成与 <image n> 对应的特殊 token 序列
    #    映射: n -> -(200 + n)
    image_special_tokens = [-(200 + n) for n in range(1, image_count + 1)]

    # 5) 拼接 input_ids
    input_ids = []
    input_ids += text_chunks[0]
    input_ids += image_special_tokens
    input_ids += text_chunks[1][offset_bos:]  # 可选跳过 BOS

    # 6) 转成张量
    return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)


# def inference(prompt,image_list,depth_map_list):
#     image_num = len(image_list)
#     #get image and depth_map text prompt
#     image_text = "\n\n".join(
#     [f"<image {i}>\n<image {i+1}>" for i in range(1, 2*image_num, 2)]
# )
#     depth_list=[]
#     for depth_map in depth_map_list:
#         if len(depth_map.getbands()) == 1:
#             img = np.array(depth_map)
#             h, w = img.shape
#             rgb_depth = np.zeros((h, w, 3), dtype=np.uint8)
#             rgb_depth[:, :, 0] = (img // 1024) * 4
#             rgb_depth[:, :, 1] = (img // 32) * 8
#             rgb_depth[:, :, 2] = (img % 32) * 8
#             depth_map = Image.fromarray(rgb_depth, 'RGB')
#         depth_list.append(depth_map)

#     image_pairs_list=[[image_list[i], depth_list[i]] for i in range(image_num)]
   




# # Prompt
prompt = 'What is the depth value of point <0.5,0.2>? Answer directly from depth map.'
text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:"
text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]
input_ids = torch.tensor(
    text_chunks[0] + [-201] + [-202] + text_chunks[1][offset_bos:],
    dtype=torch.long
).unsqueeze(0).to(device)

# Load images
image1 = Image.open("rgb.jpg")
image2 = Image.open("depth.png")
print("Image sizes:", image1.size, image2.size)

# Convert grayscale depth to RGB
if len(image2.getbands()) == 1:
    img = np.array(image2)
    h, w = img.shape
    rgb_depth = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_depth[:, :, 0] = (img // 1024) * 4
    rgb_depth[:, :, 1] = (img // 32) * 8
    rgb_depth[:, :, 2] = (img % 32) * 8
    image2 = Image.fromarray(rgb_depth, 'RGB')

# Preprocess images
image_tensor = model.process_images([image1, image2], model.config)
image_tensor = image_tensor.to(dtype=model.dtype, device=device)

# Confirm device states
print("📦 input_ids    :", input_ids.device)
print("🖼️  image_tensor :", image_tensor.device)
print("🔧 model         :", next(model.parameters()).device)
if hasattr(base, "mm_projector"):
    print("🧠 mm_projector  :", next(base.mm_projector.parameters()).device)

# Generate output
with torch.inference_mode():
    output_ids = model.generate(
        input_ids=input_ids,
        images=image_tensor,
        max_new_tokens=100,
        use_cache=True,
        repetition_penalty=1.0
    )[0]

# Decode
output_text = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
print("🤖 Output:", output_text)
