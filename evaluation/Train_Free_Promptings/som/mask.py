import torch
import os
from PIL import Image
import numpy as np
from scipy.ndimage import label

# load seem model
from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
from seem.utils.distributed import init_distributed as init_distributed_seem
from seem.modeling import build_model as build_model_seem
from task_adapter.seem.tasks import inference_seem_pano, inference_seem_interactive

from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES
from task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto

from segment_anything import sam_model_registry
from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto
from task_adapter.sam.tasks.inference_sam_m2m_interactive import inference_sam_m2m_interactive


# ----------------- init -----------------
semsam_cfg = "configs/semantic_sam_only_sa-1b_swinL.yaml"
seem_cfg   = "configs/seem_focall_unicl_lang_v1.yaml"

semsam_ckpt = "./swinl_only_sam_many2many.pth"
sam_ckpt    = "./sam_vit_h_4b8939.pth"
seem_ckpt   = "./seem_focall_v1.pt"

opt_semsam = load_opt_from_config_file(semsam_cfg)
opt_seem   = load_opt_from_config_file(seem_cfg)
opt_seem   = init_distributed_seem(opt_seem)

model_semsam = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()
model_sam    = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
model_seem   = BaseModel_Seem(opt_seem, build_model_seem(opt_seem)).from_pretrained(seem_ckpt).eval().cuda()

with torch.no_grad():
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        model_seem.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            COCO_PANOPTIC_CLASSES + ["background"], is_eval=True
        )

# ----------------- inference -----------------
@torch.no_grad()
def run_inference(image, model_name="seem", mode="Automatic", alpha=0.1, label_mode="Number", anno_mode=["Mask","Mark"]):
    # image: PIL.Image or str (image path)
    if isinstance(image, str):
        image_path = image
        image = Image.open(image_path).convert("RGB")
    # image: PIL.Image

    if label_mode == 'Alphabet':
        label_mode = 'a'
    else:
        label_mode = '1'

    text_size, hole_scale, island_scale = 640, 100, 100
    text, text_part, text_thresh = '', '', '0.0'

    if mode == "Interactive":
        mask = None  
        labeled_array, num_features = label(np.asarray(mask))
        spatial_masks = torch.stack([torch.from_numpy(labeled_array == i+1) for i in range(num_features)])
    else:
        spatial_masks = None

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        if model_name == 'semantic-sam':
            output, mask = inference_semsam_m2m_auto(
                model_semsam, image, [1], text, text_part, text_thresh,
                text_size, hole_scale, island_scale, False, label_mode, alpha, anno_mode
            )
        elif model_name == 'sam':
            if mode == "Automatic":
                output, mask = inference_sam_m2m_auto(model_sam, image, text_size, label_mode, alpha, anno_mode)
            else:
                output, mask = inference_sam_m2m_interactive(model_sam, image, spatial_masks, text_size, label_mode, alpha, anno_mode)
        elif model_name == 'seem':
            if mode == "Automatic":
                output, mask = inference_seem_pano(model_seem, image, text_size, label_mode, alpha, anno_mode)
            else:
                output, mask = inference_seem_interactive(model_seem, image, spatial_masks, text_size, label_mode, alpha, anno_mode)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    return output, mask



# ----------------- sample -----------------
if __name__ == "__main__":
    #should save the data from huggingface to local path first
    image_folder = "data"
    save_root = "output"

    for bench_name in os.listdir(image_folder):
        print("Processing benchmark:", bench_name, flush=True)

        bench_dir = os.path.join(image_folder, bench_name)
        save_dir = os.path.join(save_root, bench_name)
        os.makedirs(save_dir, exist_ok=True)

        for img_path in os.listdir(bench_dir):
            if img_path.endswith("_output.png"):
                # delete output file
                try:
                    os.remove(os.path.join(bench_dir, img_path))
                except Exception as e:
                    print(f"⚠️ Could not delete {img_path}: {e}", flush=True)
                continue

            try:
                output, mask = run_inference(
                    os.path.join(bench_dir, img_path),
                    model_name="seem",
                    mode="Automatic"
                )

                if mask is None:
                    print(f"⚠️ No mask predicted for {img_path}, skipping.", flush=True)
                    continue

                save_path = os.path.join(save_dir, img_path)

                if isinstance(output, np.ndarray):
                    img = Image.fromarray(output.astype("uint8"))
                    img.save(save_path)
                else:
                    output.save(save_path)

                print(f"✅ Saved {img_path} → {save_path}", flush=True)

            except Exception as e:
                print(f"❌ Failed on {img_path}: {e}", flush=True)

        print("bench_name:", bench_name, "✅ Finished", flush=True)


