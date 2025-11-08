import argparse
import torch
from vcoder_llava.vcoder_conversation import conv_templates, SeparatorStyle
from vcoder_llava.model.builder import load_pretrained_model
from vcoder_llava.utils import disable_torch_init
from vcoder_llava.mm_utils import process_images, tokenizer_image_token, tokenizer_depth_seg_token, get_model_name_from_path, KeywordsStoppingCriteria
from vcoder_llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    SEG_TOKEN_INDEX, DEFAULT_SEG_TOKEN,
    DEPTH_TOKEN_INDEX, DEFAULT_DEPTH_TOKEN
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, seg_image_processor, depth_image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    conv_mode = "llava_v1"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    roles = conv.roles

    image = load_image(args.image_file)
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, args)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    
    # Segmentation
    seg_image_tensor = None
    if args.seg_file is not None:
        seg_image = load_image(args.seg_file)
        seg_image_tensor = process_images([seg_image], seg_image_processor, args)
        if type(seg_image_tensor) is list:
            seg_image_tensor = [image.to(model.device, dtype=torch.float16) for image in seg_image_tensor]
        else:
            seg_image_tensor = seg_image_tensor.to(model.device, dtype=torch.float16)
    else:
        seg_image = None

    # Depth
    depth_image_tensor = None
    if args.depth_file is not None:
        depth_image = load_image(args.depth_file)
        depth_image_tensor = process_images([depth_image], depth_image_processor, args)
        if type(depth_image_tensor) is list:
            depth_image_tensor = [image.to(model.device, dtype=torch.float16) for image in depth_image_tensor]
        else:
            depth_image_tensor = depth_image_tensor.to(model.device, dtype=torch.float16)
    else:
        depth_image = None


    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            image = None

            if seg_image is not None:
                # first message
                inp = DEFAULT_SEG_TOKEN + '\n' + inp
                seg_image = None

                if depth_image is not None:
                    # first message
                    inp = DEFAULT_DEPTH_TOKEN + '\n' + inp
                    depth_image = None
            conv.append_message(conv.roles[0], inp)
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if "<seg>" not in prompt:
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        else:
            input_ids = tokenizer_depth_seg_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, SEG_TOKEN_INDEX, DEPTH_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                segs=seg_image_tensor,
                depths=depth_image_tensor,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="shi-labs/vcoder_ds_llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--seg-file", type=str, default=None)
    parser.add_argument("--depth-file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    main(args)