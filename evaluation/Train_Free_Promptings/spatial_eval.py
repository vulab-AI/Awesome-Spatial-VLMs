from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor,AutoModelForVision2Seq, LlavaForConditionalGeneration,LlavaNextProcessor, LlavaNextForConditionalGeneration,LlavaOnevisionForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
import json
import os
from datasets import load_dataset
import bitsandbytes 
from transformers import BitsAndBytesConfig
from scaffold import *

# def make_question_prompt(image, question, options):

#     return (
#         f"Answer the question based on the image:\n"
#         f"Question: {question}\n"
#         f"{options}\n"
#         "Only return the answer (the option or a answer phrase)."
#     )

def decode_generated_ids(model_name, processor, inputs, generated_ids):
    model_name = model_name.lower()

    if "llava-next-72b" in model_name:
        texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        cleaned = []
        for t in texts:
            # 专门截取最后一个 [/INST] 之后的内容
            if "[/INST]" in t:
                t = t.split("[/INST]")[-1].strip()
            cleaned.append(t)
        texts = cleaned

    else:
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

    return texts


def resize_max_500(image: Image.Image) -> Image.Image:
    max_size = 500
    width, height = image.size
    # 如果最大边长已经<=1080，直接返回原图
    if max(width, height) <= max_size:
        return image

    if width >= height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image

def make_message_prompt(item, prompt_type,model,processor,model_path,path=[None,None]):
    image_list = [f"image_{i}" for i in range(32)]
    message= []
    user_content = []

    # 构造 system message（可选）
    if prompt_type == "spatialprompt":
        message = [
            {
                "role": "system", 
                "content": [
                    {
                        "type": "text", 
                        "text": (
                            "You are a vision assistant. Use the following 4 steps sequentially to answer the question: Step 1 **Analyze the question** Step 2 **Identify up to 10 reference scales in the image, ranging from large to small sizes, and list them in the specified format** - A reference scale must be typical in size. - A reference scale can be the dimensions of an object or an object part. - A reference scale must NOT be floor tiles or floor planks. - Formulate the reference scales using the format: \"\"\"The [choose from front-to-back, side-to-side, left-to-right, diameter, height (top to bottom edge), or mounting height (bottom edge to floor)] of [object or object part] is approximately [dimension estimate].\"\"\" Step 3 **Propose a robust step-by-step plan to answer the question by using the reference scales in Step 2** - A robust step-by-step plan performs the estimation in a coarse-to-fine manner. - First, use a reliable and large-sized reference scale as the primary reference for estimation. - Then, gradually use a reliable and smaller-sized reference scale for adjustment. - Repeat until the estimation is precise enough. - When performing visual comparison, be aware of perspective distortion. - Do NOT rely on pixel measurements from the images. Step 4 **Focus on the image and follow the plan in Step 3 to answer the question**"
                        )
                    }
                ]
            }
        ]

    if prompt_type == "ccot":
        message = [
            {
                "role": "system", 
                "content": [
                    {
                        "type": "text", 
                        "text": (
                            "You are a vision assistant. Use the following 4 steps sequentially to answer the question: Step 1 **Analyze the question** Step 2 **Identify up to 10 reference scales in the image, ranging from large to small sizes, and list them in the specified format** - A reference scale must be typical in size. - A reference scale can be the dimensions of an object or an object part. - A reference scale must NOT be floor tiles or floor planks. - Formulate the reference scales using the format: \"\"\"The [choose from front-to-back, side-to-side, left-to-right, diameter, height (top to bottom edge), or mounting height (bottom edge to floor)] of [object or object part] is approximately [dimension estimate].\"\"\" Step 3 **Propose a robust step-by-step plan to answer the question by using the reference scales in Step 2** - A robust step-by-step plan performs the estimation in a coarse-to-fine manner. - First, use a reliable and large-sized reference scale as the primary reference for estimation. - Then, gradually use a reliable and smaller-sized reference scale for adjustment. - Repeat until the estimation is precise enough. - When performing visual comparison, be aware of perspective distortion. - Do NOT rely on pixel measurements from the images. Step 4 **Focus on the image and follow the plan in Step 3 to answer the question**"
                        )
                    }
                ]
            }
        ]
        for image_key in image_list:
            image_obj = item.get(image_key, None)
            if image_obj is not None:
                image = image_obj.convert("RGB")
                image = resize_max_500(image)
                user_content.append({
                    "type": "image",
                    "image": image
                })

        sg_prompt="For the provided image and question-answer pair, generate a scene graph in JSON format to improve the quality and/or detail of the answer. The scene graph can include the following:\n1. Objects that are relevant to answering the question.\n2. Object attributes that are relevant to answering the question.\n3. Object relationships that are relevant to answering the question.\nScene Graph:\n"
        user_content.append({
            "type": "text",
            "text": sg_prompt
        })
        message.append({
            "role": "user",
            "content": user_content  # ✅ 一定要加 content
        })

        #generate the scene graph first
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_input,_ = process_vision_info(message)
        inputs = processor(
            text=text,
            images=image_input,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        model.eval()
                
        with torch.no_grad():
            generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=512, do_sample=False)
        
        scene_graph = decode_generated_ids(model_path, processor, inputs, generated_ids)[0].strip()


        message = [
            {
                "role": "system", 
                "content": [
                    {
                        "type": "text", 
                        "text": (
                            "You are a vision assistant. Based on the belowing scene graph and image, answer the question. Scene graph:\n"+ scene_graph +"\n"
                        )
                    }
                ]
            }
        ]
        # #clear cache
        del inputs
        del generated_ids
        del image_input

        torch.cuda.empty_cache()

    if prompt_type == "scaffold":
        message = [
            {
                "role": "system", 
                "content": [
                    {
                        "type": "text", 
                        "text": (
                        "Images are provided, each overlaid with a grid "
                        "of dots arranged in a matrix with dimensions h by "
                        "w. Each dot on this grid is assigned a unique set "
                        "of three-dimensional coordinates labeled as (t, x, y). "
                        "The first coordinate, 't', serves to distinguish between "
                        "the two images: '1' is assigned to the first image on "
                        "the left, and '2' to the second image on the right. "
                        "The other two coordinates, 'x' and 'y', are used to "
                        "specify the dot’s spatial location within its respective "
                        "image. This labeling system is designed to assist you "
                        "in identifying and referring to specific points within "
                        "each image.\n\n"
                        "1. When you mention any key objects in the image, first output their nearest coordinates then identify them.\n"
                        "2. You use the coordinates to determine the spatial relationships of the objects. Within each column, the x-coordinate increases from top to bottom, and within each row, the y-coordinate increases from left to right.\n"
                        "3. You can search and reason region by region with the help of the dots.\n"
                        "4. Finally, provide the answer to the question."
                    )

                    }
                ]
            }
        ]
    
        
         
    if prompt_type == "som":
        message = [
            {
                "role": "system", 
                "content": [
                    {
                        "type": "text", 
                        "text": (
                            " I have labeled a bright numeric ID at the center for each visual object in the image. Using these numeric IDs, you can easily identify and refer to specific objects within the image. When answering the question, please refer to the objects by their numeric IDs instead of their names or descriptions. This will help ensure clarity and precision in your responses."
                    )

                    }
                ]
            }
        ]

    if prompt_type == "visualsketchpad":
        pass
    
    # 构造 user message
    user_content = []

    if prompt_type== "scaffold":
        images=[]
        for image_key in image_list:
            image_obj = item.get(image_key, None)
            if image_obj is not None:
                image = image_obj.convert("RGB")
                image = resize_max_500(image)
                images.append(image)
        images=process(images)

        for img in images:
            user_content.append({
                "type": "image",
                "image": img
            })
    elif prompt_type == "som":
        if path[0] is None:
            raise ValueError("For som prompt type, please provide the mask_path argument.")
        
        mask_root = os.path.join(path[0], path[1])  # bench 的 mask 路径
        for image_key in image_list:
            image_obj = item.get(image_key, None)
            if image_obj is not None:
                image_path = os.path.join(mask_root, f"{item['id']}_{image_key}.jpg")
                try:
                    # 尝试读取 mask 图片
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    # 出错时 fallback 用原图
                    print(f"[WARN] Could not load mask {image_path}: {e}. Using original image.")
                    image = image_obj.convert("RGB")
                    image = resize_max_500(image)
                
                user_content.append({
                    "type": "image",
                    "image": image
                })
        

    else:
        for image_key in image_list:
            image_obj = item.get(image_key, None)
            if image_obj is not None:
                image = image_obj.convert("RGB")
                image = resize_max_500(image)
                user_content.append({
                    "type": "image",
                    "image": image
                })

    prompt = item.get("prompt", "")
    prompt += "\nIf it's selection question, only return the option letter. Else, only return the answer phrase."
    user_content.append({
        "type": "text",
        "text": prompt
    })
    
    message.append({
        "role": "user",
        "content": user_content  # ✅ 一定要加 content
    })

    return message


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--output_path", type=str,default="output/llava_next")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompt_type", type=str, default="ccot",help="spatialprompt, ccot, scaffold, som, visualsketchpad")
    parser.add_argument("--som_image_path", type=str, default="/som/output",help="The path to the folder containing the som masked images.")
    return parser.parse_args()



#make it batchsize = 50
def main():
    args = parse_args()
    if not os.path.exists(args.output_path):
       os.makedirs(args.output_path)
    #load dataset
    dataset =load_dataset("LLDDSS/Awesome_Spatial_VQA_Benchmarks")
    # dataset = load_dataset("LLDDSS/Awesome_Spatial_VQA_Benchmarks_ViewSpatial-Bench", download_mode="force_redownload")
    #load qwen2.5 model
    processor = None
    model= None
    if 'Qwen2.5' in args.model_path:
        processor = AutoProcessor.from_pretrained(args.model_path, padding_side='left',use_fast=True)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # You can also use 'fp4' depending on your requirements
            bnb_4bit_compute_dtype=torch.bfloat16  # Ensure your hardware supports bfloat16
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, device_map="auto",quantization_config=quantization_config) 

    elif "llava-v1.6" in args.model_path or "llava-next-72b" in args.model_path:

        #llava-hf/llava-v1.6-mistral-7b-hf seems not support multiple images
        processor = LlavaNextProcessor.from_pretrained(args.model_path,use_fast=True)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # You can also use 'fp4' depending on your requirements
            bnb_4bit_compute_dtype=torch.bfloat16  # Ensure your hardware supports bfloat16
        )
        # processor = AutoProcessor.from_pretrained(args.model_path,use_fast=True)
        model = LlavaNextForConditionalGeneration.from_pretrained(args.model_path, device_map="auto", quantization_config=quantization_config)



    for bench in dataset.keys():
        messages = []
        id_list = []
        print("Processing bench:", bench)
        result_file = os.listdir(args.output_path)
        if f"{bench}_results.json" in result_file:
            print(f"Results for {bench} already exist, skipping...")
            continue
        for item in dataset[bench]:
            message = make_message_prompt(item,args.prompt_type,model,processor,args.model_path,path=[args.som_image_path,bench])
            messages.append(message)
            id_list.append({
                "bench_name": bench,
                "id": item["id"],
                "prompt": item["prompt"],
                "options": item["options"],
                "answer": item["GT"]
            })

        print("processed bench:", bench, "with", len(messages), "messages.")
        all_outputs = []

        batch_size = args.batch_size
        i = 0
        pbar = tqdm(total=len(messages), desc=f"Processing {bench}")

        while i < len(messages):
            current_batch_size = min(batch_size, len(messages) - i)
            success = False

            while not success:
                batch_messages = messages[i:i + current_batch_size]
                batch_id_list = id_list[i:i + current_batch_size]

                text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
                # text=[msg for msg in batch_messages]
                image_inputs, _ = process_vision_info(batch_messages)
                inputs = processor(
                    text=text,
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device)

                model.eval()
                try:
                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)

                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    if "Qwen2.5" in args.model_path:
                        batch_output_text = processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                    elif "llava" in args.model_path:
                        batch_output_text = processor.batch_decode(
                            [id for id in generated_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                    
                        
                    print("batch_output_text", batch_output_text, flush=True)

                    for output_text, item_id in zip(batch_output_text, batch_id_list):
                        output_text = output_text.strip()
                        all_outputs.append({
                            "bench_name": item_id["bench_name"],
                            "id": item_id["id"],
                            "prompt": item_id["prompt"],
                            "options": item_id["options"],
                            "GT": item_id["answer"],
                            "result": output_text
                        })

                    success = True
                    i += current_batch_size
                    pbar.update(current_batch_size)

                    # 如果之前缩小过batch size，恢复到原始大小
                    if batch_size < args.batch_size:
                        batch_size = args.batch_size

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"CUDA OOM with batch size {current_batch_size}, reducing batch size by half and retrying...")
                        torch.cuda.empty_cache()
                        current_batch_size = current_batch_size // 2
                        if current_batch_size == 0:
                            raise RuntimeError("Batch size reduced to 0, cannot proceed.")
                    else:
                        raise e

            torch.cuda.empty_cache()

        pbar.close()

        with open(f"{args.output_path}/{bench}_results.json", "w") as f:
            json.dump(all_outputs, f, indent=4)

            


if __name__ == "__main__":
    main()