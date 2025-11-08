import argparse
import datetime
import json
import os
import time

import gradio as gr
import hashlib

from vcoder_llava.vcoder_conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)
from vcoder_llava.constants import LOGDIR
from vcoder_llava.utils import (build_logger, server_error_msg,
                          violates_moderation, moderation_msg)
from .chat import Chat


logger = build_logger("gradio_app", "gradio_web_server.log")

headers = {"User-Agent": "VCoder Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    state = default_conversation.copy()
    return state

def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
        }
        fout.write(json.dumps(data) + "\n")

def upvote_last_response(state, model_selector, request: gr.Request):
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3

def regenerate(state, image_process_mode, seg_process_mode, depth_process_mode):
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode, prev_human_msg[1][3], seg_process_mode, prev_human_msg[1][5], depth_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None, None, None) + (disable_btn,) * 5


def clear_history(request: gr.Request):
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None, None, None) + (disable_btn,) * 5


def add_text(state, text, image, image_process_mode, seg, seg_process_mode, depth, depth_process_mode, request: gr.Request):
    logger.info(f"add_text. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None, None, None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg, None, None, None) + (
                no_change_btn,) * 5

    text = text[:1200]  # Hard cut-off
    if image is not None:
        text = text[:864]  # Hard cut-off for images
        if '<image>' not in text:
            text = '<image>\n' + text
        if seg is not None:
            if '<seg>' not in text:
                text = '<seg>\n' + text
        if depth is not None:
            if '<depth>' not in text:
                text = '<depth>\n' + text
    
        text = (text, image, image_process_mode, seg, seg_process_mode, depth, depth_process_mode)
        if len(state.get_images(return_pil=True)) > 0:
            state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None, None, None) + (disable_btn,) * 5


def http_bot(state, model_selector, temperature, top_p, max_new_tokens, request: gr.Request):
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        if "llava" in model_name.lower():
            template_name = "llava_v1"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state
    
    # Construct prompt
    prompt = state.get_prompt()

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "images": f'List of {len(state.get_images())}',
        "segs": f'List of {len(state.get_segs())}',
        "depths": f'List of {len(state.get_depths())}',
    }
    logger.info(f"==== request ====\n{pload}")

    pload['images'] = state.get_images()
    pload['segs'] = state.get_segs()
    pload['depths'] = state.get_depths()

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5


    try:
    # Stream output
        response = chat.generate_stream_gate(pload)
        for chunk in response:
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.03)
    except Exception:
        gr.Warning(server_error_msg)
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5
    logger.info(f"{output}")


title = "<h1 style='margin-bottom: -10px; text-align: center'>VCoder: Versatile Vision Encoders for Multimodal Large Language Models</h1>"
# style='
description = "<p style='font-size: 16px; margin: 5px; font-weight: w300; text-align: center'> <a href='https://praeclarumjj3.github.io/' style='text-decoration:none' target='_blank'>Jitesh Jain, </a> <a href='https://jwyang.github.io/' style='text-decoration:none' target='_blank'>Jianwei Yang, <a href='https://www.humphreyshi.com/home' style='text-decoration:none' target='_blank'>Humphrey Shi</a></p>" \
            + "<p style='font-size: 16px; margin: 5px; font-weight: w600; text-align: center'> <a href='https://praeclarumjj3.github.io/vcoder/' target='_blank'>Project Page</a> | <a href='https://praeclarumjj3.github.io/vcoder/' target='_blank'>Video</a> | <a href='https://arxiv.org/abs/2211.06220' target='_blank'>ArXiv Paper</a> | <a href='https://github.com/SHI-Labs/VCoder' target='_blank'>Github Repo</a></p>" \
            + "<p style='text-align: center; font-size: 16px; margin: 5px; font-weight: w300;'> [Note: You can obtain segmentation maps for your image using the <a href='https://huggingface.co/spaces/shi-labs/OneFormer' style='text-decoration:none' target='_blank'>OneFormer Demo</a> and the depth map from <a href='https://github.com/facebookresearch/dinov2/blob/main/notebooks/depth_estimation.ipynb' style='text-decoration:none' target='_blank'>DINOv2</a>. Please click on Regenerate button if you are unsatisfied with the generated response. You may find screenshots of our demo trials <a href='https://github.com/SHI-Labs/VCoder/blob/main/images/' style='text-decoration:none' target='_blank'>here</a>.]</p>"

tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes.
""")


learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only, subject to the [License](https://huggingface.co/lmsys/vicuna-7b-v1.5) of Vicuna-v1.5, [License](https://github.com/haotian-liu/LLaVA/blob/main/LICENSE) of LLaVA, [Terms of Use](https://cocodataset.org/#termsofuse) of the COCO dataset, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
""")

block_css = """
#buttons button {
    min-width: min(120px,100%);
}
"""

def build_demo(embed_mode):
    
    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
    with gr.Blocks(title="VCoder", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title)
            gr.Markdown(description)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=[model + "-4bit" for model in models],
                        value=models[0]+"-4bit" if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False)

                # with gr.Row():
                imagebox = gr.Image(type="pil", label="Image Input")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image", visible=False)

                with gr.Row():
                    segbox = gr.Image(type="pil", label="Seg Map")
                    seg_process_mode = gr.Radio(
                        ["Crop", "Resize", "Pad", "Default"],
                        value="Default",
                        label="Preprocess for non-square Seg Map", visible=False)
                    
                    depthbox = gr.Image(type="pil", label="Depth Map")
                    depth_process_mode = gr.Radio(
                        ["Crop", "Resize", "Pad", "Default"],
                        value="Default",
                        label="Preprocess for non-square Depth Map", visible=False)

                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.8, step=0.1, interactive=True, label="Temperature",)
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, interactive=True, label="Top P",)
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(elem_id="chatbot", label="VCoder Chatbot", height=550)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    #stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        gr.Examples(examples=[
            [f"{cur_dir}/examples/people.jpg", f"{cur_dir}/examples/people_pan.png", None, "What objects can be seen in the image?", "0.9", "1.0"],
            [f"{cur_dir}/examples/corgi.jpg", f"{cur_dir}/examples/corgi_pan.png", None, "What objects can be seen in the image?", "0.6", "0.7"],
            [f"{cur_dir}/examples/suits.jpg", f"{cur_dir}/examples/suits_pan.png", f"{cur_dir}/examples/suits_depth.jpeg", "Can you describe the depth order of the objects in this image, from closest to farthest?", "0.2", "0.5"], 
            [f"{cur_dir}/examples/depth.jpeg", f"{cur_dir}/examples/depth_pan.png", f"{cur_dir}/examples/depth_depth.png", "Can you describe the depth order of the objects in this image, from closest to farthest?", "0.2", "0.5"], 
            [f"{cur_dir}/examples/friends.jpg", f"{cur_dir}/examples/friends_pan.png", None, "What is happening in the image?", "0.8", "0.9"],
            [f"{cur_dir}/examples/suits.jpg", f"{cur_dir}/examples/suits_pan.png", None, "What objects can be seen in the image?", "0.5", "0.5"], 
        ], inputs=[imagebox, segbox, depthbox, textbox, temperature, top_p])
        
        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        upvote_btn.click(upvote_last_response,
            [state, model_selector], [textbox, upvote_btn, downvote_btn, flag_btn])
        downvote_btn.click(downvote_last_response,
            [state, model_selector], [textbox, upvote_btn, downvote_btn, flag_btn])
        flag_btn.click(flag_last_response,
            [state, model_selector], [textbox, upvote_btn, downvote_btn, flag_btn])
        regenerate_btn.click(regenerate, [state, image_process_mode, seg_process_mode, depth_process_mode],
            [state, chatbot, textbox, imagebox, segbox, depthbox] + btn_list).then(
            http_bot, [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list)
        clear_btn.click(clear_history, None, [state, chatbot, textbox, imagebox, segbox, depthbox] + btn_list)

        textbox.submit(add_text, [state, textbox, imagebox, image_process_mode, segbox, seg_process_mode, depthbox, depth_process_mode], [state, chatbot, textbox, imagebox, segbox, depthbox] + btn_list
            ).then(http_bot, [state, model_selector, temperature, top_p, max_output_tokens],
                   [state, chatbot] + btn_list)
        submit_btn.click(add_text, [state, textbox, imagebox, image_process_mode, segbox, seg_process_mode, depthbox, depth_process_mode], [state, chatbot, textbox, imagebox, segbox, depthbox] + btn_list
            ).then(http_bot, [state, model_selector, temperature, top_p, max_output_tokens],
                   [state, chatbot] + btn_list)

        demo.load(load_demo_refresh_model_list, None, [state])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="shi-labs/vcoder_ds_llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.model_name is None:
        model_paths = args.model_path.split("/")
        if model_paths[-1].startswith('checkpoint-'):
            model_name = model_paths[-2] + "_" + model_paths[-1]
        else:
            model_name = model_paths[-1]
    else:
        model_name = args.model_name

    models = [model_name]
    chat = Chat(
        args.model_path,
        args.model_base,
        args.model_name,
        args.load_8bit,
        args.load_4bit,
        args.device,
        logger
    )

    logger.info(args)
    demo = build_demo(args.embed)
    demo.queue(
        concurrency_count=args.concurrency_count,
        api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )
