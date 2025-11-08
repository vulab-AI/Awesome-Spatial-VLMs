"""
Processor class for InternVL Chat.
"""
import torch
import numpy as np
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from typing import List, Union, Dict

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
import transformers
from .tokenization_internlm2 import InternLM2Tokenizer
from .tokenization_internlm2_fast import InternLM2TokenizerFast
transformers.InternLM2TokenizerFast = InternLM2TokenizerFast
transformers.InternLM2Tokenizer = InternLM2Tokenizer


logger = logging.get_logger(__name__)



IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_START_TOKEN='<img>'
IMG_END_TOKEN='</img>'
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    if isinstance(image_file, str):
        image = Image.open(image_file).convert('RGB')
    elif isinstance(image_file, np.ndarray) or isinstance(image_file, torch.Tensor):
        image = Image.fromarray(image_file).convert('RGB')
    elif isinstance(image_file, Image.Image):
        image = image_file.convert('RGB')
    else:
        raise ValueError("image_file should be a string, numpy array or PIL image")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def find_all_str_indices(string, substring):
    indices = []
    start = 0
    while True:
        idx = string.find(substring, start)
        if idx == -1:  # substring not found
            break
        indices.append(idx)
        start = idx + 1  # look for next occurrence after current one
    return indices

class InternVLChatProcessor(ProcessorMixin):
    r"""
    Constructs a InternVLChat processor.
    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["tokenizer"]
    tokenizer_class = ("InternLM2Tokenizer", "InternLM2TokenizerFast")

    def __init__(
        self, 
        tokenizer=None, 
        input_size=448,
        num_image_token=256,
        max_num_patches=12,
        use_thumbnail=True,
        video_num_segments=32,
        max_frame_num_patches=1,
        enable_cross_attention=False,
        enable_shared_cross_attention=False,
        IMAGENET_MEAN=(0.485, 0.456, 0.406),
        IMAGENET_STD=(0.229, 0.224, 0.225),
        IMG_START_TOKEN='<img>', 
        IMG_END_TOKEN='</img>', 
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
        **kwargs,
        ):
        self.num_image_token = num_image_token
        self.enable_cross_attention = enable_cross_attention
        self.enable_shared_cross_attention = enable_shared_cross_attention
        self.input_size = input_size
        self.max_num_patches = max_num_patches
        self.max_frame_num_patches = max_frame_num_patches
        self.use_thumbnail = use_thumbnail
        self.video_num_segments = video_num_segments
        self.IMAGENET_MEAN = IMAGENET_MEAN
        self.IMAGENET_STD = IMAGENET_STD
        self.IMG_START_TOKEN = IMG_START_TOKEN
        self.IMG_END_TOKEN = IMG_END_TOKEN
        self.IMG_CONTEXT_TOKEN = IMG_CONTEXT_TOKEN
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.img_start_token_id = img_start_token_id
        img_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        self.img_end_token_id = img_end_token_id
        self.bos_token_id = tokenizer.bos_token_id
        super().__init__(tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: Union[ImageInput, List[ImageInput], List[List[ImageInput]]] = None,
        videos: Union[str, List[str]] = None,
        max_num_patches: int = None,
        max_frame_num_patches: int = None,
        video_num_segments: int = None,
        return_tensors="pt",
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
        """
        texts = text
        max_num_patches = max_num_patches if max_num_patches is not None else self.max_num_patches
        video_num_segments = video_num_segments if video_num_segments is not None else self.video_num_segments
        max_frame_num_patches = max_frame_num_patches if max_frame_num_patches is not None else self.max_frame_num_patches
        if isinstance(texts, str):
            # text is a single string
            texts = [texts]
        if isinstance(images, Union[str, Image.Image, np.ndarray, torch.Tensor]):
            images = [images]
        if isinstance(videos, str):
            videos = [videos]
        elif isinstance(videos, (np.ndarray, torch.Tensor)):
            videos = [videos]
        if isinstance(videos, list):
            if isinstance(videos[0], str):
                pass
            elif isinstance(videos[0], (np.ndarray, torch.Tensor)):
                pass
            else:
                raise ValueError("videos should be a list of strings, numpy arrays or torch tensors")
        
        if images is not None:
            image_pixel_values = [load_image(image, input_size=self.input_size, max_num=max_num_patches) for image in images]
            num_patches_list = [len(x) for x in image_pixel_values]
        
        if videos is not None:
            if isinstance(videos[0], str):
                video_results = [load_video(video_path, input_size=self.input_size, num_segments=video_num_segments) for video_path in videos]
                video_pixel_values = [x[0] for x in video_results]
                video_num_patches_list = [x[1] for x in video_results]
            elif isinstance(videos[0], (np.ndarray, torch.Tensor)):
                video_pixel_values = [[load_image(frame, input_size=self.input_size, max_num=max_frame_num_patches) for frame in video] for video in videos]
                video_num_patches_list = [[len(x) for x in video] for video in video_pixel_values]
                video_pixel_values = [torch.cat(video) for video in video_pixel_values]
                
        
        if images is not None or videos is not None:
            merged_pixel_values = []
            merged_text = "".join(texts)
            
            image_placeholder_idxs = find_all_str_indices(merged_text, "<image>")
            video_placeholder_idxs = find_all_str_indices(merged_text, "<video>")
            if images is not None:
                assert len(image_placeholder_idxs) == len(image_pixel_values), f"Expect {len(image_pixel_values)} images, but get {len(image_placeholder_idxs)} image placeholders"
            if videos is not None:
                assert len(video_placeholder_idxs) == len(video_pixel_values), f"Expect {len(video_pixel_values)} videos, but get {len(video_placeholder_idxs)} video placeholders"
                
            place_holder_idxs = sorted([(idx, "<image>") for idx in image_placeholder_idxs] + [(idx, "<video>") for idx in video_placeholder_idxs], key=lambda x: x[0])
            image_idx, video_idx = 0, 0
            for i, (idx, placeholder) in enumerate(place_holder_idxs):
                if placeholder == "<image>":
                    merged_pixel_values.append(image_pixel_values[image_idx])
                    image_idx += 1
                elif placeholder == "<video>":
                    merged_pixel_values.append(video_pixel_values[video_idx])
                    video_idx += 1
                else:
                    raise ValueError(f"Invalid placeholder: {placeholder}")
            
            if self.enable_shared_cross_attention:
                # put all the <image> and <video> placeholders to the starting of each text
                for i in range(len(texts)):
                    cur_text = texts[i]
                    text_prefix = ""
                    image_placeholder_idxs = find_all_str_indices(cur_text, "<image>")
                    video_placeholder_idxs = find_all_str_indices(cur_text, "<video>")
                    place_holder_idxs = sorted([(idx, "<image>") for idx in image_placeholder_idxs] + [(idx, "<video>") for idx in video_placeholder_idxs], key=lambda x: x[0])
                    for j, (idx, placeholder) in enumerate(place_holder_idxs):
                        text_prefix += placeholder + "\n"
                    texts[i] = text_prefix + cur_text.replace("<image>", "").replace("<video>", "")
        else:
            merged_pixel_values = None
        
        
            
        
        if merged_pixel_values:
            merged_pixel_values = torch.cat(merged_pixel_values)
        
        if images is not None:
            assert isinstance(texts, list) and isinstance(texts[0], str), "texts should be a list of strings"
            assert isinstance(images, list) and isinstance(images[0], Union[str, Image.Image, np.ndarray, torch.Tensor]), "images should be a list of strings, PIL images, numpy arrays or torch tensors"
            
            all_image_counts = sum([text.count("<image>") for text in texts])
            assert all_image_counts == len(images), f"Expect {all_image_counts} images, but get {len(images)} images"
            
            queries = []
            image_idx = 0
            for i in range(len(texts)):
                text = texts[i]
                num_image = text.count("<image>")
                _num_patches_list = num_patches_list[image_idx: image_idx + num_image]
                if num_image > 0:
                    for j, _num_patches in enumerate(_num_patches_list):
                        if not self.enable_cross_attention:
                            image_tokens = self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * self.num_image_token * _num_patches + self.IMG_END_TOKEN
                        else:
                            image_tokens = self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * _num_patches + self.IMG_END_TOKEN
                        text = text.replace("<image>", image_tokens, 1)
                    image_idx += num_image
                queries.append(text)
        else:
            queries = texts
        
        if videos is not None:
            assert isinstance(queries, list) and isinstance(queries[0], str), "queries should be a list of strings"
            
            all_video_counts = sum([query.count("<video>") for query in queries])
            assert all_video_counts == len(videos), f"Expect {all_video_counts} videos, but get {len(videos)} videos"
            
            new_queries = []
            video_idx = 0
            for i in range(len(queries)):
                query = queries[i]
                num_video = query.count("<video>")
                _video_num_patches_list = video_num_patches_list[video_idx: video_idx + num_video]
                if num_video > 0:
                    for j, _num_patches_list in enumerate(_video_num_patches_list):
                        video_prefix = ''.join([f'Frame{x+1}: <image>\n' for x in range(len(_num_patches_list))])
                        for k, _num_patches in enumerate(_num_patches_list):
                            if not self.enable_cross_attention:
                                image_tokens = self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * self.num_image_token * _num_patches + self.IMG_END_TOKEN
                            else:
                                image_tokens = self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * _num_patches + self.IMG_END_TOKEN
                            video_prefix = video_prefix.replace("<image>", image_tokens, 1)
                        query = query.replace("<video>", video_prefix, 1)
                    video_idx += num_video
                new_queries.append(query)
        else:
            new_queries = queries
        
        
        if merged_pixel_values is not None:
            if not self.enable_cross_attention:
                # count IMG_CONTEXT_TOKEN, should be equal to the number of pixel_values
                assert sum([query.count(self.IMG_CONTEXT_TOKEN) for query in new_queries]) // self.num_image_token == len(merged_pixel_values), f"Expect {len(merged_pixel_values)} pixel_values, but get {sum([query.count(self.IMG_CONTEXT_TOKEN) for query in new_queries]) // self.num_image_token} IMG_CONTEXT_TOKEN"        
            else:
                # count IMG_CONTEXT_TOKEN, should be equal to the len(merged_pixel_values)
                assert sum([query.count(self.IMG_CONTEXT_TOKEN) for query in new_queries]) == len(merged_pixel_values), f"Expect {len(merged_pixel_values)} pixel_values, but get {sum([query.count(self.IMG_CONTEXT_TOKEN) for query in new_queries])} IMG_CONTEXT_TOKEN"
        else:
            pass
            # print("No images or videos")   
        model_inputs = self.tokenizer(new_queries, return_tensors=return_tensors, **kwargs)
        model_inputs["pixel_values"] = merged_pixel_values
        
        return BatchFeature(data=model_inputs)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        return list(dict.fromkeys(tokenizer_input_names))

    def _right_pad_inputs_with_attention_mask(self, model_inputs: List[Dict]):
        results = {}
        for k in model_inputs[0].keys():
            if model_inputs[0][k] is not None:
                if k == 'input_ids':
                    # add padding
                    max_length = max([inputs[k].shape[1] for inputs in model_inputs])
                    pad_token_id = self.tokenizer.pad_token_id
                    # pad all inputs to the same length
                    results[k] = torch.cat(
                        [
                            torch.cat(
                                [
                                    inputs[k],
                                    torch.tensor(
                                        [pad_token_id] * (max_length - inputs[k].shape[1]),
                                        dtype=inputs[k].dtype,
                                        device=inputs[k].device,
                                    ).unsqueeze(0),
                                ],
                                dim=1,
                            )
                            if inputs[k].shape[1] < max_length
                            else inputs[k]
                            for inputs in model_inputs
                        ],
                        dim=0,
                    )
                elif 'attention_mask' in k:
                    v = model_inputs[0][k]
                    if v.dim() == 2:
                        # add attention mask
                        max_length = max([inputs[k].shape[1] for inputs in model_inputs])
                        results[k] = torch.cat(
                            [
                                torch.cat(
                                    [
                                        inputs[k],
                                        torch.tensor(
                                            [0] * (max_length - inputs[k].shape[1]),
                                            dtype=inputs[k].dtype,
                                            device=inputs[k].device,
                                        ).unsqueeze(0),
                                    ],
                                    dim=1,
                                )
                                if inputs[k].shape[1] < max_length
                                else inputs[k]
                                for inputs in model_inputs
                            ],
                            dim=0,
                        )
                    elif v.dim() == 4:
                        # prepared 4d attention mask, [batch_size, num_heads, q_seq_length, kv_seq_length]
                        max_q_length = max([inputs[k].shape[2] for inputs in model_inputs])
                        max_kv_length = max([inputs[k].shape[3] for inputs in model_inputs])
                        
                        all_padded_attention_mask = []
                        for inputs in model_inputs:
                            attention_mask = inputs[k]
                            cur_q_length = attention_mask.shape[2]
                            cur_kv_length = attention_mask.shape[3]
                            padded_attention_mask = torch.cat(
                                [
                                    attention_mask,
                                    torch.zeros(
                                        (attention_mask.shape[0], attention_mask.shape[1], max_q_length - cur_q_length, cur_kv_length),
                                        dtype=attention_mask.dtype,
                                        device=attention_mask.device,
                                    ),
                                ],
                                dim=2,
                            ) if attention_mask.shape[2] < max_q_length else attention_mask
                            
                            padded_attention_mask = torch.cat(
                                [
                                    padded_attention_mask,
                                    torch.zeros(
                                        (attention_mask.shape[0], attention_mask.shape[1], max_q_length, max_kv_length - cur_kv_length),
                                        dtype=attention_mask.dtype,
                                        device=attention_mask.device,
                                    ),
                                ],
                                dim=3,
                            ) if attention_mask.shape[3] < max_kv_length else padded_attention_mask
                            all_padded_attention_mask.append(padded_attention_mask)
                        results[k] = torch.cat(all_padded_attention_mask, dim=0)
                elif k == 'labels':
                    # pad with -100
                    max_length = max([inputs[k].shape[1] for inputs in model_inputs])
                    results[k] = torch.cat(
                        [
                            torch.cat(
                                [
                                    inputs[k],
                                    torch.tensor(
                                        [-100] * (max_length - inputs[k].shape[1]),
                                        dtype=inputs[k].dtype,
                                        device=inputs[k].device,
                                    ).unsqueeze(0),
                                ],
                                dim=1,
                            )
                            if inputs[k].shape[1] < max_length
                            else inputs[k]
                            for inputs in model_inputs
                        ],
                        dim=0,
                    )
                elif 'position_ids' in k:
                    # pad with 0
                    max_length = max([inputs[k].shape[1] for inputs in model_inputs])
                    results[k] = torch.cat(
                        [
                            torch.cat(
                                [
                                    inputs[k],
                                    torch.tensor(
                                        [0] * (max_length - inputs[k].shape[1]),
                                        dtype=inputs[k].dtype,
                                        device=inputs[k].device,
                                    ).unsqueeze(0),
                                ],
                                dim=1,
                            )
                            if inputs[k].shape[1] < max_length
                            else inputs[k]
                            for inputs in model_inputs
                        ],
                        dim=0,
                    )
                else:
                    results[k] = torch.cat([inputs[k] for inputs in model_inputs], dim=0)
            else:
                results[k] = None
        return results
    