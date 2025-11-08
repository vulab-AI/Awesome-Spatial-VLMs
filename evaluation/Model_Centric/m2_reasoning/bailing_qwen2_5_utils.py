from __future__ import annotations

import base64
import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO

import random
import numpy as np

import requests
import torch
import torchvision
from packaging import version

from PIL import Image
import torchaudio
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from typing import Union, Tuple, List

logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 7680 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 120

def is_decord_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("decord") is not None

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def is_image(image_file):
    if isinstance(image_file, str) and (image_file.startswith("base64,") or image_file.lower().endswith(
            ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))):
        return True
    elif isinstance(image_file, Image.Image):
        return True
    else:
        return False

def is_video(video_file):
    if isinstance(video_file, str) and video_file.lower().endswith(
            ('.mp4', '.mkv', '.avi', '.wmv', '.iso', ".webm")):
        return True
    else:
        return False

def is_audio(audio_file):
    if isinstance(audio_file, str) and audio_file.lower().endswith(
            (".wav", ".mp3", ".aac", ".flac", ".alac", ".m4a", ".ogg", ".wma", ".aiff", ".amr", ".au")):
        return True
    else:
        return False

def load_audio(audio_file, sample_rate=16000):
    waveform, orig_freq = torchaudio.load(audio_file, normalize=True)

    NORM_FACTOR_FOR_DTYPE = {
        torch.int8: 2 ** 7,
        torch.int16: 2 ** 15,
        torch.int32: 2 ** 31,
        torch.int64: 2 ** 63,
        torch.float32: 1,
        torch.float64: 1,
    }
    assert waveform.dtype in NORM_FACTOR_FOR_DTYPE, f"Unsupported waveform dtype: {waveform.dtype}"
    norm_factor = NORM_FACTOR_FOR_DTYPE[waveform.dtype]
    waveform = waveform.to(torch.float32) / norm_factor

    if len(waveform.shape) > 1:
        waveform = waveform[0]
    if orig_freq != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=sample_rate)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
    return waveform

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image

def sample_frames(num_frames, total_frames, sample="random"):
    if sample == "sequence":
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        intervals = np.linspace(start=0, stop=total_frames, num=num_frames + 1, dtype=int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == "random":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(total_frames)[:num_frames]
                frame_indices.sort()
                frame_indices = list(frame_indices)
            if len(frame_indices) < num_frames:
                padded_frame_indices = [frame_indices[-1]] * num_frames
                padded_frame_indices[:len(frame_indices)] = frame_indices
                frame_indices = padded_frame_indices
        elif sample == "uniform":
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
            if len(frame_indices) < num_frames:
                frame_indices = [
                    frame_indices[int((num_frames - 1) * i / (num_frames - 1) + 0.5)] for i in range(num_frames)
                ]
        else:
            raise NotImplementedError
    return frame_indices

def get_frames(
    ele: dict,
    total_frames: int,
) -> int:
    """calculate the number of frames for video used for model inputs.
        Args:
        ele (dict): a dict contains the configuration of video.
        total_frames (int): the original total number of frames of the video.
    Returns:
        int: the number of frames for video used for model inputs.
    """

    min_frames = ceil_by_factor(FPS_MIN_FRAMES, FRAME_FACTOR)
    max_frames = floor_by_factor(FPS_MAX_FRAMES, FRAME_FACTOR)

    if "nframes" in ele:
        num_frames = min(total_frames, ele["nframes"], max_frames)
    else:
        num_frames = min(total_frames, max_frames)
    num_frames = round_by_factor(max(num_frames, min_frames), FRAME_FACTOR)
    return num_frames

def _read_video_torchvision(
    ele: dict,
) -> torch.Tensor:
    """read video using torchvision.io.read_video
    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
        if "file://" in video_path:
            video_path = video_path[7:]

    sample_method = ele.get("sample", "sequence")
    pts_unit = "sec" if sample_method == "sequence" else "pts"
    st = time.time()
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit=pts_unit,
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    num_frames = get_frames(ele, total_frames)
    frame_indices = sample_frames(
        num_frames=num_frames, total_frames=total_frames, sample=sample_method
    )
    video = video[frame_indices]
    return video

def _read_video_decord(
    ele: dict,
) -> torch.Tensor:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"]

    st = time.time()
    vr = decord.VideoReader(video_path)
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    sample_method = ele.get("sample", "sequence")
    if sample_method == "sequence":
        total_frames = int(total_frames / video_fps * 2)

    num_frames = get_frames(ele, total_frames)
    frame_indices = sample_frames(
        num_frames=num_frames, total_frames=total_frames, sample=sample_method
    )

    video = vr.get_batch(frame_indices).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    return video

VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
}

FORCE_BAILINGNATIVE_VIDEO_READER = os.getenv("FORCE_BAILINGNATIVE_VIDEO_READER", None)

@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_BAILINGNATIVE_VIDEO_READER is not None:
        video_reader_backend = FORCE_BAILINGNATIVE_VIDEO_READER
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    print(f"bailing-native-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend

def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        if ele["video"].startswith("file://"):
            ele["video"] = ele["video"][7:]
        video_reader_backend = get_video_reader_backend()
        video = VIDEO_READER_BACKENDS[video_reader_backend](ele)

        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            num_frames, _, height, width = video.shape
            total_pixels, min_pixels = VIDEO_TOTAL_PIXELS, VIDEO_MIN_PIXELS
            max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels * FRAME_FACTOR // num_frames),
                int(VIDEO_MIN_PIXELS * 1.05))
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=28,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
        ]

        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        return images

def fetch_audio(ele: dict[str, str | torch.Tensor], return_tensor="pt") -> Union[torch.Tensor, np.ndarray]:
    if "audio" in ele:
        audio = ele["audio"]
    else:
        audio = ele["audio_url"]
    audio_obj = None
    sample_rate = ele.get("sample_rate", 16000)
    if isinstance(audio, torch.Tensor):
        image_obj = audio
    elif audio.startswith("http://") or audio.startswith("https://"):
        audio_file = BytesIO(requests.get(audio, stream=True).content)
        audio_obj = load_audio(audio_file, sample_rate=sample_rate)
    elif audio.startswith("file://"):
        audio_obj = load_audio(audio[7:], sample_rate=sample_rate)
    else:
        audio_obj = load_audio(audio, sample_rate=sample_rate)
    if return_tensor == "pt":
        return audio_obj
    else:
        return audio_obj.numpy()

def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                            "image" in ele
                            or "image_url" in ele
                            or "video" in ele
                            or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos

def process_vision_info(
    conversations: list[dict] | list[list[dict]],
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, list[
    torch.Tensor | list[np.ndarray]] | None]:
    vision_infos = extract_vision_info(conversations)
    ## Read images, videos or audios
    image_inputs = []
    video_inputs = []
    audio_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            if isinstance(vision_info["image"], (tuple, list)):
                for i in range(len(vision_info["image"])):
                    image_inputs.append(fetch_image({"type": "image", "image": vision_info["image"][i]}))
            else:
                image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info or "video_url" in vision_info:
            video_inputs.append(fetch_video(vision_info))
        elif "audio" in vision_info or "audio_url" in vision_info:
            if isinstance(vision_info["audio"], (tuple, list)):
                audio_inputs.append(fetch_image({"type": "audio", "audio": vision_info["audio"][i]}))
            else:
                audio_inputs.append(fetch_audio(vision_info))
        else:
            raise ValueError("image, image_url, video, video_url, audio or audio_url should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if len(audio_inputs) == 0:
        audio_inputs = None
    return image_inputs, video_inputs
