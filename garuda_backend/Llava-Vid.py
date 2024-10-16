#!/usr/bin/env python3

# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git

# Import necessary libraries
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
import cv2


# Suppress warnings
warnings.filterwarnings("ignore")

def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    """
    Load video frames from the given path.
    
    Args:
    - video_path (str): Path to the video file
    - max_frames_num (int): Maximum number of frames to extract
    - fps (int): Frames per second to sample
    - force_sample (bool): Whether to force uniform sampling
    
    Returns:
    - numpy.ndarray: Extracted video frames
    - str: Frame timestamps
    - float: Total video duration
    """
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    
    return spare_frames, frame_time, video_time

def load_video_with_opencv(video_path, max_frames_num):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_every_n = max(1, total_frames // max_frames_num)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i % sample_every_n == 0:
            frame = cv2.resize(frame, (336, 336))
            frames.append(frame)

        if len(frames) >= max_frames_num:
            break

    cap.release()
    return np.array(frames), fps, total_frames / fps

# Set up model parameters
pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = torch.device("mps")

# Load the pre-trained model onto CPU or MPS
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name, torch_dtype=torch.float32, attn_implementation="eager"
)
model.to(device).eval()

# Set up video parameters
video_path = "/Users/shiven/Desktop/HackHarvard/test_vid_2.mp4"
max_frames_num = 150

# Load and process the video with OpenCV
video, frame_time, video_time = load_video_with_opencv(video_path, max_frames_num)
video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"]
video = [v.to(device) for v in video]

# Set up conversation parameters
conv_template = "qwen_1_5"  # Make sure to use the correct chat template for different models
time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. Please answer the following questions related to this video."
question = DEFAULT_IMAGE_TOKEN + f"{time_instruction}\nPlease describe this video in detail."

# Prepare the conversation
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

# Tokenize the input
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

# Generate the response
cont = model.generate(
    input_ids,
    images=video,
    modalities=["video"],
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)

# Decode and print the output
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
print(text_outputs)