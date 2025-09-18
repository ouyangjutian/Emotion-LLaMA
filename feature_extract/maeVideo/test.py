# @Time    : 6/23/23 7:27 PM
# @Author  : bbbdbbb
# @File    : test.py
# @Description :

import av
import torch
import numpy as np

from transformers import AutoImageProcessor, VideoMAEForVideoClassification
from huggingface_hub import hf_hub_download

np.random.seed(0)


def read_video_pyav(container, indices):
    """
    ...     Decode the video with PyAV decoder.
    ...     Args:
    ...         container (`av.container.input.InputContainer`): PyAV container.
    ...         indices (`List[int]`): List of frame indices to decode.
    ...     Returns:
    ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    ...
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len

    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


device = torch.device('cuda:0')

# video clip consists of 300 frames (10 seconds at 30 FPS)
file_path = hf_hub_download(
        repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
    )
container = av.open(file_path)

# sample 16 frames
indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
video = read_video_pyav(container, indices)
print('video shape: ', torch.from_numpy(video).shape) # torch.Size([16, 360, 640, 3])

image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = model.to(device)

inputs = image_processor(list(video), return_tensors="pt")
inputs = inputs.to(device)
print('inputs shape: ', inputs['pixel_values'].shape)   # torch.Size([1, 16, 3, 224, 224])

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# model predicts one of the 400 Kinetics-400 classes
print('logits shape: ', logits.shape)
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])






