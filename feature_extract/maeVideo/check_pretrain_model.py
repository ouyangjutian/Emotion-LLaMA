# @Time    : 6/24/23 10:30 AM
# @Author  : bbbdbbb
# @File    : check_pretrain_model.py
# @Description :

from transformers import VideoMAEImageProcessor, VideoMAEForPreTraining
model = VideoMAEForPreTraining.from_pretrained("/home/amax/project/MER2023/MER2023-Baseline/feature_extraction/visual/maeVideo/videomae-large")

print(model) # [1,  ？, 1568]