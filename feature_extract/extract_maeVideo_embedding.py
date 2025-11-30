# @Time    : 6/24/23 9:33 AM
# @Author  : bbbdbbb
# @File    : extract_maeVideo_embedding.py
# @Description : load maeVideo model to extract video feature embedding

# 导入必要的库
import os  # 提供操作系统相关功能，如文件路径操作
import argparse  # 用于解析命令行参数
import numpy as np  # 提供高效的数值计算功能
import matplotlib.pyplot as plt  # 用于数据可视化

# PyTorch相关库
import torch  # PyTorch核心库
import torch.nn.parallel  # 支持模型并行计算
import torch.optim  # 提供优化算法（如SGD、Adam）
import torch.utils.data  # 数据加载和批处理工具
import torchvision.transforms as transforms  # 图像预处理（如归一化、裁剪）
from timm.models.layers import trunc_normal_  # 提供权重初始化方法
from timm.models import create_model  # 用于创建预定义模型

# 系统路径相关
import sys
sys.path.append('../../')  # 将项目根目录添加到Python路径
import config  # 项目配置文件
from dataset import FaceDataset  # 自定义数据集加载类

# MAE Video相关模块
from maeVideo import models_vit  # 提供ViT模型实现
from collections import OrderedDict  # 有序字典，用于保持键的顺序
from maeVideo.modeling_finetune import vit_large_patch16_224  # 预训练的ViT模型
from maeVideo.dataset_MER import train_data_loader, test_data_loader  # 训练和测试数据加载器


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    """
    加载预训练模型的权重到当前模型
    :param model: 当前模型实例
    :param state_dict: 预训练模型的权重字典
    :param prefix: 权重键名前缀（用于模块嵌套）
    :param ignore_missing: 需要忽略的权重键名（用|分隔）
    """
    missing_keys = []  # 记录缺失的权重键
    unexpected_keys = []  # 记录意外的权重键
    error_msgs = []  # 记录加载过程中的错误信息
    metadata = getattr(state_dict, '_metadata', None)  # 获取权重字典的元数据
    state_dict = state_dict.copy()  # 复制权重字典以避免修改原始数据
    if metadata is not None:
        state_dict._metadata = metadata  # 保留元数据

    def load(module, prefix=''):
        """递归加载权重到模型的各个子模块"""
        """递归加载权重到模型"""
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')  # 递归加载子模块

    load(model, prefix=prefix)  # 加载权重

    # 处理缺失的权重键
    warn_missing_keys = []  # 需要警告的缺失键
    ignore_missing_keys = []  # 忽略的缺失键
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):  # 检查是否需要忽略
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)  # 需要警告的键
        else:
            ignore_missing_keys.append(key)  # 忽略的键

    missing_keys = warn_missing_keys  # 更新缺失键列表

    # 打印相关信息
    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))  # 打印所有错误信息


class TubeMaskingGenerator:
    """
    生成视频掩码的类，用于MAE（Masked Autoencoder）训练
    """
    def __init__(self, input_size, mask_ratio):
        """
        初始化掩码生成器
        :param input_size: 输入视频的尺寸（帧数, 高度, 宽度）
        :param mask_ratio: 掩码比例
        """
        self.frames, self.height, self.width = input_size  # 视频的帧数、高度、宽度
        self.num_patches_per_frame =  self.height * self.width  # 每帧的patch数量
        self.total_patches = self.frames * self.num_patches_per_frame  # 总patch数量
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)  # 每帧掩码数量
        self.total_masks = self.frames * self.num_masks_per_frame  # 总掩码数量

    def __repr__(self):
        """返回掩码生成器的描述信息"""
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        """
        生成掩码
        :return: 一维掩码数组，0表示保留，1表示掩码
        """
        # 每帧的掩码：0表示保留，1表示掩码
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),  # 保留的patch
            np.ones(self.num_masks_per_frame),  # 掩码的patch
        ])
        np.random.shuffle(mask_per_frame)  # 随机打乱
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()  # 扩展到所有帧
        return mask


if __name__ == '__main__':
    """主程序入口"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--dataset', type=str, default='EMER', help='input dataset')  # 数据集名称
    parser.add_argument('--feature_level', type=str, default='UTTERANCE', help='feature level [FRAME or UTTERANCE]') # 特征级别
    parser.add_argument('--pretrain_model', type=str, default='VoxCeleb_ckp49', help='pth of pretrain MAE model') # 预训练MAE模型的路径
    parser.add_argument('--feature_name', type=str, default='VoxCeleb_ckp49', help='pth of pretrain MAE model')
    parser.add_argument('--device', default='cuda:1',
                        help='device to use for training / testing') # 设备
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train') # 模型名称
    parser.add_argument('--nb_classes', default=6, type=int, help='number of the classification types') # 分类类别数量
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--global_pool', action='store_true') # # 使用全局池化
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification') # 使用类别token代替全局池化
    parser.add_argument('--batch_size', default=1, type=int) # 批处理大小

    params = parser.parse_args()  # 解析参数

    print(f'==> Extracting maeVideo embedding...')
    # 根据数据集设置路径
    if params.dataset == "MER2023":
        face_dir = "/home/amax/big_space/datasets/MER2023/dataset-process/openface_face"  # 人脸数据目录
        save_dir = "/home/amax/big_space/datasets/MER2023/dataset-process/features_tmp/maeV_199_UTT"  # 特征保存目录
        list_file = "/home/amax/big_space/datasets/list_files/MER2023_NCEV.txt"  # 数据列表文件
    elif params.dataset == "EMER":
        face_dir = "/home/amax/big_space/datasets/MER2024/EMER/all_face"  # 人脸数据目录
        save_dir = "/home/amax/big_space/datasets/MER2024/EMER/features_tmp/maeV_199_UTT"  # 特征保存目录
        list_file = "/home/amax/big_space/datasets/list_files/EMER_332_NCE.txt"  # 数据列表文件
    elif params.dataset == "MER2024":
        face_dir = "/home/amax/big_space/datasets/MER2024/dataset-process/all_face"  # 人脸数据目录
        save_dir = "/home/amax/big_space/datasets/MER2024/dataset-process/features_tmp/maeV_199_UTT"  # 特征保存目录
        list_file = "/home/amax/big_space/datasets/list_files/MER2024_12065_NCE.txt"  # 数据列表文件
    elif params.dataset == "MER2024_20000":
        face_dir = "/home/amax/big_space/datasets/MER2024/dataset-process/all_face"  # 人脸数据目录
        save_dir = "/home/amax/big_space/datasets/MER2024/dataset-process/features_20000_tmp/maeV_199_UTT"  # 特征保存目录
        list_file = "/home/amax/big_space/datasets/list_files/MER2024_candidate_20000.txt"  # 数据列表文件
    elif params.dataset == "DFEW":
        face_dir = "/home/amax/big_space/datasets/DFEW/dataset-process/openface_face"  # 人脸数据目录
        save_dir = "/home/amax/big_space/datasets/DFEW/dataset-process/features_tmp/maeV_199_UTT"  # 特征保存目录
        list_file = "/home/amax/big_space/datasets/list_files/DFEW_set_1_train.txt"  # 数据列表文件
        # list_file = "/home/amax/big_space/datasets/list_files/DFEW_set_1_test.txt"  # 可选测试集列表
    if not os.path.exists(save_dir): os.makedirs(save_dir)  # 创建保存目录


    # 加载模型
    model = vit_large_patch16_224()  # 初始化ViT模型

    if True:
        # 加载预训练模型
        checkpoint_file = "/home/amax/project/MER2024/MER2024-Baseline/pretrained_models/maeVideo/maeVideo_ckp199.pth"
        print("Load pre-trained checkpoint from: %s" % checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location='cpu')  # 加载检查点

        # 检查模型键
        checkpoint_model = None
        for model_key in 'model|module'.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]  # 获取模型权重
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint  # 直接使用检查点

        # 处理模型权重
        state_dict = model.state_dict()  # 获取当前模型的状态字典
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]  # 删除不匹配的键

        # 重构权重字典
        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]  # 移除backbone前缀
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]  # 移除encoder前缀
            else:
                new_dict[key] = checkpoint_model[key]  # 直接保留
        checkpoint_model = new_dict  # 更新检查点模型

        # 插值位置嵌入
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']  # 获取预训练的位置嵌入
            embedding_size = pos_embed_checkpoint.shape[-1]  # 嵌入维度
            num_patches = model.patch_embed.num_patches  # patch数量
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 额外token数量

            # 原始位置嵌入的高度和宽度
            orig_size = int(
                ((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (16 // model.patch_embed.tubelet_size)) ** 0.5)
            # 新位置嵌入的高度和宽度
            new_size = int((num_patches // (16 // model.patch_embed.tubelet_size)) ** 0.5)
            # 类别token和距离token保持不变
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]  # 提取额外token
                # 仅插值位置token
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # 重塑张量以便插值
                pos_tokens = pos_tokens.reshape(-1, 16 // model.patch_embed.tubelet_size, orig_size, orig_size,
                                                embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                # 双三次插值
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                # 重塑回原始形状
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, 16 // model.patch_embed.tubelet_size, new_size,
                                                                    new_size, embedding_size)
                pos_tokens = pos_tokens.flatten(1, 3)  # 展平
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)  # 合并额外token和插值后的token
                checkpoint_model['pos_embed'] = new_pos_embed  # 更新位置嵌入

        load_state_dict(model, checkpoint_model, prefix='')  # 加载权重
        # trunc_normal_(model.head.weight, std=2e-5)  # 可选：初始化头部权重

    device = torch.device(params.device)  # 设置设备

    model.to(device)  # 将模型移动到设备
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[1])  # 多GPU并行

    # 计算模型的可训练参数数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 加载数据集
    dataset = test_data_loader(list_file, face_dir)  # 使用测试数据加载器
    # dataset = train_data_loader(list_file, face_dir)  # 可选：使用训练数据加载器

    # 创建数据加载器
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,  # 批处理大小
        num_workers=10,  # 数据加载线程数
        drop_last=True,  # 丢弃最后不完整的批次
    )

    # 处理每个视频
    i = 1  # 当前视频索引
    vids = len(data_loader)  # 总视频数量
    for images, video_name in data_loader:
        print(f"Processing video ' ({i}/{vids})...")  # 打印进度
        i = i + 1
        images = images.to(device)  # 将图像移动到设备
        embedding = model(images)  # 提取特征嵌入

        print("embedding :", embedding.shape)  # 打印嵌入形状
        embedding = embedding.cpu().detach().numpy()  # 转换为NumPy数组

        # 保存结果
        EMBEDDING_DIM = max(-1, np.shape(embedding)[-1])  # 获取嵌入维度

        video_name = video_name[0]  # 获取视频名称

        csv_file = os.path.join(save_dir, f'{video_name}.npy')  # 保存路径
        if params.feature_level == 'FRAME':  # 帧级别特征
            embedding = np.array(embedding).squeeze()  # 去除冗余维度
            if len(embedding) == 0:  # 处理空嵌入
                embedding = np.zeros((1, EMBEDDING_DIM))  # 填充零
            elif len(embedding.shape) == 1:  # 处理一维嵌入
                embedding = embedding[np.newaxis, :]  # 增加维度
            np.save(csv_file, embedding)  # 保存为.npy文件
        elif params.feature_level == 'BLK':  # 块级别特征
            embedding = np.array(embedding)
            if len(embedding) == 0:  # 处理空嵌入
                embedding = np.zeros((257, EMBEDDING_DIM))  # 填充零
            elif len(embedding.shape) == 3:  # 处理三维嵌入
                embedding = np.mean(embedding, axis=0)  # 沿时间维度平均
            np.save(csv_file, embedding)  # 保存为.npy文件
        else:  # 其他特征级别
            embedding = np.array(embedding).squeeze()  # 去除冗余维度
            if len(embedding) == 0:  # 处理空嵌入
                embedding = np.zeros((EMBEDDING_DIM,))  # 填充零
            elif len(embedding.shape) == 2:  # 处理二维嵌入
                embedding = np.mean(embedding, axis=0)  # 沿时间维度平均
            print("csv_file: ", csv_file)  # 打印保存路径
            print("embedding: ", embedding)  # 打印嵌入内容
            np.save(csv_file, embedding)  # 保存为.npy文件

# 以下是不同数据集的运行示例
# MER2023
# python -u extract_maeVideo_embedding.py    --dataset='MER2023' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='maeVideo_ckp199' --feature_name='maeVideo'

# EMER
# python -u extract_maeVideo_embedding.py    --dataset='EMER' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='maeVideo_ckp199' --feature_name='maeVideo'

# MER2024
# python -u extract_maeVideo_embedding.py    --dataset='MER2024' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='maeVideo_ckp199' --feature_name='maeVideo'
# MER2024_20000
# python -u extract_maeVideo_embedding.py    --dataset='MER2024_20000' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='maeVideo_ckp199' --feature_name='maeVideo'

# DFEW
# python -u extract_maeVideo_embedding.py    --dataset='DFEW' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='maeVideo_ckp199' --feature_name='maeVideo'