"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os  # 操作系统模块
import logging  # 日志模块
import contextlib  # 上下文管理模块

from omegaconf import OmegaConf  # 配置管理工具
import numpy as np  # 数值计算库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
from transformers import LlamaTokenizer  # Llama分词器
from peft import (
    LoraConfig,  # LoRA配置
    get_peft_model,  # 获取LoRA模型
    prepare_model_for_int8_training,  # 准备模型用于int8训练
)

from minigpt4.common.dist_utils import download_cached_file  # 下载缓存文件工具
from minigpt4.common.utils import get_abs_path, is_url  # 工具函数
from minigpt4.models.eva_vit import create_eva_vit_g  # 创建EVA视觉模型
from minigpt4.models.modeling_llama import LlamaForCausalLM  # Llama因果语言模型



class BaseModel(nn.Module):
    """Base class for models."""

    def __init__(self):
        super().__init__()  # 调用父类构造函数

    @property
    def device(self):
        return list(self.parameters())[-1].device  # 返回模型参数所在的设备

    def load_checkpoint(self, url_or_filename):
        """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """

        if is_url(url_or_filename):  # 检查是否为URL
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True  # 下载缓存文件
            )
            checkpoint = torch.load(cached_file, map_location="cpu")  # 加载检查点到CPU
        elif os.path.isfile(url_or_filename):  # 检查是否为本地文件
            checkpoint = torch.load(url_or_filename, map_location="cpu")  # 加载检查点到CPU
        else:
            raise RuntimeError("checkpoint url or path is invalid")  # 抛出异常

        if "model" in checkpoint.keys():  # 检查检查点是否包含模型状态
            state_dict = checkpoint["model"]  # 获取模型状态字典
        else:
            state_dict = checkpoint  # 直接使用检查点

        msg = self.load_state_dict(state_dict, strict=False)  # 加载状态字典

        logging.info("Missing keys {}".format(msg.missing_keys))  # 记录缺失的键
        logging.info("load checkpoint from %s" % url_or_filename)  # 记录加载来源

        return msg  # 返回加载消息

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Build a pretrained model from default configuration file, specified by model_type.

        Args:
            - model_type (str): model type, specifying architecture and checkpoints.

        Returns:
            - model (nn.Module): pretrained or finetuned model, depending on the configuration.
        """
        model_cfg = OmegaConf.load(cls.default_config_path(model_type)).model  # 加载默认配置
        model = cls.from_config(model_cfg)  # 根据配置构建模型

        return model  # 返回模型

    @classmethod
    def default_config_path(cls, model_type):
        assert (
            model_type in cls.PRETRAINED_MODEL_CONFIG_DICT
        ), "Unknown model type {}".format(model_type)  # 检查模型类型是否有效
        return get_abs_path(cls.PRETRAINED_MODEL_CONFIG_DICT[model_type])  # 返回配置路径

    def load_checkpoint_from_config(self, cfg, **kwargs):
        """
        Load checkpoint as specified in the config file.

        If load_finetuned is True, load the finetuned model; otherwise, load the pretrained model.
        When loading the pretrained model, each task-specific architecture may define their
        own load_from_pretrained() method.
        """
        load_finetuned = cfg.get("load_finetuned", True)  # 获取是否加载微调模型的配置
        if load_finetuned:  # 如果加载微调模型
            finetune_path = cfg.get("finetuned", None)  # 获取微调模型路径
            assert (
                finetune_path is not None
            ), "Found load_finetuned is True, but finetune_path is None."  # 检查路径是否有效
            self.load_checkpoint(url_or_filename=finetune_path)  # 加载微调模型
        else:  # 如果加载预训练模型
            # load pre-trained weights
            pretrain_path = cfg.get("pretrained", None)  # 获取预训练模型路径
            assert "Found load_finetuned is False, but pretrain_path is None."  # 检查路径是否有效
            self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)  # 加载预训练模型

    def before_evaluation(self, **kwargs):
        pass  # 评估前的钩子方法

    def show_n_params(self, return_str=True):
        tot = 0  # 参数总数
        for p in self.parameters():  # 遍历所有参数
            w = 1  # 参数数量
            for x in p.shape:  # 遍历参数形状
                w *= x  # 计算参数数量
            tot += w  # 累加参数数量
        if return_str:  # 是否返回字符串
            if tot >= 1e6:  # 百万级参数
                return "{:.1f}M".format(tot / 1e6)  # 返回百万单位
            else:  # 千级参数
                return "{:.1f}K".format(tot / 1e3)  # 返回千单位
        else:  # 返回数值
            return tot  # 返回参数总数

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")  # 检查是否在GPU上

        if enable_autocast:  # 如果启用自动混合精度
            return torch.cuda.amp.autocast(dtype=dtype)  # 返回自动混合精度上下文
        else:  # 如果在CPU上
            return contextlib.nullcontext()  # 返回空上下文

    @classmethod
    def init_vision_encoder(
        cls, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision, freeze
    ):
        logging.info('Loading VIT')  # 记录加载视觉编码器

        assert model_name == "eva_clip_g", "vit model must be eva_clip_g for current version of MiniGPT-4"  # 检查模型名称
        if not freeze:  # 如果未冻结模型
            precision = "fp32"  # fp16 is not for training  # 使用fp32精度训练
        visual_encoder = create_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision  # 视觉编码器参数
        )

        ln_vision = LayerNorm(visual_encoder.num_features)  # 视觉编码器的层归一化

        if freeze:  # 如果冻结模型
            for name, param in visual_encoder.named_parameters():  # 遍历视觉编码器参数
                param.requires_grad = False  # 禁用梯度
            visual_encoder = visual_encoder.eval()  # 设置为评估模式
            visual_encoder.train = disabled_train  # 禁用训练模式
            for name, param in ln_vision.named_parameters():  # 遍历层归一化参数
                param.requires_grad = False  # 禁用梯度
            ln_vision = ln_vision.eval()  # 设置为评估模式
            ln_vision.train = disabled_train  # 禁用训练模式
            logging.info("freeze vision encoder")  # 记录冻结信息

        logging.info('Loading VIT Done')  # 记录加载完成
        return visual_encoder, ln_vision  # 返回视觉编码器和层归一化

    # lora_target_modules=["q_proj","v_proj"], **lora_kargs):
    def init_llm(cls, llama_model_path, low_resource=False, low_res_device=0, lora_r=0,
                 lora_target_modules=["q_proj","k_proj"], **lora_kargs):
        logging.info('Loading LLAMA')  # 记录加载LLAMA
        llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False)  # 加载分词器
        llama_tokenizer.pad_token = "$$"  # 设置填充标记

        if low_resource:  # 如果是低资源模式
            llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,  # 使用fp16
                load_in_8bit=True,  # 加载为8位
                device_map={'': low_res_device}  # 设备映射
            )
        else:  # 正常模式
            llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,  # 使用fp16
            )

        if lora_r > 0:  # 如果启用LoRA
            llama_model = prepare_model_for_int8_training(llama_model)  # 准备模型用于int8训练
            loraconfig = LoraConfig(
                r=lora_r,  # LoRA秩
                bias="none",  # 无偏置
                task_type="CAUSAL_LM",  # 任务类型
                target_modules=lora_target_modules,  # 目标模块
                **lora_kargs  # 其他LoRA参数
            )

            print("loraconfig:", loraconfig)  # 打印LoRA配置
            llama_model = get_peft_model(llama_model, loraconfig)

            llama_model.print_trainable_parameters()

        else:
            for name, param in llama_model.named_parameters():
                param.requires_grad = False
        logging.info('Loading LLAMA Done')
        return llama_model, llama_tokenizer


    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)





