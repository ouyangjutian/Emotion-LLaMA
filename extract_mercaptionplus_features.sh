#!/bin/bash
# 提取MERCaptionPlus数据集的AU特征
# 使用summary_description字段（纯净的assistant描述）

# 激活conda环境
# source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama

cd /home/project/Emotion-LLaMA

# 配置路径
DATASET="mer2023"
MER_FACTORY_OUTPUT="/home/project/MER-Factory/output/MERCaptionPlus"
CSV_PATH="/home/project/Dataset/Emotion/MER2025/dataset/mer2025-dataset/track2_train_mercaptionplus.csv"
SAVE_ROOT="./preextracted_features"
DEVICE="cuda:2"
video_root="/home/project/Dataset/Emotion/MER2025/dataset/mer2025-dataset/video"

echo "🚀 开始提取MERCaptionPlus AU特征..."
echo "📂 MER-Factory输出: $MER_FACTORY_OUTPUT"
echo "📊 CSV文件: $CSV_PATH"
echo "💾 保存目录: $SAVE_ROOT"
echo ""

# 提取AU特征（仅AU模态）
python extract_multimodal_features_precompute.py \
    --dataset $DATASET \
    --modality au \
    --device $DEVICE \
    --mer-factory-output $MER_FACTORY_OUTPUT \
    --csv_path $CSV_PATH \
    --csv_column name \
    --save_root $SAVE_ROOT

echo ""
echo "✅ AU特征提取完成！"
echo "📁 输出目录: $SAVE_ROOT/$DATASET/au/"
echo ""
