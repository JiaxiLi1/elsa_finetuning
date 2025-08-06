#!/bin/bash
# OwLore merge数据集所有模型评估脚本

echo "🎯 开始评估所有merge数据集训练的模型"
echo "📅 开始时间: $(date)"
echo ""

# 激活评估环境
source /home/rtx3090/miniconda3/etc/profile.d/conda.sh
conda activate owlore_eval
echo "✅ 激活环境: owlore_eval"
echo ""

# 创建结果目录
mkdir -p final_results

# 1. 评估 LoRA 模型
echo "============================================================"
echo "📊 评估 LoRA 模型"
echo "============================================================"
python run_owlore_eval.py \
    --model_path "output_models/my_finetuned_model_lora/checkpoint-1739/complete_model" \
    --finetune_method lora \
    --adapter_scope all \
    --rank 8 \
    --alpha 16 \
    --sparsity 0.001 \
    --gamma 0.7 \
    --sparse_method svd \
    --sparse_svd_rank 256 \
    --tasks "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa" \
    --output_path "final_results/LoRA_results" \
    --num_fewshot 5 \
    --batch_size auto

if [ $? -eq 0 ]; then
    echo "✅ LoRA 评估完成"
else
    echo "❌ LoRA 评估失败"
fi

echo ""
echo "😴 休息10秒..."
sleep 10

# 2. 评估 LORO 模型
echo "============================================================"
echo "📊 评估 LORO 模型"
echo "============================================================"
python run_owlore_eval.py \
    --model_path "output_models/my_finetuned_model_loro/checkpoint-1739/complete_model" \
    --finetune_method loro \
    --adapter_scope all \
    --rank 8 \
    --alpha 16 \
    --sparsity 0.001 \
    --gamma 0.7 \
    --sparse_method svd \
    --sparse_svd_rank 256 \
    --tasks "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa" \
    --output_path "final_results/LORO_results" \
    --num_fewshot 5 \
    --batch_size auto

if [ $? -eq 0 ]; then
    echo "✅ LORO 评估完成"
else
    echo "❌ LORO 评估失败"
fi

echo ""
echo "😴 休息10秒..."
sleep 10

# 3. 评估 ELSA_SiLU_False 模型
echo "============================================================"
echo "📊 评估 ELSA_SiLU_False 模型"
echo "============================================================"
python run_owlore_eval.py \
    --model_path "output_models/my_finetuned_model_elsa_silufalse/checkpoint-1739/complete_model" \
    --finetune_method elsa \
    --adapter_scope all \
    --rank 7 \
    --alpha 14 \
    --sparsity 0.001 \
    --gamma 0.7 \
    --sparse_method svd \
    --sparse_svd_rank 256 \
    --no-cola_silu \
    --tasks "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa" \
    --output_path "final_results/ELSA_SiLU_False_results" \
    --num_fewshot 5 \
    --batch_size auto

if [ $? -eq 0 ]; then
    echo "✅ ELSA_SiLU_False 评估完成"
else
    echo "❌ ELSA_SiLU_False 评估失败"
fi

echo ""
echo "😴 休息10秒..."
sleep 10

# 4. 评估 ELSA_SiLU_True 模型
echo "============================================================"
echo "📊 评估 ELSA_SiLU_True 模型"
echo "============================================================"
python run_owlore_eval.py \
    --model_path "output_models/my_finetuned_model_elsa_silutrue/checkpoint-1739/complete_model" \
    --finetune_method elsa \
    --adapter_scope all \
    --rank 7 \
    --alpha 14 \
    --sparsity 0.001 \
    --gamma 0.7 \
    --sparse_method svd \
    --sparse_svd_rank 256 \
    --cola_silu \
    --tasks "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa" \
    --output_path "final_results/ELSA_SiLU_True_results" \
    --num_fewshot 5 \
    --batch_size auto

if [ $? -eq 0 ]; then
    echo "✅ ELSA_SiLU_True 评估完成"
else
    echo "❌ ELSA_SiLU_True 评估失败"
fi

echo ""
echo "😴 休息10秒..."
sleep 10

# 5. 评估 COLA 模型
echo "============================================================"
echo "📊 评估 COLA 模型"
echo "============================================================"
python run_owlore_eval.py \
    --model_path "output_models/my_finetuned_model_cola/checkpoint-1739/complete_model" \
    --finetune_method cola \
    --adapter_scope all \
    --rank 8 \
    --alpha 16 \
    --sparsity 0.001 \
    --gamma 0.7 \
    --sparse_method svd \
    --sparse_svd_rank 256 \
    --cola_silu \
    --cola_init \
    --tasks "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa" \
    --output_path "final_results/COLA_results" \
    --num_fewshot 5 \
    --batch_size auto

if [ $? -eq 0 ]; then
    echo "✅ COLA 评估完成"
else
    echo "❌ COLA 评估失败"
fi

echo ""
echo "🎉 所有模型评估完成!"
echo "📅 完成时间: $(date)"
echo ""
echo "📊 评估结果汇总:"
echo "  - LoRA: final_results/LoRA_results"
echo "  - LORO: final_results/LORO_results"
echo "  - ELSA_SiLU_False: final_results/ELSA_SiLU_False_results"
echo "  - ELSA_SiLU_True: final_results/ELSA_SiLU_True_results"
echo "  - COLA: final_results/COLA_results"
echo ""
echo "📁 查看结果文件:"
ls -la final_results/ 2>/dev/null || echo "final_results目录为空"