#!/bin/bash
# OwLore merge数据集训练模型的评估脚本
# 直接评估已训练好的模型

echo "🎯 开始评估merge数据集训练的所有模型"
echo "📅 开始时间: $(date)"
echo ""

# 激活conda环境的函数
activate_conda() {
    source /home/rtx3090/miniconda3/etc/profile.d/conda.sh
    conda activate $1
    echo "✅ 激活环境: $1"
}

# 激活评估环境
activate_conda owlore_eval

# 1. 评估 LoRA 模型
echo "============================================================"
echo "📊 评估 LoRA 模型"
echo "============================================================"

# 检查是否有最终模型文件，否则使用最新checkpoint
if [ -f "output_models/my_finetuned_model_lora/model.safetensors" ]; then
    model_path="output_models/my_finetuned_model_lora"
    echo "使用最终模型: $model_path"
elif [ -d "output_models/my_finetuned_model_lora/checkpoint-1739" ]; then
    model_path="output_models/my_finetuned_model_lora/checkpoint-1739"
    echo "使用checkpoint: $model_path"
elif [ -d "output_models/my_finetuned_model_lora/checkpoint-60" ]; then
    model_path="output_models/my_finetuned_model_lora/checkpoint-60"
    echo "使用checkpoint: $model_path"
else
    echo "❌ LoRA 模型文件不存在"
    model_path=""
fi

if [ -n "$model_path" ]; then
    python run_owlore_eval.py \
        --model_path "$model_path" \
        --finetune_method lora \
        --adapter_scope all \
        --rank 8 \
        --alpha 16 \
        --sparsity 0.001 \
        --gamma 0.7 \
        --sparse_method svd \
        --sparse_svd_rank 256 \
        --cola_silu true \
        --cola_init false \
        --svd_inverse false \
        --tasks "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa" \
        --output_path "final_results/LoRA_results" \
        --num_fewshot 5 \
        --batch_size auto
    
    if [ $? -eq 0 ]; then
        echo "✅ LoRA 评估完成"
    else
        echo "❌ LoRA 评估失败"
    fi
fi

echo ""
echo "😴 休息10秒..."
sleep 10

# 2. 评估 ELSA_SiLU_True 模型
echo "============================================================"
echo "📊 评估 ELSA_SiLU_True 模型"
echo "============================================================"

if [ -d "output_models/my_finetuned_model_elsa/checkpoint-60" ]; then
    python run_owlore_eval.py \
        --model_path "output_models/my_finetuned_model_elsa/checkpoint-60" \
        --finetune_method elsa \
        --adapter_scope all \
        --rank 7 \
        --alpha 14 \
        --sparsity 0.001 \
        --gamma 0.7 \
        --sparse_method svd \
        --sparse_svd_rank 256 \
        --cola_silu true \
        --cola_init false \
        --svd_inverse false \
        --tasks "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa" \
        --output_path "final_results/ELSA_SiLU_True_results" \
        --num_fewshot 5 \
        --batch_size auto
    
    if [ $? -eq 0 ]; then
        echo "✅ ELSA_SiLU_True 评估完成"
    else
        echo "❌ ELSA_SiLU_True 评估失败"
    fi
else
    echo "❌ ELSA_SiLU_True 模型目录不存在: output_models/my_finetuned_model_elsa/checkpoint-60"
fi

echo ""
echo "😴 休息10秒..."
sleep 10

# 3. 评估 COLA 模型
echo "============================================================"
echo "📊 评估 COLA 模型"
echo "============================================================"

if [ -f "output_models/my_finetuned_model_cola/model.safetensors" ]; then
    python run_owlore_eval.py \
        --model_path "output_models/my_finetuned_model_cola" \
        --finetune_method cola \
        --adapter_scope all \
        --rank 8 \
        --alpha 16 \
        --sparsity 0.001 \
        --gamma 0.7 \
        --sparse_method svd \
        --sparse_svd_rank 256 \
        --cola_silu true \
        --cola_init true \
        --svd_inverse false \
        --tasks "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa" \
        --output_path "final_results/COLA_results" \
        --num_fewshot 5 \
        --batch_size auto
    
    if [ $? -eq 0 ]; then
        echo "✅ COLA 评估完成"
    else
        echo "❌ COLA 评估失败"
    fi
else
    echo "❌ COLA 模型文件不存在: output_models/my_finetuned_model_cola/model.safetensors"
fi

echo ""
echo "😴 休息10秒..."
sleep 10

# 4. 评估 LORO 模型
echo "============================================================"
echo "📊 评估 LORO 模型"
echo "============================================================"

if [ -d "output_models/my_finetuned_model/checkpoint-60" ]; then
    python run_owlore_eval.py \
        --model_path "output_models/my_finetuned_model/checkpoint-60" \
        --finetune_method loro \
        --adapter_scope all \
        --rank 8 \
        --alpha 16 \
        --sparsity 0.001 \
        --gamma 0.7 \
        --sparse_method svd \
        --sparse_svd_rank 256 \
        --cola_silu true \
        --cola_init false \
        --svd_inverse false \
        --tasks "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa" \
        --output_path "final_results/LORO_results" \
        --num_fewshot 5 \
        --batch_size auto
    
    if [ $? -eq 0 ]; then
        echo "✅ LORO 评估完成"
    else
        echo "❌ LORO 评估失败"
    fi
else
    echo "❌ LORO 模型目录不存在: output_models/my_finetuned_model/checkpoint-60"
fi

echo ""
echo "🎉 所有模型评估完成!"
echo "📅 完成时间: $(date)"
echo "📁 查看结果:"
echo "  - 评估结果: final_results/"
echo ""
echo "📊 结果文件:"
ls -la final_results/ 2>/dev/null || echo "final_results目录不存在"