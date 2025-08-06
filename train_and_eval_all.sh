#!/bin/bash
# OwLore 训练和评估脚本
# 简单直接，挨个执行训练和评估

echo "🎯 OwLore 完整实验开始"
echo "📅 开始时间: $(date)"
echo ""

# 激活conda环境的函数
activate_conda() {
    source /home/rtx3090/miniconda3/etc/profile.d/conda.sh
    conda activate $1
    echo "✅ 激活环境: $1"
}

# 1. LoRA 模型
echo "============================================================"
echo "🎯 开始训练和评估 LoRA 模型"
echo "============================================================"

echo "🏋️ 训练 LoRA 模型..."
activate_conda owlore_fine
python examples/finetune.py \
    --model_name_or_path "meta-llama/Llama-3.2-1B" \
    --dataset_path "data/OwLore_Dataset/merge" \
    --output_dir "output_models/my_finetuned_model_lora" \
    --overwrite_output_dir \
    --num_train_epochs 1.0 \
    --learning_rate 3e-4 \
    --block_size 512 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --run_name "LoRA_training" \
    --optim adamw_hf \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 10000 \
    --dataloader_num_workers 1 \
    --gradient_checkpointing False \
    --use_flash_attention 1 \
    --use_lisa 0 \
    --disable_group_texts 0 \
    --seed 111 \
    --galore False \
    --lora_rank 8 \
    --lora_alpha_custom 16 \
    --cola_silu True \
    --cola_init False \
    --adapter_scope all \
    --sparsity 0.001 \
    --gamma 0.7 \
    --sparse_method svd \
    --sparse_svd_rank 256 \
    --svd_inverse False \
    --finetune_method lora

if [ $? -eq 0 ]; then
    echo "✅ LoRA 训练完成"
    echo "📊 评估 LoRA 模型..."
    activate_conda owlore_eval
    python run_owlore_eval.py \
        --model_path "output_models/my_finetuned_model_lora/complete_model" \
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
else
    echo "❌ LoRA 训练失败，跳过评估"
fi

echo ""
echo "😴 休息30秒..."
sleep 30

# 2. ELSA_SiLU_True 模型
echo "============================================================"
echo "🎯 开始训练和评估 ELSA_SiLU_True 模型"
echo "============================================================"

echo "🏋️ 训练 ELSA_SiLU_True 模型..."
activate_conda owlore_fine
python examples/finetune.py \
    --model_name_or_path "meta-llama/Llama-3.2-1B" \
    --dataset_path "data/OwLore_Dataset/merge" \
    --output_dir "output_models/my_finetuned_model_elsa_silutrue" \
    --overwrite_output_dir \
    --num_train_epochs 1.0 \
    --learning_rate 3e-4 \
    --block_size 512 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --run_name "ELSA_SiLU_True_training" \
    --optim adamw_hf \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 10000 \
    --dataloader_num_workers 1 \
    --gradient_checkpointing False \
    --use_flash_attention 1 \
    --use_lisa 0 \
    --disable_group_texts 0 \
    --seed 111 \
    --galore False \
    --lora_rank 7 \
    --lora_alpha_custom 14 \
    --cola_silu True \
    --cola_init False \
    --adapter_scope all \
    --sparsity 0.001 \
    --gamma 0.7 \
    --sparse_method svd \
    --sparse_svd_rank 256 \
    --svd_inverse False \
    --finetune_method elsa

if [ $? -eq 0 ]; then
    echo "✅ ELSA_SiLU_True 训练完成"
    echo "📊 评估 ELSA_SiLU_True 模型..."
    activate_conda owlore_eval
    python run_owlore_eval.py \
        --model_path "output_models/my_finetuned_model_elsa_silutrue/complete_model" \
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
    echo "❌ ELSA_SiLU_True 训练失败，跳过评估"
fi

echo ""
echo "😴 休息30秒..."
sleep 30

# 3. ELSA_SiLU_False 模型
echo "============================================================"
echo "🎯 开始训练和评估 ELSA_SiLU_False 模型"
echo "============================================================"

echo "🏋️ 训练 ELSA_SiLU_False 模型..."
activate_conda owlore_fine
python examples/finetune.py \
    --model_name_or_path "meta-llama/Llama-3.2-1B" \
    --dataset_path "data/OwLore_Dataset/merge" \
    --output_dir "output_models/my_finetuned_model_elsa_silufalse" \
    --overwrite_output_dir \
    --num_train_epochs 1.0 \
    --learning_rate 3e-4 \
    --block_size 512 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --run_name "ELSA_SiLU_False_training" \
    --optim adamw_hf \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 10000 \
    --dataloader_num_workers 1 \
    --gradient_checkpointing False \
    --use_flash_attention 1 \
    --use_lisa 0 \
    --disable_group_texts 0 \
    --seed 111 \
    --galore False \
    --lora_rank 7 \
    --lora_alpha_custom 14 \
    --cola_silu False \
    --cola_init False \
    --adapter_scope all \
    --sparsity 0.001 \
    --gamma 0.7 \
    --sparse_method svd \
    --sparse_svd_rank 256 \
    --svd_inverse False \
    --finetune_method elsa

if [ $? -eq 0 ]; then
    echo "✅ ELSA_SiLU_False 训练完成"
    echo "📊 评估 ELSA_SiLU_False 模型..."
    activate_conda owlore_eval
    python run_owlore_eval.py \
        --model_path "output_models/my_finetuned_model_elsa_silufalse/complete_model" \
        --finetune_method elsa \
        --adapter_scope all \
        --rank 7 \
        --alpha 14 \
        --sparsity 0.001 \
        --gamma 0.7 \
        --sparse_method svd \
        --sparse_svd_rank 256 \
        --cola_silu false \
        --cola_init false \
        --svd_inverse false \
        --tasks "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa" \
        --output_path "final_results/ELSA_SiLU_False_results" \
        --num_fewshot 5 \
        --batch_size auto
    
    if [ $? -eq 0 ]; then
        echo "✅ ELSA_SiLU_False 评估完成"
    else
        echo "❌ ELSA_SiLU_False 评估失败"
    fi
else
    echo "❌ ELSA_SiLU_False 训练失败，跳过评估"
fi

echo ""
echo "😴 休息30秒..."
sleep 30

# 4. COLA 模型
echo "============================================================"
echo "🎯 开始训练和评估 COLA 模型"
echo "============================================================"

echo "🏋️ 训练 COLA 模型..."
activate_conda owlore_fine
python examples/finetune.py \
    --model_name_or_path "meta-llama/Llama-3.2-1B" \
    --dataset_path "data/OwLore_Dataset/merge" \
    --output_dir "output_models/my_finetuned_model_cola" \
    --overwrite_output_dir \
    --num_train_epochs 1.0 \
    --learning_rate 3e-4 \
    --block_size 512 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --run_name "COLA_training" \
    --optim adamw_hf \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 10000 \
    --dataloader_num_workers 1 \
    --gradient_checkpointing False \
    --use_flash_attention 1 \
    --use_lisa 0 \
    --disable_group_texts 0 \
    --seed 111 \
    --galore False \
    --lora_rank 8 \
    --lora_alpha_custom 16 \
    --cola_silu True \
    --cola_init True \
    --adapter_scope all \
    --sparsity 0.001 \
    --gamma 0.7 \
    --sparse_method svd \
    --sparse_svd_rank 256 \
    --svd_inverse False \
    --finetune_method cola

if [ $? -eq 0 ]; then
    echo "✅ COLA 训练完成"
    echo "📊 评估 COLA 模型..."
    activate_conda owlore_eval
    python run_owlore_eval.py \
        --model_path "output_models/my_finetuned_model_cola/complete_model" \
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
    echo "❌ COLA 训练失败，跳过评估"
fi

echo ""
echo "😴 休息30秒..."
sleep 30

# 5. LORO 模型
echo "============================================================"
echo "🎯 开始训练和评估 LORO 模型"
echo "============================================================"

echo "🏋️ 训练 LORO 模型..."
activate_conda owlore_fine
python examples/finetune.py \
    --model_name_or_path "meta-llama/Llama-3.2-1B" \
    --dataset_path "data/OwLore_Dataset/merge" \
    --output_dir "output_models/my_finetuned_model_loro" \
    --overwrite_output_dir \
    --num_train_epochs 1.0 \
    --learning_rate 3e-4 \
    --block_size 512 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --run_name "LORO_training" \
    --optim adamw_hf \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 10000 \
    --dataloader_num_workers 1 \
    --gradient_checkpointing False \
    --use_flash_attention 1 \
    --use_lisa 0 \
    --disable_group_texts 0 \
    --seed 111 \
    --galore False \
    --lora_rank 8 \
    --lora_alpha_custom 16 \
    --cola_silu True \
    --cola_init False \
    --adapter_scope all \
    --sparsity 0.001 \
    --gamma 0.7 \
    --sparse_method svd \
    --sparse_svd_rank 256 \
    --svd_inverse False \
    --finetune_method loro

if [ $? -eq 0 ]; then
    echo "✅ LORO 训练完成"
    echo "📊 评估 LORO 模型..."
    activate_conda owlore_eval
    python run_owlore_eval.py \
        --model_path "output_models/my_finetuned_model_loro/complete_model" \
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
    echo "❌ LORO 训练失败，跳过评估"
fi

echo ""
echo "🎉 所有实验完成!"
echo "📅 完成时间: $(date)"
echo "📁 查看结果:"
echo "  - 训练模型: output_models/"
echo "  - 评估结果: final_results/"