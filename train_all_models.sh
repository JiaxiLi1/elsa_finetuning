#!/bin/bash
# OwLore 批量训练脚本

echo "🚀 开始批量训练OwLore模型..."
echo "开始时间: $(date)"

# 创建日志目录
mkdir -p experiment_logs

# 1. 训练LoRA模型
echo ""
echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
echo "🏋️ 训练LoRA模型 (1/5)"
echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
python examples/finetune.py \
    --model_name_or_path "meta-llama/Llama-3.2-1B" \
    --dataset_path "data/OwLore_Dataset/merge" \
    --output_dir output_models/my_finetuned_model_lora \
    --overwrite_output_dir \
    --num_train_epochs 1.0 \
    --learning_rate 3e-4 \
    --block_size 512 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --run_name my_finetuning_run \
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
    --finetune_method lora \
    2>&1 | tee experiment_logs/train_lora_$(date +%Y%m%d_%H%M%S).log

echo "✅ LoRA模型训练完成"
sleep 30

# 2. 训练ELSA模型 (SiLU=True)
echo ""
echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
echo "🏋️ 训练ELSA模型 SiLU=True (2/5)"
echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
python examples/finetune.py \
    --model_name_or_path "meta-llama/Llama-3.2-1B" \
    --dataset_path "data/OwLore_Dataset/merge" \
    --output_dir output_models/my_finetuned_model_elsa_silutrue \
    --overwrite_output_dir \
    --num_train_epochs 1.0 \
    --learning_rate 3e-4 \
    --block_size 512 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --run_name my_finetuning_run \
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
    --finetune_method elsa \
    2>&1 | tee experiment_logs/train_elsa_silutrue_$(date +%Y%m%d_%H%M%S).log

echo "✅ ELSA SiLU=True模型训练完成"
sleep 30

# 3. 训练ELSA模型 (SiLU=False)
echo ""
echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
echo "🏋️ 训练ELSA模型 SiLU=False (3/5)"
echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
python examples/finetune.py \
    --model_name_or_path "meta-llama/Llama-3.2-1B" \
    --dataset_path "data/OwLore_Dataset/merge" \
    --output_dir output_models/my_finetuned_model_elsa_silufalse \
    --overwrite_output_dir \
    --num_train_epochs 1.0 \
    --learning_rate 3e-4 \
    --block_size 512 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --run_name my_finetuning_run \
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
    --finetune_method elsa \
    2>&1 | tee experiment_logs/train_elsa_silufalse_$(date +%Y%m%d_%H%M%S).log

echo "✅ ELSA SiLU=False模型训练完成"
sleep 30

# 4. 训练COLA模型
echo ""
echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
echo "🏋️ 训练COLA模型 (4/5)"
echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
python examples/finetune.py \
    --model_name_or_path "meta-llama/Llama-3.2-1B" \
    --dataset_path "data/OwLore_Dataset/merge" \
    --output_dir output_models/my_finetuned_model_cola \
    --overwrite_output_dir \
    --num_train_epochs 1.0 \
    --learning_rate 3e-4 \
    --block_size 512 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --run_name my_finetuning_run \
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
    --finetune_method cola \
    2>&1 | tee experiment_logs/train_cola_$(date +%Y%m%d_%H%M%S).log

echo "✅ COLA模型训练完成"
sleep 30

# 5. 训练LORO模型
echo ""
echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
echo "🏋️ 训练LORO模型 (5/5)"
echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
python examples/finetune.py \
    --model_name_or_path "meta-llama/Llama-3.2-1B" \
    --dataset_path "data/OwLore_Dataset/merge" \
    --output_dir output_models/my_finetuned_model_loro \
    --overwrite_output_dir \
    --num_train_epochs 1.0 \
    --learning_rate 3e-4 \
    --block_size 512 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --run_name my_finetuning_run \
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
    --finetune_method loro \
    2>&1 | tee experiment_logs/train_loro_$(date +%Y%m%d_%H%M%S).log

echo "✅ LORO模型训练完成"

echo ""
echo "🎉 所有模型训练完成!"
echo "结束时间: $(date)"
echo "训练日志保存在: experiment_logs/"