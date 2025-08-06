# OwLore 模型评估指南

## 问题背景

OwLore 使用了特殊的低秩训练方法（loro、elsa、cola、lora），这些方法修改了模型的结构，导致保存的模型权重格式与标准的 transformers 不兼容。因此，无法直接使用标准的 lm-evaluation-harness 来评估这些模型。

## 解决方案

我们创建了专门的评估脚本来处理不同的 OwLore 训练方法：

### 1. 主要脚本文件

- `eval_owlore_model.py`: 核心模型加载器，支持所有 OwLore 方法
- `run_owlore_eval.py`: 完整的评估脚本，集成了 lm_eval

### 2. 支持的训练方法

- **loro**: 低秩适配器参数化
- **elsa**: 混合低秩稀疏适配器
- **cola**: COLA 混合适配器
- **lora**: LoRA 混合适配器

## 使用方法

### 基本语法

```bash
python run_owlore_eval.py \\
    --model_path <模型路径> \\
    --finetune_method <训练方法> \\
    --adapter_scope <适配器范围> \\
    --rank <低秩维度> \\
    --alpha <缩放因子> \\
    --tasks <评估任务> \\
    --num_fewshot <少样本数量> \\
    --batch_size <批大小> \\
    --limit <样本限制>
```

### 不同方法的调用示例

#### 1. LORO 方法
```bash
python run_owlore_eval.py \\
    --model_path /path/to/output_models/my_finetuned_model/checkpoint-60/complete_model \\
    --finetune_method loro \\
    --adapter_scope all \\
    --rank 6 \\
    --alpha 12 \\
    --tasks boolq,piqa,hellaswag \\
    --num_fewshot 5 \\
    --batch_size auto
```

#### 2. ELSA 方法
```bash
python run_owlore_eval.py \\
    --model_path /path/to/output_models/my_finetuned_model/checkpoint-60/complete_model \\
    --finetune_method elsa \\
    --adapter_scope all \\
    --rank 6 \\
    --alpha 12 \\
    --tasks boolq,piqa,hellaswag \\
    --num_fewshot 5 \\
    --batch_size auto
```

#### 3. COLA 方法
```bash
python run_owlore_eval.py \\
    --model_path /path/to/output_models/my_finetuned_model/checkpoint-60/complete_model \\
    --finetune_method cola \\
    --adapter_scope all \\
    --rank 6 \\
    --alpha 12 \\
    --tasks boolq,piqa,hellaswag \\
    --num_fewshot 5 \\
    --batch_size auto
```

#### 4. LORA 方法
```bash
python run_owlore_eval.py \\
    --model_path /path/to/output_models/my_finetuned_model/checkpoint-60/complete_model \\
    --finetune_method lora \\
    --adapter_scope all \\
    --rank 6 \\
    --alpha 12 \\
    --tasks boolq,piqa,hellaswag \\
    --num_fewshot 5 \\
    --batch_size auto
```

### 参数说明

- `--model_path`: 训练保存的模型路径，通常是 `checkpoint-XX/complete_model`
- `--finetune_method`: 训练时使用的方法 (loro/elsa/cola/lora)
- `--adapter_scope`: 适配器应用范围，通常是 "all"
- `--rank`: 低秩适配器的维度，与训练时的 `lora_rank` 对应
- `--alpha`: 缩放因子，与训练时的 `lora_alpha_custom` 对应
- `--tasks`: 评估任务，可以是单个任务或逗号分隔的多个任务
- `--num_fewshot`: 少样本学习的样本数量
- `--batch_size`: 批处理大小，可以设置为 "auto"
- `--limit`: 限制每个任务的样本数量（仅用于测试）

### 常用评估任务

- **推理任务**: `boolq`, `piqa`, `hellaswag`, `winogrande`
- **阅读理解**: `arc_easy`, `arc_challenge`, `openbookqa`
- **常识推理**: `social_iqa`

### 环境要求

确保在正确的 conda 环境中运行：
```bash
conda activate owlore_eval
```

### 输出结果

评估结果会保存在 `--output_path` 指定的目录中，包含：
- 聚合结果的 JSON 文件
- 每个样本的详细结果 JSONL 文件

## 注意事项

1. **模型路径**: 确保使用 `complete_model` 目录下的模型，而不是根目录
2. **参数匹配**: `rank` 和 `alpha` 必须与训练时的参数完全匹配
3. **内存使用**: 大模型评估可能需要较多 GPU 内存
4. **批大小**: 如果遇到内存不足，可以减小 `batch_size`

## 故障排除

如果遇到模型加载错误：
1. 检查模型路径是否正确
2. 确认训练方法参数是否匹配
3. 验证 rank 和 alpha 参数是否与训练时一致
4. 确保在正确的 conda 环境中运行