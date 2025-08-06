#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OwLore 自动化训练和评估脚本
按顺序训练5个模型，然后按顺序评估每个模型
"""

import os
import subprocess
import time
import json
from datetime import datetime

# 训练配置
TRAINING_CONFIGS = [
    {
        "name": "LoRA",
        "method": "lora", 
        "output_dir": "output_models/my_finetuned_model_lora",
        "rank": 8,
        "alpha": 16,
        "cola_silu": True,
        "cola_init": False,
        "dataset": "merge",  # merge数据集
        "test_tasks": "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"
    },
    {
        "name": "ELSA_SiLU_True",
        "method": "elsa",
        "output_dir": "output_models/my_finetuned_model_elsa_silutrue", 
        "rank": 7,
        "alpha": 14,
        "cola_silu": True,
        "cola_init": False,
        "dataset": "merge",  # merge数据集
        "test_tasks": "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"
    },
    {
        "name": "ELSA_SiLU_False",
        "method": "elsa",
        "output_dir": "output_models/my_finetuned_model_elsa_silufalse",
        "rank": 7, 
        "alpha": 14,
        "cola_silu": False,
        "cola_init": False,
        "dataset": "merge",  # merge数据集
        "test_tasks": "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"
    },
    {
        "name": "COLA",
        "method": "cola",
        "output_dir": "output_models/my_finetuned_model_cola",
        "rank": 8,
        "alpha": 16,
        "cola_silu": True,
        "cola_init": True,
        "dataset": "merge",  # merge数据集  
        "test_tasks": "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"
    },
    {
        "name": "LORO",
        "method": "loro",
        "output_dir": "output_models/my_finetuned_model_loro",
        "rank": 8,
        "alpha": 16, 
        "cola_silu": True,
        "cola_init": False,
        "dataset": "merge",  # merge数据集
        "test_tasks": "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"
    }
]

# 基础训练参数
BASE_TRAIN_ARGS = [
    "--model_name_or_path", "meta-llama/Llama-3.2-1B",
    "--dataset_path", "data/OwLore_Dataset/merge",
    "--overwrite_output_dir",
    "--num_train_epochs", "1.0",
    "--learning_rate", "3e-4",
    "--block_size", "512", 
    "--per_device_train_batch_size", "8",
    "--gradient_accumulation_steps", "1",
    "--bf16",
    "--torch_dtype", "bfloat16",
    "--run_name", "my_finetuning_run",
    "--optim", "adamw_hf",
    "--validation_split_percentage", "0",
    "--logging_steps", "1",
    "--do_train",
    "--ddp_timeout", "72000",
    "--save_steps", "10000",
    "--dataloader_num_workers", "1",
    "--gradient_checkpointing", "False",
    "--use_flash_attention", "1",
    "--use_lisa", "0", 
    "--disable_group_texts", "0",
    "--seed", "111",
    "--galore", "False",
    "--adapter_scope", "all",
    "--sparsity", "0.001",
    "--gamma", "0.7",
    "--sparse_method", "svd",
    "--sparse_svd_rank", "256",
    "--svd_inverse", "False"
]

def run_command(cmd, description, log_file=None):
    """运行命令并记录输出"""
    print(f"\n{'='*60}")
    print(f"开始执行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        if log_file:
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # 实时输出并写入日志
                for line in process.stdout:
                    print(line.strip())
                    f.write(line)
                    f.flush()
                
                process.wait()
                return_code = process.returncode
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return_code = result.returncode
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if return_code == 0:
            print(f"✅ {description} 完成! 耗时: {duration:.2f}秒")
            return True
        else:
            print(f"❌ {description} 失败! 返回码: {return_code}")
            return False
            
    except Exception as e:
        print(f"❌ 执行 {description} 时发生错误: {str(e)}")
        return False

def train_model(config):
    """训练单个模型"""
    print(f"\n🚀 开始训练 {config['name']} 模型...")
    
    # 构建训练命令
    train_cmd = ["python", "examples/finetune.py"] + BASE_TRAIN_ARGS + [
        "--output_dir", config["output_dir"],
        "--lora_rank", str(config["rank"]),
        "--lora_alpha_custom", str(config["alpha"]),
        "--cola_silu", str(config["cola_silu"]),
        "--cola_init", str(config["cola_init"]),
        "--finetune_method", config["method"]
    ]
    
    # 创建日志文件
    log_dir = "experiment_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/train_{config['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    success = run_command(
        train_cmd, 
        f"训练 {config['name']} 模型",
        log_file
    )
    
    if success:
        # 检查complete_model是否存在
        complete_model_path = os.path.join(config["output_dir"], "complete_model")
        if os.path.exists(complete_model_path):
            print(f"✅ {config['name']} 模型训练完成，complete_model已保存")
            return True
        else:
            print(f"⚠️ {config['name']} 模型训练完成，但找不到complete_model目录")
            return False
    
    return False

def evaluate_model(config):
    """评估单个模型"""
    print(f"\n📊 开始评估 {config['name']} 模型...")
    
    # 查找最新的checkpoint中的complete_model
    output_dir = config["output_dir"]
    complete_model_path = None
    
    # 先检查直接的complete_model目录
    direct_complete_model = os.path.join(output_dir, "complete_model")
    if os.path.exists(direct_complete_model):
        complete_model_path = direct_complete_model
    else:
        # 查找checkpoint目录中的complete_model
        if os.path.exists(output_dir):
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                # 按checkpoint编号排序，取最新的
                checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                latest_checkpoint = checkpoints[-1]
                checkpoint_complete_model = os.path.join(output_dir, latest_checkpoint, "complete_model")
                if os.path.exists(checkpoint_complete_model):
                    complete_model_path = checkpoint_complete_model
    
    if not complete_model_path:
        print(f"❌ 找不到 {config['name']} 模型的complete_model目录")
        return False
    
    print(f"📂 使用模型路径: {complete_model_path}")
    
    # 构建评估命令
    eval_cmd = [
        "python", "run_owlore_eval.py",
        "--model_path", complete_model_path,
        "--finetune_method", config["method"],
        "--adapter_scope", "all",
        "--rank", str(config["rank"]),
        "--alpha", str(config["alpha"]),
        "--sparsity", "0.001",
        "--gamma", "0.7",
        "--sparse_method", "svd",
        "--sparse_svd_rank", "256",
        "--cola_silu", str(config["cola_silu"]).lower(),
        "--cola_init", str(config["cola_init"]).lower(),
        "--svd_inverse", "false",
        "--tasks", config["test_tasks"],
        "--output_path", f"experiment_results/{config['name']}_results",
        "--num_fewshot", "5",
        "--batch_size", "auto"
    ]
    
    # 创建结果目录
    os.makedirs("experiment_results", exist_ok=True)
    
    # 创建日志文件
    log_dir = "experiment_logs"
    log_file = f"{log_dir}/eval_{config['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    success = run_command(
        eval_cmd,
        f"评估 {config['name']} 模型",
        log_file
    )
    
    return success

def main():
    """主函数"""
    print("🎯 OwLore 自动化训练和评估开始!")
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建必要的目录
    os.makedirs("experiment_logs", exist_ok=True)
    os.makedirs("experiment_results", exist_ok=True)
    
    # 记录实验配置
    with open("experiment_logs/experiment_config.json", "w") as f:
        json.dump({
            "start_time": datetime.now().isoformat(),
            "configs": TRAINING_CONFIGS
        }, f, indent=2, ensure_ascii=False)
    
    # 阶段1: 训练所有模型
    print("\n" + "="*80)
    print("🏋️ 阶段1: 开始训练所有模型")
    print("="*80)
    
    training_results = {}
    for i, config in enumerate(TRAINING_CONFIGS, 1):
        print(f"\n📋 训练进度: {i}/{len(TRAINING_CONFIGS)}")
        success = train_model(config)
        training_results[config['name']] = success
        
        if not success:
            print(f"❌ {config['name']} 训练失败，继续下一个模型...")
        
        # 训练间隔，让GPU休息一下
        if i < len(TRAINING_CONFIGS):
            print("😴 等待30秒让GPU休息...")
            time.sleep(30)
    
    # 阶段2: 评估所有模型
    print("\n" + "="*80)
    print("📊 阶段2: 开始评估所有模型")
    print("="*80)
    
    evaluation_results = {}
    for i, config in enumerate(TRAINING_CONFIGS, 1):
        print(f"\n📋 评估进度: {i}/{len(TRAINING_CONFIGS)}")
        
        if not training_results[config['name']]:
            print(f"⏭️ 跳过 {config['name']} 评估 (训练失败)")
            evaluation_results[config['name']] = False
            continue
        
        success = evaluate_model(config)
        evaluation_results[config['name']] = success
        
        # 评估间隔
        if i < len(TRAINING_CONFIGS):
            print("😴 等待10秒...")
            time.sleep(10)
    
    # 最终报告
    print("\n" + "="*80)
    print("📋 实验完成报告")
    print("="*80)
    
    print("\n🏋️ 训练结果:")
    for name, success in training_results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {name}: {status}")
    
    print("\n📊 评估结果:")
    for name, success in evaluation_results.items():
        if name not in training_results or not training_results[name]:
            status = "⏭️ 跳过"
        else:
            status = "✅ 成功" if success else "❌ 失败"
        print(f"  {name}: {status}")
    
    # 保存最终结果
    final_results = {
        "end_time": datetime.now().isoformat(),
        "training_results": training_results,
        "evaluation_results": evaluation_results
    }
    
    with open("experiment_logs/final_results.json", "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 所有实验完成! 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📁 日志文件保存在: experiment_logs/")
    print("📁 结果文件保存在: experiment_results/")

if __name__ == "__main__":
    main()