#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OwLore 完整实验脚本 - 训练后立即评估
适合下班前启动，自动完成所有训练和评估
"""

import os
import subprocess
import time
import json
from datetime import datetime

def log_message(message, log_file=None):
    """记录消息到控制台和日志文件"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(formatted_message + "\n")
            f.flush()

def run_command_with_conda(cmd, conda_env, description, log_file=None):
    """在指定conda环境中运行命令并实时输出"""
    log_message(f"🚀 开始: {description}", log_file)
    log_message(f"环境: {conda_env}", log_file)
    log_message(f"命令: {' '.join(cmd)}", log_file)
    
    start_time = time.time()
    
    try:
        # 构建带conda环境的命令
        if conda_env:
            # 使用bash -c来执行conda activate和实际命令
            # 先source conda的初始化脚本，然后激活环境
            # 使用shlex.quote来正确处理命令中的引号和特殊字符
            import shlex
            escaped_cmd = ' '.join(shlex.quote(arg) for arg in cmd)
            full_cmd = [
                "bash", "-c", 
                f"source /home/rtx3090/miniconda3/etc/profile.d/conda.sh && conda activate {conda_env} && {escaped_cmd}"
            ]
        else:
            full_cmd = cmd
        
        process = subprocess.Popen(
            full_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时输出
        output_lines = []
        for line in process.stdout:
            line = line.strip()
            if line:  # 只输出非空行
                log_message(line, log_file)
                output_lines.append(line)
        
        process.wait()
        return_code = process.returncode
        
        end_time = time.time()
        duration = end_time - start_time
        
        if return_code == 0:
            log_message(f"✅ {description} 完成! 耗时: {duration:.1f}秒", log_file)
            return True, output_lines
        else:
            log_message(f"❌ {description} 失败! 返回码: {return_code}", log_file)
            return False, output_lines
    except Exception as e:
        log_message(f"❌ {description} 执行异常: {str(e)}", log_file)
        return False, []

def run_command(cmd, description, log_file=None):
    """运行命令并实时输出（不指定conda环境）"""
    return run_command_with_conda(cmd, None, description, log_file)

def train_and_evaluate_model(config, main_log_file):
    """训练并立即评估单个模型"""
    model_name = config['name']
    log_message(f"\n{'='*60}", main_log_file)
    log_message(f"🎯 开始处理 {model_name} 模型", main_log_file)
    log_message(f"{'='*60}", main_log_file)
    
    # 1. 训练模型
    log_message(f"🏋️ 第一步: 训练 {model_name} 模型", main_log_file)
    
    train_cmd = [
        "python", "examples/finetune.py",
        "--model_name_or_path", "meta-llama/Llama-3.2-1B",
        "--dataset_path", "data/OwLore_Dataset/merge",
        "--output_dir", config["output_dir"],
        "--overwrite_output_dir",
        "--num_train_epochs", "1.0",
        "--learning_rate", "3e-4",
        "--block_size", "512",
        "--per_device_train_batch_size", "8",
        "--gradient_accumulation_steps", "1",
        "--bf16",
        "--torch_dtype", "bfloat16",
        "--run_name", f"{model_name}_training",
        "--optim", "adamw_hf",
        "--validation_split_percentage", "0",
        "--logging_steps", "1",
        "--do_train",
        "--ddp_timeout", "72000",
        "--save_steps", "10000",  # 只在最后保存
        "--dataloader_num_workers", "1",
        "--gradient_checkpointing", "False",
        "--use_flash_attention", "1",
        "--use_lisa", "0",
        "--disable_group_texts", "0",
        "--seed", "111",
        "--galore", "False",
        "--lora_rank", str(config["rank"]),
        "--lora_alpha_custom", str(config["alpha"]),
        "--cola_silu", str(config["cola_silu"]),
        "--cola_init", str(config["cola_init"]),
        "--adapter_scope", "all",
        "--sparsity", "0.001",
        "--gamma", "0.7",
        "--sparse_method", "svd",
        "--sparse_svd_rank", "256",
        "--svd_inverse", "False",
        "--finetune_method", config["method"]
    ]
    
    train_success, _ = run_command_with_conda(train_cmd, "owlore_fine", f"训练 {model_name}", main_log_file)
    
    if not train_success:
        log_message(f"❌ {model_name} 训练失败，跳过评估", main_log_file)
        return False, False
    
    # 查找complete_model - 支持不同的保存位置
    output_dir = config["output_dir"]
    complete_model_path = None
    
    # 先检查直接的complete_model目录
    direct_complete_model = os.path.join(output_dir, "complete_model")
    if os.path.exists(direct_complete_model):
        complete_model_path = direct_complete_model
    else:
        # 查找最新的checkpoint目录中的complete_model
        import glob
        checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if checkpoint_dirs:
            # 按数字排序，取最新的
            latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
            checkpoint_complete_model = os.path.join(latest_checkpoint, "complete_model")
            if os.path.exists(checkpoint_complete_model):
                complete_model_path = checkpoint_complete_model
    
    if not complete_model_path:
        log_message(f"❌ {model_name} 找不到complete_model目录", main_log_file)
        return True, False
    
    log_message(f"✅ {model_name} 训练完成，找到complete_model: {complete_model_path}", main_log_file)
    
    # 短暂休息
    log_message("😴 训练完成，休息10秒后开始评估...", main_log_file)
    time.sleep(10)
    
    # 2. 评估模型
    log_message(f"📊 第二步: 评估 {model_name} 模型", main_log_file)
    
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
        "--tasks", "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa",
        "--output_path", f"final_results/{model_name}_results",
        "--num_fewshot", "5",
        "--batch_size", "auto"
    ]
    
    eval_success, _ = run_command_with_conda(eval_cmd, "owlore_eval", f"评估 {model_name}", main_log_file)
    
    if eval_success:
        log_message(f"✅ {model_name} 评估完成", main_log_file)
    else:
        log_message(f"❌ {model_name} 评估失败", main_log_file)
    
    return train_success, eval_success

def main():
    """主函数"""
    start_time = datetime.now()
    print("🎯 OwLore 完整实验开始!")
    print(f"📅 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("🌙 适合下班前启动，自动完成所有训练和评估")
    
    # 创建目录
    os.makedirs("experiment_logs", exist_ok=True)
    os.makedirs("final_results", exist_ok=True)
    
    # 主日志文件
    main_log_file = f"experiment_logs/complete_experiment_{start_time.strftime('%Y%m%d_%H%M%S')}.log"
    
    # 实验配置
    configs = [
        {
            "name": "LoRA",
            "method": "lora", 
            "output_dir": "output_models/my_finetuned_model_lora",
            "rank": 8, "alpha": 16,
            "cola_silu": True, "cola_init": False
        },
        {
            "name": "ELSA_SiLU_True",
            "method": "elsa",
            "output_dir": "output_models/my_finetuned_model_elsa_silutrue", 
            "rank": 7, "alpha": 14,
            "cola_silu": True, "cola_init": False
        },
        {
            "name": "ELSA_SiLU_False",
            "method": "elsa",
            "output_dir": "output_models/my_finetuned_model_elsa_silufalse",
            "rank": 7, "alpha": 14,
            "cola_silu": False, "cola_init": False
        },
        {
            "name": "COLA",
            "method": "cola",
            "output_dir": "output_models/my_finetuned_model_cola",
            "rank": 8, "alpha": 16,
            "cola_silu": True, "cola_init": True
        },
        {
            "name": "LORO",
            "method": "loro",
            "output_dir": "output_models/my_finetuned_model_loro",
            "rank": 8, "alpha": 16,
            "cola_silu": True, "cola_init": False
        }
    ]
    
    log_message("🎯 实验配置:", main_log_file)
    for i, config in enumerate(configs, 1):
        extra_info = ""
        if config['method'] == 'cola':
            extra_info = f", cola_init={config['cola_init']}"
        elif config['method'] == 'elsa':
            extra_info = f", cola_silu={config['cola_silu']}"
        log_message(f"  {i}. {config['name']}: {config['method']} (rank={config['rank']}, alpha={config['alpha']}{extra_info})", main_log_file)
    
    # 执行实验
    results = {}
    
    for i, config in enumerate(configs, 1):
        log_message(f"\n🔄 进度: {i}/{len(configs)}", main_log_file)
        
        train_success, eval_success = train_and_evaluate_model(config, main_log_file)
        results[config['name']] = {
            'train_success': train_success,
            'eval_success': eval_success
        }
        
        # 模型间休息
        if i < len(configs):
            log_message("😴 模型完成，休息30秒后处理下一个模型...", main_log_file)
            time.sleep(30)
    
    # 最终报告
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    log_message(f"\n{'='*60}", main_log_file)
    log_message("🎉 所有实验完成!", main_log_file)
    log_message(f"📅 结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}", main_log_file)
    log_message(f"⏱️ 总耗时: {total_duration}", main_log_file)
    log_message(f"{'='*60}", main_log_file)
    
    log_message("\n📋 最终结果:", main_log_file)
    train_success_count = 0
    eval_success_count = 0
    
    for name, result in results.items():
        train_status = "✅" if result['train_success'] else "❌"
        eval_status = "✅" if result['eval_success'] else "❌" if result['train_success'] else "⏭️"
        
        log_message(f"  {name}:", main_log_file)
        log_message(f"    训练: {train_status}", main_log_file)
        log_message(f"    评估: {eval_status}", main_log_file)
        
        if result['train_success']:
            train_success_count += 1
        if result['eval_success']:
            eval_success_count += 1
    
    log_message(f"\n📊 统计:", main_log_file)
    log_message(f"  训练成功: {train_success_count}/{len(configs)}", main_log_file)
    log_message(f"  评估成功: {eval_success_count}/{len(configs)}", main_log_file)
    
    # 保存结果
    final_results = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_duration_seconds": total_duration.total_seconds(),
        "results": results
    }
    
    with open("experiment_logs/final_summary.json", "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    log_message(f"\n📁 文件位置:", main_log_file)
    log_message(f"  详细日志: {main_log_file}", main_log_file)
    log_message(f"  结果摘要: experiment_logs/final_summary.json", main_log_file)
    log_message(f"  评估结果: final_results/", main_log_file)
    
    print(f"\n🌟 实验完成! 可以安心下班了~ 总耗时: {total_duration}")

if __name__ == "__main__":
    main()