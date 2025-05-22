#!/usr/bin/env python3
"""
使用自定义模型加载器评估包含自定义模块的模型示例

这个脚本展示了如何使用 lm-eval 的自定义模型加载器来评估
使用 LoRA、CoLA、ELSA、LoRO 等自定义模块训练的模型。

使用方法:
1. 命令行方式:
   python -m lm_eval --model hf-custom --model_args pretrained=/path/to/your/model,custom_modules_path=/path/to/custom/modules.py --tasks mmlu --output_path results

2. 通过这个脚本:
   python example_custom_evaluation.py
"""

import os
import sys
from pathlib import Path

# 添加lm_eval到路径
sys.path.insert(0, str(Path(__file__).parent))

from lm_eval import cli_evaluate
import argparse


def create_custom_args():
    """创建自定义评估参数"""

    # 这里设置你的模型路径
    MODEL_PATH = "/home/rtx3090/code_jiaxi/OwLore/output_models/my_finetuned_model/checkpoint-5/complete_model"

    # 自定义模块路径（可选，如果为None会自动搜索）
    CUSTOM_MODULES_PATH = None  # 例如: "/path/to/your/custom_modules.py"

    # 创建模拟的命令行参数
    class Args:
        def __init__(self):
            # 使用自定义模型加载器
            self.model = "hf-custom"

            # 模型参数
            if CUSTOM_MODULES_PATH:
                self.model_args = {
                    "pretrained": MODEL_PATH,
                    "custom_modules_path": CUSTOM_MODULES_PATH,
                    "trust_remote_code": True,
                    "device": "cuda",  # 或 "cpu"
                }
            else:
                self.model_args = {
                    "pretrained": MODEL_PATH,
                    "trust_remote_code": True,
                    "device": "cuda",  # 或 "cpu"
                }

            # 评估任务
            self.tasks = "mmlu"  # 你可以改为其他任务，如 "hellaswag,arc_easy,arc_challenge"

            # 输出设置
            self.output_path = "custom_model_results"
            self.log_samples = True

            # Few-shot 设置
            self.num_fewshot = 5

            # 批处理设置
            self.batch_size = "auto"
            self.max_batch_size = 8

            # 缓存设置
            self.cache_requests = "true"
            self.use_cache = None

            # 其他设置
            self.device = "cuda"
            self.limit = None  # 设为小数字如10进行快速测试
            self.samples = None
            self.check_integrity = False
            self.write_out = False
            self.system_instruction = None
            self.apply_chat_template = False
            self.fewshot_as_multiturn = False
            self.show_config = False
            self.include_path = None
            self.gen_kwargs = None
            self.verbosity = "DEBUG"  # 使用DEBUG级别查看详细加载过程
            self.wandb_args = ""
            self.wandb_config_args = ""
            self.hf_hub_log_args = ""
            self.predict_only = False
            self.seed = [0, 1234, 1234, 1234]
            self.trust_remote_code = True
            self.confirm_run_unsafe_code = False
            self.metadata = None

    return Args()


def main():
    """主函数"""
    print("🚀 开始使用自定义模型加载器进行评估...")

    # 创建参数
    args = create_custom_args()

    # 显示配置信息
    print(f"📁 模型路径: {args.model_args['pretrained']}")
    print(f"🏷️  模型类型: {args.model}")
    print(f"📊 评估任务: {args.tasks}")
    print(f"💾 输出路径: {args.output_path}")
    print(f"🔢 Few-shot数量: {args.num_fewshot}")
    print(f"⚡ 批处理大小: {args.batch_size}")

    try:
        # 开始评估
        cli_evaluate(args)
        print("✅ 评估完成!")

    except Exception as e:
        print(f"❌ 评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def command_line_example():
    """显示命令行使用示例"""
    print("\n" + "=" * 50)
    print("📝 命令行使用示例:")
    print("=" * 50)

    examples = [
        # 基本用法
        """
# 基本用法（自动搜索自定义模块）:
python -m lm_eval \\
    --model hf-custom \\
    --model_args pretrained=/path/to/your/complete_model,trust_remote_code=True \\
    --tasks mmlu \\
    --output_path mmlu_results \\
    --num_fewshot 5 \\
    --batch_size auto \\
    --cache_requests true
        """,

        # 指定自定义模块路径
        """
# 指定自定义模块路径:
python -m lm_eval \\
    --model hf-custom \\
    --model_args pretrained=/path/to/your/complete_model,custom_modules_path=/path/to/custom_modules.py,trust_remote_code=True \\
    --tasks mmlu \\
    --output_path mmlu_results \\
    --num_fewshot 5 \\
    --batch_size auto \\
    --cache_requests true
        """,

        # 多任务评估
        """
# 多任务评估:
python -m lm_eval \\
    --model hf-custom \\
    --model_args pretrained=/path/to/your/complete_model,trust_remote_code=True \\
    --tasks hellaswag,arc_easy,arc_challenge,mmlu \\
    --output_path comprehensive_results \\
    --num_fewshot 5 \\
    --batch_size auto \\
    --log_samples \\
    --cache_requests true
        """
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n示例 {i}:")
        print(example.strip())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--examples":
        command_line_example()
    else:
        main()
        command_line_example()