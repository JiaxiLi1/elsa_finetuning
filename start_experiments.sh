#!/bin/bash
# OwLore 实验启动脚本
# 下班前运行这个脚本，第二天来看结果

echo "🎯 OwLore 完整实验启动"
echo "📅 启动时间: $(date)"
echo "🌙 适合下班前启动，自动完成所有训练和评估"
echo ""

# 运行完整实验（脚本内部会自动切换环境）
source /home/rtx3090/miniconda3/etc/profile.d/conda.sh && python run_complete_experiments.py

echo ""
echo "🎉 实验脚本执行完成!"
echo "📁 查看结果:"
echo "  - 详细日志: experiment_logs/"
echo "  - 评估结果: final_results/"
echo "📅 完成时间: $(date)"