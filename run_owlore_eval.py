#!/usr/bin/env python
# coding=utf-8
"""
Run lm_eval with OwLore models.
This script creates a custom model wrapper and runs lm_eval evaluation.
"""

import sys
import os
import tempfile
import json
import subprocess
import torch
from pathlib import Path

# Add the OwLore modules to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_owlore_model import OwLoreModelLoader

def create_custom_hf_model_class(model_path, finetune_method="loro", adapter_scope="all", rank=6, alpha=12):
    """Create a custom HF model class file for lm_eval to use."""
    
    custom_model_code = f'''
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add the OwLore path
sys.path.insert(0, "{os.path.dirname(os.path.abspath(__file__))}")

from lowrank_adpt_module import apply_lowrank_adpt_param
from safetensors.torch import load_file

class OwLoreCustomModel:
    """Custom model class for OwLore models with low-rank adapters."""
    
    def __init__(self):
        self.model_path = "{model_path}"
        self.finetune_method = "{finetune_method}"
        self.adapter_scope = "{adapter_scope}"
        self.rank = {rank}
        self.alpha = {alpha}
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Apply low-rank adapter parameterization
        if self.finetune_method == "loro":
            apply_lowrank_adpt_param(
                self.model,
                model_type="llama",
                scope=self.adapter_scope,
                rank=self.rank,
                alpha=self.alpha,
                init="xavier",
            )
            
            # Load trained weights
            state_dict = load_file(os.path.join(self.model_path, "model.safetensors"))
            self.model.load_state_dict(state_dict, strict=False)
        
        self.model.eval()
        self.device = next(self.model.parameters()).device
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def eval(self):
        return self.model.eval()
    
    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self
    
    def get_tokenizer(self):
        return self.tokenizer

# Create the model instance
owlore_model = OwLoreCustomModel()
'''
    return custom_model_code

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run lm_eval with OwLore models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--finetune_method", type=str, default="loro", choices=["loro", "elsa", "cola", "lora"], help="Fine-tuning method used (loro, elsa, cola, lora)")
    parser.add_argument("--adapter_scope", type=str, default="all", help="Adapter scope")
    parser.add_argument("--rank", type=int, default=6, help="Low-rank adapter rank")
    parser.add_argument("--alpha", type=int, default=12, help="Low-rank adapter alpha")
    parser.add_argument("--sparsity", type=float, default=0.001, help="Sparsity for hybrid adapters")
    parser.add_argument("--gamma", type=float, default=0.7, help="Gamma for hybrid adapters")
    parser.add_argument("--sparse_method", type=str, default="svd", help="Sparse method for hybrid adapters")
    parser.add_argument("--sparse_svd_rank", type=int, default=256, help="SVD rank for sparse method")
    parser.add_argument("--cola_silu", action="store_true", help="Use SiLU activation for COLA")
    parser.add_argument("--no-cola_silu", dest="cola_silu", action="store_false", help="Disable SiLU activation for COLA")
    parser.add_argument("--cola_init", action="store_true", help="COLA initialization method")
    parser.add_argument("--svd_inverse", action="store_true", help="Use SVD inverse")
    parser.set_defaults(cola_silu=True)
    parser.add_argument("--tasks", type=str, default="boolq", help="Evaluation tasks")
    parser.add_argument("--output_path", type=str, default="owlore_results", help="Output path for results")
    parser.add_argument("--num_fewshot", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--batch_size", type=str, default="auto", help="Batch size")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples per task")
    
    args = parser.parse_args()
    
    print(f"Running OwLore evaluation...")
    print(f"Model: {args.model_path}")
    print(f"Method: {args.finetune_method}, Scope: {args.adapter_scope}, Rank: {args.rank}, Alpha: {args.alpha}")
    print(f"Tasks: {args.tasks}")
    print(f"Output: {args.output_path}")
    
    # First verify the model loads correctly
    print("\\nVerifying model loading...")
    try:
        loader = OwLoreModelLoader(
            args.model_path, 
            args.finetune_method, 
            args.adapter_scope, 
            args.rank, 
            args.alpha,
            args.sparsity,
            args.gamma,
            args.sparse_method,
            args.sparse_svd_rank,
            args.cola_silu,
            args.cola_init,
            args.svd_inverse
        )
        model, tokenizer = loader.load_model_and_tokenizer()
        print("✓ Model verification successful!")
        
        # Clean up
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ Model verification failed: {e}")
        return
    
    # Now run the actual evaluation using direct python call to lm_eval
    print("\\nRunning lm_eval evaluation...")
    
    # Prepare the command
    cmd = [
        "python", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={args.model_path}",
        "--tasks", args.tasks,
        "--output_path", args.output_path,
        "--num_fewshot", str(args.num_fewshot),
        "--batch_size", args.batch_size,
        "--log_samples"
    ]
    
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the evaluation
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Evaluation completed successfully!")
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Evaluation failed with return code {e.returncode}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        
        # Try a simpler approach - just use our custom model directly
        print("\\nTrying direct evaluation approach...")
        try:
            # Import lm_eval modules directly
            import lm_eval
            from lm_eval.models.huggingface import HFLM
            
            # Create a custom HFLM wrapper
            class OwLoreHFLM(HFLM):
                def __init__(self, **kwargs):
                    # Override the model loading
                    self.owlore_loader = OwLoreModelLoader(
                        args.model_path, 
                        args.finetune_method, 
                        args.adapter_scope, 
                        args.rank, 
                        args.alpha
                    )
                    model, tokenizer = self.owlore_loader.load_model_and_tokenizer()
                    
                    # Set the attributes that HFLM expects
                    self._model = model
                    self.tokenizer = tokenizer
                    self._device = next(model.parameters()).device
                    self.vocab_size = len(tokenizer)
                    
                    # Set other required attributes
                    self.batch_size = 1
                    self.max_length = 2048
                    
            # Create the model
            owlore_hflm = OwLoreHFLM()
            
            # Run evaluation
            results = lm_eval.simple_evaluate(
                model=owlore_hflm,
                tasks=args.tasks.split(','),
                num_fewshot=args.num_fewshot,
                batch_size=args.batch_size,
                limit=args.limit,
            )
            
            print("✓ Direct evaluation successful!")
            print("Results:")
            for task, result in results['results'].items():
                print(f"  {task}:")
                for metric, value in result.items():
                    if isinstance(value, (int, float)):
                        print(f"    {metric}: {value:.4f}")
                        
            # Save results
            os.makedirs(args.output_path, exist_ok=True)
            with open(f"{args.output_path}/results.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output_path}/results.json")
            
        except Exception as e2:
            print(f"✗ Direct evaluation also failed: {e2}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()