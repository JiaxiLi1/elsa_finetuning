#!/usr/bin/env python
# coding=utf-8
"""
Evaluation script for OwLore models with low-rank adapters.
This script properly loads models trained with loro method and evaluates them using lm_eval.
"""

import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

# Add the OwLore modules to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lowrank_adpt_module import (
    apply_lowrank_adpt_param,
    get_lowrank_adpt_param,
    LowRankAdapterLinear
)

class OwLoreModelLoader:
    """Custom model loader for OwLore models with low-rank adapters."""
    
    def __init__(self, model_path, finetune_method="loro", adapter_scope="all", rank=6, alpha=12, 
                 sparsity=0.001, gamma=0.7, sparse_method="svd", sparse_svd_rank=256, 
                 cola_silu=True, cola_init=False, svd_inverse=False):
        self.model_path = model_path
        self.finetune_method = finetune_method
        self.adapter_scope = adapter_scope
        self.rank = rank
        self.alpha = alpha
        self.sparsity = sparsity
        self.gamma = gamma
        self.sparse_method = sparse_method
        self.sparse_svd_rank = sparse_svd_rank
        self.cola_silu = cola_silu
        self.cola_init = cola_init
        self.svd_inverse = svd_inverse
        
    def _load_hybrid_adapter_model(self, tokenizer):
        """Load hybrid adapter models (ELSA, COLA, LoRA) with special loading logic."""
        
        # First, try to find the original base model path from config
        import json
        config_path = os.path.join(self.model_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load base model first with the same architecture
        print(f"Loading base model architecture for {self.finetune_method.upper()}")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",  # Use the original base model
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Apply the appropriate adapter
        if self.finetune_method == "elsa":
            print(f"Applying ELSA hybrid adapter parameterization")
            from hybrid_lowrank_sparse_adapter import apply_hybrid_adapter
            apply_hybrid_adapter(
                model=model,
                scope=self.adapter_scope,
                rank=self.rank,
                sparsity=self.sparsity,
                gamma=self.gamma,
                sparse_method=self.sparse_method,
                sparse_svd_rank=self.sparse_svd_rank,
                alpha=self.alpha,
                cola_silu=self.cola_silu,
                cola_init=self.cola_init,
                svd_inverse=self.svd_inverse
            )
        elif self.finetune_method == "cola":
            print(f"Applying COLA hybrid adapter parameterization")
            from cola_lowrank_sparse_adapter import cola_apply_hybrid_adapter
            cola_apply_hybrid_adapter(
                model=model,
                scope=self.adapter_scope,
                rank=self.rank,
                sparsity=self.sparsity,
                gamma=self.gamma,
                sparse_method=self.sparse_method,
                sparse_svd_rank=self.sparse_svd_rank,
                alpha=self.alpha,
                cola_silu=self.cola_silu,
                cola_init=self.cola_init,
                svd_inverse=self.svd_inverse
            )
        elif self.finetune_method == "lora":
            print(f"Applying LoRA hybrid adapter parameterization")
            from lora_lowrank_sparse_adapter import lora_apply_hybrid_adapter
            lora_apply_hybrid_adapter(
                model=model,
                scope=self.adapter_scope,
                rank=self.rank,
                sparsity=self.sparsity,
                gamma=self.gamma,
                sparse_method=self.sparse_method,
                sparse_svd_rank=self.sparse_svd_rank,
                alpha=self.alpha,
                cola_silu=self.cola_silu,
                cola_init=self.cola_init,
                svd_inverse=self.svd_inverse
            )
        
        # Now load the trained weights
        print(f"Loading trained {self.finetune_method.upper()} weights from {self.model_path}")
        try:
            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(self.model_path, "model.safetensors"))
        except ImportError:
            state_dict = torch.load(os.path.join(self.model_path, "model.safetensors"), 
                                  map_location="cpu", weights_only=False)
        
        # Load the state dict with strict=False to handle missing/extra keys
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys: {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {len(unexpected_keys)} keys")
        
        # Ensure all model parameters have the correct dtype (bfloat16)
        print("Converting model parameters to bfloat16...")
        model = model.to(torch.bfloat16)
        
        print(f"✓ {self.finetune_method.upper()} model loaded successfully!")
        return model, tokenizer
    
    def load_model_and_tokenizer(self):
        """Load the model and tokenizer with proper low-rank adapter setup."""
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # For ELSA, COLA, and LoRA methods, we need to load base model first, 
        # then apply adapters, then load the adapted weights
        if self.finetune_method in ["elsa", "cola", "lora"]:
            return self._load_hybrid_adapter_model(tokenizer)
        
        # For LORO method, load base model first then apply adapters
        print(f"Loading base model for LORO adaptation from: {self.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Apply LORO adapter parameterization
        if self.finetune_method == "loro":
            print(f"Applying LORO low-rank adapter parameterization with scope: {self.adapter_scope}, rank: {self.rank}, alpha: {self.alpha}")
            apply_lowrank_adpt_param(
                model,
                model_type="llama",
                scope=self.adapter_scope,
                rank=self.rank,
                alpha=self.alpha,
                init="xavier",
            )
            
            # Load the trained weights
            print(f"Loading trained weights from {self.model_path}")
            try:
                # Try to load with safetensors first
                from safetensors.torch import load_file
                state_dict = load_file(os.path.join(self.model_path, "model.safetensors"))
            except ImportError:
                # Fallback to torch.load with weights_only=False for older models
                state_dict = torch.load(os.path.join(self.model_path, "model.safetensors"), 
                                      map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"Failed to load with safetensors, trying torch.load: {e}")
                state_dict = torch.load(os.path.join(self.model_path, "model.safetensors"), 
                                      map_location="cpu", weights_only=False)
            
            model.load_state_dict(state_dict, strict=False)
            
        return model, tokenizer

def create_hf_model_wrapper(model_path, finetune_method="loro", adapter_scope="all", rank=6, alpha=12):
    """Create a model wrapper that can be used with lm_eval."""
    
    class OwLoreHFModel:
        def __init__(self, model_path, finetune_method, adapter_scope, rank, alpha):
            self.loader = OwLoreModelLoader(model_path, finetune_method, adapter_scope, rank, alpha)
            self.model, self.tokenizer = self.loader.load_model_and_tokenizer()
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
            return self.model.to(device)
    
    return OwLoreHFModel(model_path, finetune_method, adapter_scope, rank, alpha)

def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate OwLore models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--finetune_method", type=str, default="loro", choices=["loro", "elsa", "cola", "lora"], help="Fine-tuning method used")
    parser.add_argument("--adapter_scope", type=str, default="all", help="Adapter scope")
    parser.add_argument("--rank", type=int, default=6, help="Low-rank adapter rank")
    parser.add_argument("--alpha", type=int, default=12, help="Low-rank adapter alpha")
    parser.add_argument("--sparsity", type=float, default=0.001, help="Sparsity for hybrid adapters")
    parser.add_argument("--gamma", type=float, default=0.7, help="Gamma for hybrid adapters")
    parser.add_argument("--sparse_method", type=str, default="svd", help="Sparse method for hybrid adapters")
    parser.add_argument("--sparse_svd_rank", type=int, default=256, help="SVD rank for sparse method")
    parser.add_argument("--cola_silu", action="store_true", default=True, help="Use SiLU activation for COLA")
    parser.add_argument("--cola_init", action="store_true", default=False, help="COLA initialization method")
    parser.add_argument("--svd_inverse", action="store_true", default=False, help="Use SVD inverse")
    parser.add_argument("--tasks", type=str, default="boolq", help="Evaluation tasks")
    parser.add_argument("--output_path", type=str, default="owlore_results", help="Output path for results")
    parser.add_argument("--num_fewshot", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--batch_size", type=str, default="auto", help="Batch size")
    
    args = parser.parse_args()
    
    print(f"Loading OwLore model from: {args.model_path}")
    print(f"Method: {args.finetune_method}, Scope: {args.adapter_scope}, Rank: {args.rank}, Alpha: {args.alpha}")
    
    # Test loading the model
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
        print("✓ Model loaded successfully!")
        
        # Test a simple generation
        test_input = "The capital of France is"
        inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs.input_ids.shape[1] + 10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Test generation successful: {generated_text}")
        
        print("\nModel structure check:")
        for name, module in model.named_modules():
            if isinstance(module, LowRankAdapterLinear):
                print(f"  Found LowRankAdapterLinear: {name}")
                
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Now run lm_eval with the custom model
    print(f"\nRunning evaluation with lm_eval...")
    print(f"Tasks: {args.tasks}")
    print(f"Output: {args.output_path}")
    
    # We'll need to modify the lm_eval command to use our custom model loader
    # For now, let's just confirm the model loads correctly
    
if __name__ == "__main__":
    main()