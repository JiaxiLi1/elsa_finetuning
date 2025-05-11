#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import sys
import os
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
from transformers import HfArgumentParser

from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)
from dataclasses import dataclass, field
from lmflow.datasets.dataset import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline

@dataclass
class LoRAArguments:
    use_lora1: bool = field(default=False)
    lora_attn: bool = field(default=False)
    lora_mlp: bool = field(default=False)
    lora_lm_head: bool = field(default=False)
    init_method: str = field(default="SVD")
    rank_ratio: float = field(default=0.1)
    sparse: bool = field(default=False)
    sparse_sparsity: float = field(default=0.1)
    sparse_w_zero: bool = field(default=True)
    sparse_gamma: float = field(default=0.8)
    sparse_gamma_update: bool = field(default=False)
    sparse_gamma_all_apply: bool = field(default=False)
    sparse_method: str = field(default="random")
    sparse_svd_rank: int = field(default=128)
    sparse_only: bool = field(default=False)
    sparse_svd_inverse: bool = field(default=True)

class SVDLowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank_ratio: float,
                 original_weight: torch.Tensor, bias: bool = True, use_mup: bool = False,
                 layer_name: str = "", coef_a: float = 1.0, coef_b: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_mup = use_mup
        self.layer_name = layer_name
        self.coef_a = coef_a
        self.coef_b = coef_b

        max_rank = min(in_features, out_features)
        self.rank = max(1, int(max_rank * rank_ratio))

        self.matrix_A = nn.Parameter(torch.empty((in_features, self.rank)))
        self.matrix_B = nn.Parameter(torch.empty((self.rank, out_features)))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        if use_mup:
            self.temp_weight = nn.Parameter(torch.empty_like(original_weight))
            self.standard_init()
        else:
            self.apply_svd_init(original_weight, self.layer_name)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def standard_init(self):
        nn.init.kaiming_normal_(self.temp_weight, mode='fan_in', nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def apply_svd_init(self, weight_matrix, layer_name=""):

        print(f"\nProcessing layer: {layer_name}")
        print(f"Original weight shape: {weight_matrix.shape}")
        print(f"Target shapes - A: {self.matrix_A.shape}, B: {self.matrix_B.shape}")

        weight_matrix = weight_matrix.t()

        U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)

        U_r = U[:, :self.rank]  # shape: [in_features, rank]
        S_r = S[:self.rank]  # shape: [rank]
        V_r = Vh[:self.rank, :].t()  # shape: [out_features, rank]

        print(f"After SVD - U_r: {U_r.shape}, S_r: {S_r.shape}, V_r: {V_r.shape}")

        S_sqrt = torch.sqrt(S_r)

        with torch.no_grad():

            A = U_r * S_sqrt.view(1, -1)
            B = (V_r * S_sqrt.view(1, -1)).t()

            print(f"Computed shapes - A: {A.shape}, B: {B.shape}")

            assert A.shape == self.matrix_A.shape, f"Shape mismatch for A: got {A.shape}, expected {self.matrix_A.shape}"
            assert B.shape == self.matrix_B.shape, f"Shape mismatch for B: got {B.shape}, expected {self.matrix_B.shape}"

            A = A * self.coef_a
            B = B * self.coef_b

            self.matrix_A.data.copy_(A)
            self.matrix_B.data.copy_(B)

            if self.rank > 1:
                A_ortho = torch.mm(self.matrix_A.t(), self.matrix_A)
                B_ortho = torch.mm(self.matrix_B, self.matrix_B.t())
                eye = torch.eye(self.rank, device=self.matrix_A.device)
                A_err = torch.norm(A_ortho - eye).item()
                B_err = torch.norm(B_ortho - eye).item()
                if A_err > 1e-5 or B_err > 1e-5:
                    print(f"Warning: Matrices not orthogonal. A error: {A_err:.6f}, B error: {B_err:.6f}")

        reconstructed = self.matrix_A @ self.matrix_B
        orig_std = torch.std(weight_matrix).item()
        orig_mean = torch.mean(weight_matrix).item()
        recon_std = torch.std(reconstructed).item()
        recon_mean = torch.mean(reconstructed).item()
        std_diff = abs(recon_std - orig_std)

        print(f"Stats:")
        print(f"Original std: {orig_std:.6f}")
        print(f"Original mean: {orig_mean:.6f}")
        print(f"Reconstructed std: {recon_std:.6f}")
        print(f"Reconstructed mean: {recon_mean:.6f}")
        print(f"Std difference: {std_diff:.6f}")
        # print(f"Top-{self.rank} singular values: {S_r.tolist()}\n")

    def mup_init(self):
        if self.use_mup:
            assert hasattr(self.temp_weight, 'infshape'), "temp_weight missing infshape"

            self.apply_svd_init(self.temp_weight, self.layer_name)

            del self.temp_weight
            self._parameters.pop('temp_weight', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input @ self.matrix_A @ self.matrix_B
        if self.bias is not None:
            output = output + self.bias
        return output


def replace_linear_with_svd_low_rank(module, rank_ratio, target_modules=None, use_mup=False, coef_a=1.0, coef_b=1.0):
    if target_modules is None:
        target_modules = ["attn", "mlp", "lm_head"]

    replacements = {}
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            if any(t in name for t in target_modules):
                low_rank_layer = SVDLowRankLinear(
                    child.in_features,
                    child.out_features,
                    rank_ratio,
                    child.weight.data,
                    bias=child.bias is not None,
                    use_mup=use_mup,
                    layer_name=name,
                    coef_a=coef_a,
                    coef_b=coef_b
                )
                replacements[name] = low_rank_layer

                original_size = child.in_features * child.out_features
                low_rank_size = (child.in_features * low_rank_layer.rank +
                                 low_rank_layer.rank * child.out_features)
                print(f"Replacing layer: {name}")
                print(f"  Original params: {original_size}")
                print(f"  Low rank params: {low_rank_size}")
                print(f"  Compression ratio: {low_rank_size / original_size:.2%}\n")

    for name, replacement in replacements.items():
        parts = name.split('.')
        current = module
        for part in parts[:-1]:
            current = getattr(current, part)
        setattr(current, parts[-1], replacement)


def create_svd_low_rank_llama(model, rank_ratio, target_modules=None, use_mup=False, coef_a=1.0, coef_b=1.0):
    print("\nCreating SVD initialized low-rank LLaMA model...")
    replace_linear_with_svd_low_rank(model, rank_ratio, target_modules, use_mup=use_mup, coef_a=coef_a, coef_b=coef_b)
    return model

def register_activation_hooks(model):
    def hook(module, input, output):
        module.last_output = output

    for name, module in model.named_modules():
        if isinstance(module, (SVDLowRankLinear)):
            module.register_forward_hook(hook)

def print_param_size(model):
    total = 0
    for name, param in model.named_parameters():
        param_size = param.numel()
        print(f"{name}: {param_size:,} parameters")
        total += param_size
    print(f"Total: {total:,} parameters ({total / 1e9:.2f}B)")

    attn_params = sum(p.numel() for n, p in model.named_parameters() if 'attn' in n)
    mlp_params = sum(p.numel() for n, p in model.named_parameters() if 'mlp' in n)
    emb_params = sum(p.numel() for n, p in model.named_parameters() if 'embed' in n)

    print(f"\nGroup stats:")
    print(f"Attention layers: {attn_params / 1e6:.2f}M parameters")
    print(f"MLP layers: {mlp_params / 1e6:.2f}M parameters")
    print(f"Embedding layers: {emb_params / 1e6:.2f}M parameters")

def get_module_by_name(model, name):
    names = name.split('.')
    module = model
    for n in names:
        module = getattr(module, n)
    return module

def initialize_sparse_weights(tokenizer, base_model, model, sparse_layers, dataloader, device, args):
    if not args.sparse:
        return



    if args.sparse_method == "gradient":
        first_batch = next(iter(dataloader))
        first_batch = {k: v.to(device) for k, v in first_batch.items()}
        labels = first_batch["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        loss = base_model(**first_batch, labels=labels).loss
        loss.backward()

        base_gradients = []
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Linear):
                if any(t in name for t in ["attn", "mlp"]):
                    base_gradients.append(module.weight.grad)

        for layer, grad in zip(sparse_layers, base_gradients):
            layer.initialize_mask(grad)
    else:
        for layer in sparse_layers:
            layer.initialize_mask()

    base_model.zero_grad()
    model.zero_grad()


class HybridSparseLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, original_weight: torch.Tensor,
                 lowrank_module, sparsity: float, sparse_method: str = "random",
                 sparse_svd_rank: int = None, w_zero: bool = False, gamma: float = 0.5,
                 gamma_update: bool = False, gamma_all_apply: bool = True,
                 bias: bool = True, sparse_only: bool = False, svd_inverse: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gamma_all_apply = gamma_all_apply
        self.lowrank_module = lowrank_module
        self.sparse_only = sparse_only
        self.svd_inverse = svd_inverse

        self.weight = nn.Parameter(original_weight.clone())
        self.register_buffer('original_weight', original_weight.clone())
        self.register_buffer('mask', torch.ones_like(original_weight, dtype=torch.bool))
        self.mask_initialized = False
        self.sparsity = sparsity
        self.sparse_method = sparse_method
        self.sparse_svd_rank = sparse_svd_rank
        self.w_zero = w_zero

        if gamma_update:
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.register_buffer('gamma', torch.tensor(gamma))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def _get_butterfly_factor_mask(self, rows: int, cols: int, stride: int) -> torch.Tensor:
        """Generate mask for a single butterfly factor with given stride."""
        mask = torch.zeros(rows, cols, dtype=torch.bool, device=self.weight.device)

        # For each block in this stride
        for block_start in range(0, min(rows, cols), stride):
            block_size = stride // 2

            # Connect each element in first half to corresponding elements in second half
            for i in range(block_start, block_start + block_size):
                for j in range(block_start + block_size, min(block_start + stride, cols)):
                    if i < rows:  # Ensure we don't exceed matrix dimensions
                        mask[i, j] = True
                        if j < rows:  # For square matrices, make it symmetric
                            mask[j, i] = True

        return mask

    def initialize_mask(self, gradient=None):
        with torch.no_grad():
            if self.sparse_method == "butterfly":
                rows, cols = self.weight.shape
                n = min(rows, cols)
                log_n = int(math.log2(n))

                # Start with empty mask
                current_mask = torch.zeros_like(self.weight, dtype=torch.bool)
                current_mask.diagonal().fill_(True)

                # Try adding butterfly factors with increasing stride
                for stage in range(log_n):
                    stride = 1 << (stage + 1)  # 2, 4, 8, ...

                    # Get mask for this butterfly factor
                    factor_mask = self._get_butterfly_factor_mask(rows, cols, stride)

                    # Calculate new mask if we add this factor
                    potential_mask = current_mask | factor_mask

                    # Calculate sparsity if we add this factor
                    potential_sparsity = potential_mask.float().mean().item()

                    # If adding this factor would exceed our sparsity budget, stop
                    if potential_sparsity > self.sparsity:
                        print(potential_sparsity)
                        break

                    # Otherwise, add this factor
                    current_mask = potential_mask

                target_params = int(self.sparsity * self.weight.numel())
                current_params = torch.sum(current_mask).item()

                if current_params < target_params:
                    needed_params = target_params - current_params
                    print(f"Adding {needed_params} random parameters to reach target sparsity")

                    available_positions = ~current_mask

                    false_indices = torch.nonzero(available_positions.view(-1)).squeeze()

                    if false_indices.numel() > 0:
                        num_to_add = min(needed_params, false_indices.numel())
                        perm = torch.randperm(false_indices.numel())[:num_to_add]
                        indices_to_add = false_indices[perm]

                        current_mask.view(-1)[indices_to_add] = True

                        print(f"Target params: {target_params}")
                        print(f"Final number of True values: {torch.sum(current_mask).item()}")
                        print(f"Final sparsity: {torch.sum(current_mask).item() / current_mask.numel()}")

                self.mask = current_mask
            elif self.sparse_method == "gradient":
                # grad_abs = torch.abs(gradient).float()
                # k = int(self.sparsity * self.weight.numel())
                # threshold = torch.kthvalue(grad_abs.view(-1),
                #                            grad_abs.numel() - k)[0]
                # self.mask = grad_abs >= threshold
                grad_abs = torch.abs(gradient).float()
                k = int(self.sparsity * self.weight.numel())
                _, top_k_indices = torch.topk(grad_abs.view(-1), k)
                self.mask = torch.zeros_like(self.weight, dtype=torch.bool)
                self.mask.view(-1)[top_k_indices] = True
            elif self.sparse_method == "svd":
                # weight_float = self.weight.to(torch.float32)
                # U, S, Vh = torch.linalg.svd(weight_float, full_matrices=False)
                #
                # k = min(self.sparse_svd_rank, len(S))
                # U_k = U[:, :k]
                # S_k = S[:k]
                # Vh_k = Vh[:k, :]
                #
                # reconstructed = U_k @ torch.diag(S_k) @ Vh_k
                #
                # magnitude = torch.abs(reconstructed).float()
                # k = int(self.sparsity * self.weight.numel())
                # threshold = torch.kthvalue(magnitude.view(-1),
                #                            magnitude.numel() - k)[0]
                # self.mask = magnitude >= threshold
                weight_float = self.weight.to(torch.float32)
                U, S, Vh = torch.linalg.svd(weight_float, full_matrices=False)

                lowrank_k = int(self.lowrank_module.rank)

                if self.svd_inverse:
                    k = min(self.sparse_svd_rank, len(S) - lowrank_k)
                    start_idx = len(S) - k
                    end_idx = len(S)
                else:
                    k = min(self.sparse_svd_rank, len(S) - lowrank_k)
                    start_idx = lowrank_k
                    end_idx = start_idx + k

                U_k = U[:, start_idx:end_idx]
                S_k = S[start_idx:end_idx]
                Vh_k = Vh[start_idx:end_idx, :]

                reconstructed = U_k @ torch.diag(S_k) @ Vh_k

                magnitude = torch.abs(reconstructed).float()
                k = int(self.sparsity * self.weight.numel())
                threshold = torch.kthvalue(magnitude.view(-1),
                                           magnitude.numel() - k)[0]
                self.mask = magnitude >= threshold
            else:  # random
                k = int(self.sparsity * self.weight.numel())
                perm = torch.randperm(self.weight.numel())
                selected_indices = perm[:k]
                self.mask = torch.zeros_like(self.weight, dtype=torch.bool)
                self.mask.view(-1)[selected_indices] = True

            if self.w_zero:
                self.weight.data[~self.mask] = 0
            elif self.sparse_only:
                self.weight.data[~self.mask] = self.original_weight[~self.mask]

            self.mask_initialized = True
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.sparse_only:
            with torch.no_grad():
                self.weight.data[~self.mask] = self.original_weight[~self.mask]

            output = F.linear(input, self.weight)

            if self.training and self.lowrank_module is not None:
                dummy = (self.lowrank_module.matrix_A @ self.lowrank_module.matrix_B).sum() * 0
                output = output + dummy
        else:
            lowrank_weight = (self.lowrank_module.matrix_A @ self.lowrank_module.matrix_B).t()

            if self.gamma_all_apply:
                combined_weight = self.gamma * self.weight + (1 - self.gamma) * lowrank_weight
                effective_weight = torch.where(self.mask, combined_weight,
                                               combined_weight.detach() if not self.w_zero
                                               else torch.zeros_like(combined_weight))
            else:
                combined_weight = self.gamma * self.weight + (1 - self.gamma) * lowrank_weight
                effective_weight = torch.where(self.mask, combined_weight, lowrank_weight)

            output = F.linear(input, effective_weight)
        if self.bias is not None:
            output = output + self.bias
        return output


def create_hybrid_sparse_lowrank_model(base_model, lowrank_model, args):
    if not args.sparse:
        return lowrank_model, None

    sparse_layers = []

    for base_name, base_module in base_model.named_modules():
        if isinstance(base_module, nn.Linear):
            if any(t in base_name for t in ["attn", "mlp"]):
                lowrank_module = get_module_by_name(lowrank_model, base_name)

                if isinstance(lowrank_module, (SVDLowRankLinear)):
                    hybrid_layer = HybridSparseLinear(
                        base_module.in_features,
                        base_module.out_features,
                        base_module.weight.data,
                        lowrank_module,
                        sparsity=args.sparse_sparsity,
                        sparse_method=args.sparse_method,
                        sparse_svd_rank=args.sparse_svd_rank,
                        w_zero=args.sparse_w_zero,
                        gamma=args.sparse_gamma,
                        gamma_update=args.sparse_gamma_update,
                        gamma_all_apply=args.sparse_gamma_all_apply,
                        bias=base_module.bias is not None,
                        sparse_only=args.sparse_only,
                        svd_inverse=args.sparse_svd_inverse
                    )
                    sparse_layers.append(hybrid_layer)

                    parts = base_name.split('.')
                    current = lowrank_model
                    for part in parts[:-1]:
                        current = getattr(current, part)
                    setattr(current, parts[-1], hybrid_layer)

    return lowrank_model, sparse_layers

def main():
	# Parses arguments
    pipeline_name = "finetuner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments, LoRAArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, pipeline_args, lora_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, pipeline_args, lora_args = parser.parse_args_into_dataclasses()

    # Initialization
    finetuner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )

    args = type('Args', (), {})()

    for curr_args in [lora_args]:
        for key, value in vars(curr_args).items():
            setattr(args, key, value)


    print(data_args)
    dataset = Dataset(data_args)
    base_model = AutoModel.get_model(model_args)
    lowrank_model = AutoModel.get_model(model_args)

    if args.use_lora1:
        target_modules = []
        target_modules.append("attn")
        target_modules.append("mlp")

        if args.init_method == "SVD":
            lowrank_model = create_svd_low_rank_llama(
                model=lowrank_model.backend_model,
                rank_ratio=args.rank_ratio,
                target_modules=target_modules if target_modules else None,
            )

        print(print(sum(p.numel() for p in lowrank_model.parameters() if p.requires_grad) / 1e6))
        model, sparse_layers = create_hybrid_sparse_lowrank_model(
            base_model=base_model.backend_model,
            lowrank_model=lowrank_model,
            args=args
        )

        if args.sparse:
            initialize_sparse_weights(None, base_model, model.backend_model, sparse_layers, None, lowrank_model.device, args)
            total_true = sum(layer.mask.sum().item() for layer in sparse_layers if hasattr(layer, "mask"))
            total_params = sum(layer.weight.numel() for layer in sparse_layers if hasattr(layer, "weight"))
            if total_params > 0:
                ratio = total_true / total_params
                print(f"mask True : {ratio:.4f}")

        del base_model, lowrank_model
        torch.cuda.empty_cache()


    print(1)
    # Finetuning
    tuned_model = finetuner.tune(model=model, dataset=dataset)


if __name__ == '__main__':
    main()
