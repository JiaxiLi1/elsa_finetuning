import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class HybridSparseAdapter(nn.Module):

    def __init__(self, linear, rank, sparsity=0.05, gamma=0.5, sparse_method="svd",
                 sparse_svd_rank=None, alpha=1.0, init="svd", cola_silu=False,
                 cola_init=False, svd_inverse=False):
        super(HybridSparseAdapter, self).__init__()
        self.in_dim = linear.in_features
        self.out_dim = linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.init = init
        self.sparsity = sparsity
        self.sparse_method = sparse_method
        self.sparse_svd_rank = sparse_svd_rank
        self.cola_silu = cola_silu
        self.cola_init = cola_init
        self.svd_inverse = svd_inverse

        # Save original weight for initialization
        self.register_buffer('original_weight', linear.weight.clone())

        # Create low-rank factors
        self.matrix_A = nn.Parameter(
            torch.zeros(self.in_dim, self.rank).to(linear.weight.device),
            requires_grad=True
        )
        self.matrix_B = nn.Parameter(
            torch.zeros(self.rank, self.out_dim).to(linear.weight.device),
            requires_grad=True
        )

        # Sparse component parameters
        self.values = None  # Will be initialized in initialize_mask
        self.register_buffer('selected_col_indices', None)  # Will be set in initialize_mask

        self.register_buffer('gamma', torch.tensor(gamma))

        # Bias parameters
        if hasattr(linear, "bias") and linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.clone())
        else:
            self.bias = None

        if self.cola_silu:
            self.activation = nn.SiLU()

        if self.cola_init:
            self.initialize_cola()
        else:
            self.apply_svd_init(linear.weight.data)

        # self.original_weight.data -= self.alpha / self.rank * self.matrix_B.t() @ self.matrix_A.t()

        self.initialize_mask()

    def initialize_cola(self):
        """
        Initialize low-rank matrices using CoLA method
        """
        with torch.no_grad():
            target_sdv = (self.in_dim + self.out_dim) ** (-1 / 2)
            scale_factor = self.rank ** (-1 / 4) * target_sdv ** (1 / 2)
            self.matrix_A.data.copy_(torch.randn_like(self.matrix_A) * scale_factor)
            self.matrix_B.data.copy_(torch.randn_like(self.matrix_B) * scale_factor)

    def apply_svd_init(self, weight_matrix):
        """
        Initialize low-rank matrices using SVD decomposition
        """
        # Transpose weight matrix to match dimensions
        weight_matrix = weight_matrix.t()
        original_dtype = weight_matrix.dtype
        weight_matrix = weight_matrix.to(torch.float32)

        # Perform SVD
        U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)

        # Select top-k components
        U_r = U[:, :self.rank]  # shape: [in_dim, rank]
        S_r = S[:self.rank]  # shape: [rank]
        V_r = Vh[:self.rank, :].t()  # shape: [out_dim, rank]

        # Calculate matrix_A and matrix_B
        S_sqrt = torch.sqrt(S_r)
        with torch.no_grad():
            A = U_r * S_sqrt.view(1, -1)
            B = (V_r * S_sqrt.view(1, -1)).t()

            if original_dtype != torch.float32:
                A = A.to(original_dtype)
                B = B.to(original_dtype)

            self.matrix_A.data.copy_(A)
            self.matrix_B.data.copy_(B)

    def initialize_mask(self):
        """Initialize sparse mask based on selected method"""
        num_cols_to_keep = max(1, int(self.sparsity * self.in_dim))

        if self.sparse_method == "svd":
            weight_float = self.original_weight.to(torch.float32)
            U, S, Vh = torch.linalg.svd(weight_float.t(), full_matrices=False)

            if self.svd_inverse:
                k = min(self.sparse_svd_rank or self.rank, len(S) - self.rank)
                start_idx = len(S) - k
                end_idx = len(S)
            else:
                k = min(self.sparse_svd_rank or self.rank, len(S) - self.rank)
                start_idx = self.rank
                end_idx = start_idx + k

            U_k = U[:, start_idx:end_idx]
            S_k = S[start_idx:end_idx]
            Vh_k = Vh[start_idx:end_idx, :]

            reconstructed = U_k @ torch.diag(S_k) @ Vh_k
            col_norms = torch.norm(reconstructed.t(), dim=0)  # [in_dim]
            _, topk_cols = torch.topk(col_norms, num_cols_to_keep, largest=True)
        elif self.sparse_method == "random":
            # Randomly select columns
            topk_cols = torch.randperm(self.in_dim)[:num_cols_to_keep]

        topk_cols, _ = torch.sort(topk_cols)

        # Set selected column indices
        self.register_buffer("selected_col_indices", topk_cols)

        # Initialize values for sparse component
        sparse_weight = self.original_weight[:, topk_cols]
        self.values = nn.Parameter(sparse_weight.clone())

    def forward(self, x):
        out = F.linear(x, self.original_weight)

        if self.cola_silu:
            # Use SiLU activation between A and B matrices
            temp = torch.matmul(x, self.matrix_A)
            temp = self.activation(temp)
            low_rank_output = torch.matmul(temp, self.matrix_B)
        else:
            # Standard low-rank computation
            temp = torch.matmul(x, self.matrix_A)
            low_rank_output = torch.matmul(temp, self.matrix_B)

        # Sparse component
        input_selected = x[:, :, self.selected_col_indices]
        sparse_output = F.linear(input_selected, self.values)

        # Combine using gamma parameter
        combined_adapter_output = self.gamma * low_rank_output + (1 - self.gamma) * sparse_output
        # combined_adapter_output = low_rank_output
        out = out + self.alpha / self.rank * combined_adapter_output

        # Add bias if it exists
        if self.bias is not None:
            out = out + self.bias

        return out


def apply_hybrid_adapter(model, scope="qv", rank=8, sparsity=0.05, gamma=0.5,
                         sparse_method="svd", sparse_svd_rank=None, alpha=1.0,
                         cola_silu=False, cola_init=False, svd_inverse=False):
    """
    Apply hybrid low-rank + sparse adapter to LLaMA or other Transformer models

    Args:
        model: Model to modify
        scope: Modules to adapt ("all", "qkv", "qv", etc)
        rank: Rank for low-rank component
        sparsity: Sparsity (percentage of columns to keep)
        gamma: Initial value for mixing parameter
        sparse_method: Method for selecting sparse columns ("svd", "random")
        sparse_svd_rank: Rank to use for SVD-based mask initialization
        alpha: Scaling factor for low-rank component
        cola_silu: Whether to use SiLU activation in low-rank component
        cola_init: Whether to use CoLA initialization
        svd_inverse: Whether to use inverse SVD for sparse component

    Returns:
        trainable_param_count: Number of trainable parameters
    """
    # Define target modules based on scope
    module_names_dict = {
        "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "alll": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "qkv": ["q_proj", "k_proj", "v_proj"],
        "qv": ["q_proj", "v_proj"],
        "qk": ["q_proj", "k_proj"],
        "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "self_attn": ["q_proj", "k_proj", "v_proj"],
        "ff": ["o_proj", "gate_proj", "up_proj", "down_proj"]
    }

    module_names = module_names_dict.get(scope, ["q_proj", "v_proj"])  # Default to qv

    print(f"\nApplying hybrid adapter to model, scope: {scope}, rank: {rank}, sparsity: {sparsity}, "
          f"gamma: {gamma}, sparse_method: {sparse_method}, cola_init: {cola_init}, svd_inverse: {svd_inverse}")

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad_(False)

    # embedding_layer = model.get_input_embeddings()
    # if embedding_layer is not None:
    #     embedding_layer.weight.requires_grad_(True)

    # Collect adapter parameters
    adapter_params = []

    # Handle LLaMA model architecture
    model_layers = None

    # Check different model architectures
    if hasattr(model, "model"):
        # Most LLaMA implementations have this structure
        if hasattr(model.model, "layers"):
            model_layers = model.model.layers
        elif hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
            model_layers = model.model.decoder.layers
    elif hasattr(model, "layers"):
        # Direct access to layers
        model_layers = model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        # Some implementations use transformer.layers
        model_layers = model.transformer.layers
    elif hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        # Some decoder-only models
        model_layers = model.decoder.layers

    if model_layers is None:
        raise ValueError("Could not find LLaMA layers in model. Please check model architecture.")

    # Process LLaMA layers
    for i, layer in enumerate(model_layers):
        # Check self-attention components
        if hasattr(layer, "self_attn"):
            attn_module = layer.self_attn
            for name, sub_module in attn_module.named_children():
                if isinstance(sub_module, nn.Linear) and any(n in name for n in module_names):
                    hybrid_module = HybridSparseAdapter(
                        sub_module, rank, sparsity, gamma, sparse_method,
                        sparse_svd_rank, alpha, init="svd",
                        cola_silu=cola_silu, cola_init=cola_init,
                        svd_inverse=svd_inverse
                    )
                    setattr(attn_module, name, hybrid_module)
                    adapter_params.extend([p for p in hybrid_module.parameters() if p.requires_grad])
                    print(f"Replaced layer.{i}.self_attn.{name}: "
                          f"Original shape {sub_module.weight.shape} -> "
                          f"Low-rank A:{hybrid_module.matrix_A.shape}, "
                          f"B:{hybrid_module.matrix_B.shape}, "
                          f"Sparse columns: {len(hybrid_module.selected_col_indices)}")

        # Check MLP components
        if hasattr(layer, "mlp"):
            mlp_module = layer.mlp
            for name, sub_module in mlp_module.named_children():
                if isinstance(sub_module, nn.Linear) and (
                        scope in ["mlp", "alll", "ff", "all"] or name in module_names):
                    hybrid_module = HybridSparseAdapter(
                        sub_module, rank, sparsity, gamma, sparse_method,
                        sparse_svd_rank, alpha, init="svd",
                        cola_silu=cola_silu, cola_init=cola_init,
                        svd_inverse=svd_inverse
                    )
                    setattr(mlp_module, name, hybrid_module)
                    adapter_params.extend([p for p in hybrid_module.parameters() if p.requires_grad])
                    print(f"Replaced layer.{i}.mlp.{name}: "
                          f"Original shape {sub_module.weight.shape} -> "
                          f"Low-rank A:{hybrid_module.matrix_A.shape}, "
                          f"B:{hybrid_module.matrix_B.shape}, "
                          f"Sparse columns: {len(hybrid_module.selected_col_indices)}")

    # Statistics
    trainable_param_count = sum(p.numel() for p in adapter_params)
    total_param_count = sum(p.numel() for p in model.parameters())
    trainable_ratio = trainable_param_count / total_param_count

    print(f"\nAdapter application completed!")
    print(f"Trainable parameters: {trainable_param_count:,} ({trainable_param_count / 1e6:.2f}M)")
    print(f"Total parameters: {total_param_count:,} ({total_param_count / 1e6:.2f}M)")
    print(f"Trainable ratio: {trainable_ratio:.2%}\n")

    return trainable_param_count


def get_optimizer_param_groups(model, weight_decay=0.0, lr=0.0002, gamma_lr=0.0001):
    """
    Get optimizer parameter groups for training
    """
    a_params = []  # Low-rank A matrix
    b_params = []  # Low-rank B matrix
    sparse_params = []  # Sparse values

    for name, module in model.named_modules():
        if isinstance(module, HybridSparseAdapter):
            a_params.append(module.matrix_A)
            b_params.append(module.matrix_B)
            sparse_params.append(module.values)

    param_groups = [
        {
            "params": a_params + b_params + sparse_params,
            "weight_decay": weight_decay,
            "lr": lr,
            "name": "adapter_params"
        }
    ]

    return param_groups