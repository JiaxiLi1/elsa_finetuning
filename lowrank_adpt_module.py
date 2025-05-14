import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankAdapterLinear(nn.Module):
    def __init__(self, linear, rank, alpha, init):
        super(LowRankAdapterLinear, self).__init__()
        self.in_dim = linear.in_features
        self.out_dim = linear.out_features
        assert alpha >= 0, f"Invalid alpha: {alpha}, expected to be > 0."
        self.alpha = alpha
        self.init = init
        self.full_rank = min(self.in_dim, self.out_dim)

        self.register_buffer("weight", linear.weight.clone())
        self.weight_in = nn.Parameter(
            torch.randn(self.in_dim, rank).to(linear.weight),
            requires_grad=True,
        )
        self.weight_out = nn.Parameter(
            torch.randn(self.out_dim, rank).to(linear.weight),
            requires_grad=True,
        )

        self.bias = (
            nn.Parameter(
                torch.zeros(self.out_dim).to(linear.weight),
                requires_grad=True,
            )
            if hasattr(linear, "bias") and linear.bias is not None
            else None
        )

        self._init_weight()

    def extra_repr(self):
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, rank={self.rank}, alpha={self.alpha}, init={self.init}"

    def _init_weight(self):
        if self.init == "xavier":
            nn.init.xavier_normal_(self.weight_in)
            nn.init.xavier_normal_(self.weight_out)

        elif self.init == "spec_top":
            dtype = self.weight_in.dtype
            U, _, V = torch.svd(self.weight.float())
            self.weight_out.data = U[:, : self.rank].to(dtype)
            self.weight_in.data = V[:, : self.rank].to(dtype)
            self.weight_in.data *= np.sqrt(2 / (self.in_dim + self.rank))
            self.weight_out.data *= np.sqrt(2 / (self.out_dim + self.rank))

        elif self.init == "spec_low":  # better then spec_top
            dtype = self.weight_in.dtype
            U, _, V = torch.svd(self.weight.float())
            self.weight_out.data = U[:, -self.rank :].to(dtype)
            self.weight_in.data = V[:, -self.rank :].to(dtype)
            self.weight_in.data *= np.sqrt(2 / (self.in_dim + self.rank))
            self.weight_out.data *= np.sqrt(2 / (self.out_dim + self.rank))

        elif self.init == "eig_low":  # better then spec_top
            dtype = self.weight_in.dtype
            U, S, V = torch.svd(self.weight.float())
            self.weight_out.data = (
                U[:, -self.rank :] @ S[-self.rank :].sqrt().diag()
            ).to(dtype)
            self.weight_in.data = (
                V[:, -self.rank :] @ S[-self.rank :].sqrt().diag()
            ).to(dtype)
            self.weight_in.data *= np.sqrt(2 / (self.in_dim + self.rank))
            self.weight_out.data *= np.sqrt(2 / (self.out_dim + self.rank))

        elif self.init == "xavorth":
            dtype = self.weight_in.dtype
            nn.init.orthogonal_(self.weight_in.float()).to(dtype)
            nn.init.orthogonal_(self.weight_out.float()).to(dtype)
            self.weight_in.data *= np.sqrt(2 / (self.in_dim + self.rank))
            self.weight_out.data *= np.sqrt(2 / (self.out_dim + self.rank))

        elif self.init == "kaiming":
            nn.init.kaiming_normal_(self.weight_in)
            nn.init.kaiming_normal_(self.weight_out)

        elif self.init == "orth":
            dtype = self.weight_in.dtype
            nn.init.orthogonal_(self.weight_in.float()).to(dtype)
            nn.init.orthogonal_(self.weight_out.float()).to(dtype)

        elif self.init.startswith("randn"):
            std = float(self.init.split("_")[-1])
            nn.init.normal_(self.weight_in, mean=0, std=std)
            nn.init.normal_(self.weight_out, mean=0, std=std)

        elif "const" in self.init:
            const = float(self.init.split("_")[-1])
            assert const != 0
            self.weight_in.data.fill_(const)
            self.weight_out.data.fill_(const)

        else:
            raise ValueError(f"Invalid init method: {self.init}")

        self.weight.data -= self.alpha / self.rank * self.weight_out @ self.weight_in.T
        self.weight_out.data = self.weight_out.data.contiguous()
        self.weight_in.data = self.weight_in.data.contiguous()

        print(f"\nInitialized {self} with {self.init} method.")
        print(f"Absorbed the initilaization successfully by W <-- W - M @ N.T\n")

    @property
    def rank(self):
        return min(min(self.weight_in.shape), min(self.weight_out.shape))

    def forward(self, x):
        ## X = （W + alpha * B @ A.T） @ X
        out = x @ self.weight.T
        out = out + self.alpha / self.rank * x @ self.weight_in @ self.weight_out.T
        if self.bias is not None:
            out = out + self.bias

        return out


def apply_lowrank_adpt_param(model, model_type, scope, rank, alpha, init):
    if model_type == "llama":
        if scope == "qv":
            # 原始逻辑：只对 q_proj 和 v_proj 应用
            for i, layer in enumerate(model.model.layers):
                self_attn = layer.self_attn

                for proj_name in ["q_proj", "v_proj"]:
                    if hasattr(self_attn, proj_name):
                        sub_module = getattr(self_attn, proj_name)

                        if isinstance(sub_module, nn.Linear):
                            setattr(
                                self_attn,
                                proj_name,
                                LowRankAdapterLinear(sub_module, rank, alpha, init),
                            )

                            print(
                                f"layer.{i}.self_attn.{proj_name}: {sub_module} --> {getattr(self_attn, proj_name)}.\n"
                            )

                            del sub_module

        elif scope == "all":
            # 新增逻辑：对所有注意力权重和 MLP 权重应用
            for i, layer in enumerate(model.model.layers):
                # 处理注意力权重（qkvo）
                self_attn = layer.self_attn

                for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    if hasattr(self_attn, proj_name):
                        sub_module = getattr(self_attn, proj_name)

                        if isinstance(sub_module, nn.Linear):
                            setattr(
                                self_attn,
                                proj_name,
                                LowRankAdapterLinear(sub_module, rank, alpha, init),
                            )

                            print(
                                f"layer.{i}.self_attn.{proj_name}: {sub_module} --> {getattr(self_attn, proj_name)}.\n"
                            )

                            del sub_module

                # 处理 MLP 权重（up_proj, down_proj, gate_proj）
                mlp = layer.mlp

                for proj_name in ["up_proj", "down_proj", "gate_proj"]:
                    if hasattr(mlp, proj_name):
                        sub_module = getattr(mlp, proj_name)

                        if isinstance(sub_module, nn.Linear):
                            setattr(
                                mlp,
                                proj_name,
                                LowRankAdapterLinear(sub_module, rank, alpha, init),
                            )

                            print(
                                f"layer.{i}.mlp.{proj_name}: {sub_module} --> {getattr(mlp, proj_name)}.\n"
                            )

                            del sub_module
        else:
            raise ValueError(f"Invalid scope: {scope}, expected 'qv' or 'all'")
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented")


def get_lowrank_adpt_param(model, lr_scaler=1.0):
    lowrank_params_in = []
    lowrank_params_out = []
    for name, module in model.named_modules():
        if isinstance(module, LowRankAdapterLinear):
            if "weight_in" in module.state_dict():
                lowrank_params_in.append(module.weight_in)
            if "weight_out" in module.state_dict():
                lowrank_params_out.append(module.weight_out)

    id_lowrank_params = [id(p) for p in lowrank_params_in + lowrank_params_out]

    for name, p in model.named_parameters():
        if id(p) not in id_lowrank_params:
            if "embed_tokens" in name or "layer_norm" in name:
                continue
            p.requires_grad_(False)
    # for p in model.parameters():
    #     if id(p) not in id_lowrank_params:
    #         p.requires_grad_(False)

    param_groups = [
        {
            "type": "lowrank_in",
            "params": lowrank_params_in,
            "lr_scaler": lr_scaler,
        },
        {
            "type": "lowrank_out",
            "params": lowrank_params_out,
            "lr_scaler": lr_scaler,
        },
    ]

    return param_groups
