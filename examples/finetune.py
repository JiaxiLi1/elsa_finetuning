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

import sys
import os
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
from transformers import HfArgumentParser

from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)

from lmflow.datasets.dataset import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline


def main():
	# Parses arguments
    pipeline_name = "finetuner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, pipeline_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()

    # Initialization
    finetuner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )
    dataset = Dataset(data_args)
    model = AutoModel.get_model(model_args)

    backend_model = model.get_backend_model()

    if model_args.finetune_method == "elsa":
        from hybrid_lowrank_sparse_adapter import apply_hybrid_adapter, get_optimizer_param_groups


        print(f"Model type: {type(backend_model).__name__}")


        trainable_param_count = apply_hybrid_adapter(
            model=backend_model,
            scope=model_args.adapter_scope,
            rank=model_args.lora_rank,
            sparsity=model_args.sparsity,
            gamma=model_args.gamma,
            sparse_method=model_args.sparse_method,
            sparse_svd_rank=model_args.sparse_svd_rank,
            alpha=model_args.lora_alpha,
            cola_silu=model_args.cola_silu,
            cola_init=model_args.cola_init,
            svd_inverse=model_args.svd_inverse
        )
    elif model_args.finetune_method == "cola":
        from cola_lowrank_sparse_adapter import cola_apply_hybrid_adapter, cola_get_optimizer_param_groups


        print(f"Model type: {type(backend_model).__name__}")


        trainable_param_count = cola_apply_hybrid_adapter(
            model=backend_model,
            scope=model_args.adapter_scope,
            rank=model_args.lora_rank,
            sparsity=model_args.sparsity,
            gamma=model_args.gamma,
            sparse_method=model_args.sparse_method,
            sparse_svd_rank=model_args.sparse_svd_rank,
            alpha=model_args.lora_alpha,
            cola_silu=model_args.cola_silu,
            cola_init=model_args.cola_init,
            svd_inverse=model_args.svd_inverse
        )
    elif model_args.finetune_method == "lora":
        from lora_lowrank_sparse_adapter import lora_apply_hybrid_adapter, lora_get_optimizer_param_groups


        print(f"Model type: {type(backend_model).__name__}")


        trainable_param_count = lora_apply_hybrid_adapter(
            model=backend_model,
            scope=model_args.adapter_scope,
            rank=model_args.lora_rank,
            sparsity=model_args.sparsity,
            gamma=model_args.gamma,
            sparse_method=model_args.sparse_method,
            sparse_svd_rank=model_args.sparse_svd_rank,
            alpha=model_args.lora_alpha,
            cola_silu=model_args.cola_silu,
            cola_init=model_args.cola_init,
            svd_inverse=model_args.svd_inverse
        )
    elif model_args.finetune_method == "loro":
        from loro_optim import LOROAdamW
        from lowrank_adpt_module import (
            apply_lowrank_adpt_param,
            get_lowrank_adpt_param,
        )

        # apply lowrank adpater parameterization
        if model_args.adapter_scope is not None:
            apply_lowrank_adpt_param(
                backend_model,
                model_type="llama",
                scope=model_args.adapter_scope,
                rank=model_args.lora_rank,
                alpha=model_args.lora_alpha,
                init="xavier",
            )
        else:
            Warning(f"\nUsing full-rank model ...\n")

    # embedding_layer = backend_model.get_input_embeddings()
    # if embedding_layer is not None:
    #     embedding_layer.weight.requires_grad_(True)

    model._backend_model = backend_model

    # Finetuning
    tuned_model = finetuner.tune(model=model, dataset=dataset)


if __name__ == '__main__':
    main()
