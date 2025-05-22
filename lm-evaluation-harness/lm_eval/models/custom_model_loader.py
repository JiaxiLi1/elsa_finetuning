import os
import sys
import torch
import logging
from typing import Optional, Union
from pathlib import Path

from ..api.registry import register_model
from .huggingface import HFLM

eval_logger = logging.getLogger(__name__)


@register_model("hf-custom", "custom")
class CustomHFLM(HFLM):

    def __init__(
            self,
            custom_modules_path: Optional[str] = None,
            **kwargs
    ):

        self.custom_modules_path = custom_modules_path
        super().__init__(**kwargs)

    def _add_potential_module_paths(self):

        potential_paths = [
            #
            ".",
            "..",
            "../..",
            #
            "../src",
            "../../src",
            "../lmflow",
            "../../lmflow",
            #
            # "/home/rtx3090/code_jiaxi/OwLore",
            # "/home/rtx3090/code_jiaxi/OwLore/src",
            # "/home/rtx3090/code_jiaxi/OwLore/src/lmflow",
        ]

        for path_str in potential_paths:
            abs_path = os.path.abspath(path_str)
            if os.path.exists(abs_path) and abs_path not in sys.path:
                sys.path.insert(0, abs_path)
                eval_logger.debug(f"add path to sys.path: {abs_path}")

        #
        eval_logger.debug(f"current Python path: {sys.path[:5]}...")  # 只显示前5个

    def _load_custom_modules(self, pretrained_path: str):

        model_dir = Path(pretrained_path)


        finetune_config_path = model_dir / "finetune_config.json"
        finetune_method = None

        if finetune_config_path.exists():
            try:
                import json
                with open(finetune_config_path, 'r') as f:
                    config = json.load(f)
                    finetune_method = config.get("finetune_method")
                    eval_logger.info(f": {finetune_method}")
            except Exception as e:
                eval_logger.warning(f"load finetuning config fail!: {e}")

        # 根据微调方法导入相应的模块
        imported_modules = {}

        if finetune_method in ["elsa", "cola", "loro", "lora"] or self.custom_modules_path:
            #
            self._add_potential_module_paths()

            try:
                #
                if finetune_method == "elsa":
                    eval_logger.info("import elsa...")
                    try:
                        import hybrid_lowrank_sparse_adapter as elsa_module
                        imported_modules["elsa"] = elsa_module
                        eval_logger.info(" successfully import hybrid_lowrank_sparse_adapter")
                    except ImportError as e:
                        eval_logger.warning(f"cannot import ELSA: {e}")
                        eval_logger.info("Make sure custom modules in the training environment are in the Python path")

                elif finetune_method == "cola":
                    eval_logger.info("import CoLA related module...")
                    try:
                        import cola_lowrank_sparse_adapter as cola_module
                        imported_modules["cola"] = cola_module
                        eval_logger.info("successfully import cola_lowrank_sparse_adapter")
                    except ImportError as e:
                        eval_logger.warning(f"cannot import CoLA: {e}")
                        eval_logger.info("Make sure custom modules in the training environment are in the Python path")

                elif finetune_method == "lora":
                    eval_logger.info("import LoRA...")
                    try:
                        import lora_lowrank_sparse_adapter as lora_module
                        imported_modules["lora"] = lora_module
                        eval_logger.info("successfully import lora_lowrank_sparse_adapter")
                    except ImportError as e:
                        eval_logger.warning(f"cannot import LoRA: {e}")
                        eval_logger.info("💡 Make sure custom modules in the training environment are in the Python path")

                elif finetune_method == "loro":
                    eval_logger.info("import LoRO...")
                    try:
                        import lowrank_adpt_module as loro_module
                        import loro_optim
                        imported_modules["loro"] = loro_module
                        imported_modules["loro_optim"] = loro_optim
                        eval_logger.info("successfully import LoRO")
                    except ImportError as e:
                        eval_logger.warning(f"cannot import LoRO: {e}")
                        eval_logger.info("Make sure custom modules in the training environment are in the Python path")

            except Exception as e:
                eval_logger.warning(f"Error importing fine-tuning module: {e}")

        #
        if self.custom_modules_path:
            modules_path = Path(self.custom_modules_path)
        else:
            #
            possible_locations = [
                model_dir / "custom_modules.py",
                model_dir / "modeling.py",
                model_dir / "custom_modeling.py",
                model_dir / "models.py",
                model_dir.parent / "custom_modules.py",
            ]

            modules_path = None
            for location in possible_locations:
                if location.exists():
                    modules_path = location
                    break

        if modules_path and modules_path.exists():
            eval_logger.info(f"Loading custom module files: {modules_path}")

            #
            module_dir = str(modules_path.parent)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)

            try:
                #
                import importlib.util
                spec = importlib.util.spec_from_file_location("custom_modules", modules_path)
                custom_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(custom_module)

                imported_modules["custom"] = custom_module
                eval_logger.info("Successfully loaded custom module file")
            except Exception as e:
                eval_logger.error(f"Failed to load custom module file: {e}")

        return imported_modules, finetune_method

    def _check_custom_modules_in_model(self, model, finetune_method):

        eval_logger.info("Checking for custom blocks in the model...")

        custom_modules_found = {}

        for name, module in model.named_modules():
            module_type = type(module).__name__

            #
            if "HybridSparseAdapter" in module_type and finetune_method == "elsa":
                custom_modules_found.setdefault("elsa", []).append(name)

            #
            elif "cola_HybridSparseAdapter" in module_type and finetune_method == "cola":
                custom_modules_found.setdefault("cola", []).append(name)

            #
            elif "lora_HybridSparseAdapter" in module_type and finetune_method == "lora":
                custom_modules_found.setdefault("lora", []).append(name)

            #
            elif finetune_method == "loro" and ("lowrank" in module_type.lower() or "loro" in module_type.lower()):
                custom_modules_found.setdefault("loro", []).append(name)

        #
        if custom_modules_found:
            eval_logger.info("Find the custom module:")
            for method, modules in custom_modules_found.items():
                eval_logger.info(f"   {method.upper()} module: {len(modules)} ")
                for module_name in modules[:3]:  #
                    eval_logger.info(f"    - {module_name}")
                if len(modules) > 3:
                    eval_logger.info(f"    - ... remaining {len(modules) - 3} ")
        else:
            eval_logger.warning(f"️  Not found in model {finetune_method} corresponding custom modules")

        return custom_modules_found

    def _create_custom_model_with_lora(self, pretrained: str, config, finetune_method: str, **kwargs):

        eval_logger.info(f"  Rebuilding the model architecture {finetune_method} ...")

        try:
            #
            from transformers import LlamaForCausalLM

            #
            revision = kwargs.get('revision', 'main')
            dtype = kwargs.get('dtype', 'auto')
            parallelize = kwargs.get('parallelize', False)
            gpus = kwargs.get('gpus', None)
            max_memory_per_gpu = kwargs.get('max_memory_per_gpu', None)
            max_cpu_memory = kwargs.get('max_cpu_memory', None)
            offload_folder = kwargs.get('offload_folder', './offload')

            #
            model_kwargs = dict(kwargs)
            special_handling_args = {'peft', 'delta', 'autogptq', 'gptqmodel'}
            for arg in special_handling_args:
                model_kwargs.pop(arg, None)

            model_kwargs.update(
                self._get_accelerate_args(
                    parallelize=parallelize,
                    device_map=kwargs.get("device_map", None),
                    max_memory_per_gpu=max_memory_per_gpu,
                    max_cpu_memory=max_cpu_memory,
                    offload_folder=offload_folder,
                    gpus=gpus,
                )
            )

            #
            processed_args = {
                'parallelize', 'gpus', 'max_memory_per_gpu',
                'max_cpu_memory', 'offload_folder', 'device_map',
                'dtype', 'revision', 'trust_remote_code'
            }
            for arg in processed_args:
                model_kwargs.pop(arg, None)

            from .utils import get_dtype
            torch_dtype = get_dtype(dtype)

            #
            eval_logger.info(" Creating the base model architecture...")

            #
            base_model_path = Path(pretrained) / "base_model"
            if base_model_path.exists():
                eval_logger.info(f" Load the base model from the base_model directory: {base_model_path}")
                base_model = LlamaForCausalLM.from_pretrained(
                    str(base_model_path),
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    **model_kwargs
                )
            else:
                eval_logger.info("️  Creating a new base model architecture...")
                base_model = LlamaForCausalLM(config)

            #
            if finetune_method == "lora":
                eval_logger.info(" apply LoRA module to base model...")
                model = self._apply_lora_to_model(base_model, finetune_method)
            elif finetune_method == "elsa":
                eval_logger.info(" apply ELSA module to base model...")
                model = self._apply_elsa_to_model(base_model, finetune_method, pretrained)
            elif finetune_method == "cola":
                eval_logger.info(" apply CoLA module to base model...")
                model = self._apply_cola_to_model(base_model, finetune_method)
            elif finetune_method == "loro":
                eval_logger.info(" apply LoRO module to base model...")
                model = self._apply_loro_to_model(base_model, finetune_method)
            else:
                eval_logger.warning(f"  Unknown fine-tuning method: {finetune_method}，using the base model")
                model = base_model

            #
            eval_logger.info(" Loading model weights...")
            model_file = Path(pretrained) / "model.safetensors"
            if model_file.exists():
                #
                try:
                    import safetensors.torch
                    state_dict = safetensors.torch.load_file(str(model_file))

                    #
                    model_state_dict = model.state_dict()
                    matched_weights = {}
                    missing_weights = []
                    unexpected_weights = []

                    for name, param in state_dict.items():
                        if name in model_state_dict:
                            if param.shape == model_state_dict[name].shape:
                                matched_weights[name] = param
                            else:
                                eval_logger.warning(
                                    f"  Weight shape mismatch: {name}, expect {model_state_dict[name].shape}, actual {param.shape}")
                        else:
                            unexpected_weights.append(name)

                    for name in model_state_dict:
                        if name not in matched_weights:
                            missing_weights.append(name)

                    eval_logger.info(f" Weight shape match: {len(matched_weights)} ")
                    if missing_weights:
                        eval_logger.warning(f"️  Missing weights: {len(missing_weights)}  (Top 5: {missing_weights[:5]})")
                    if unexpected_weights:
                        eval_logger.warning(
                            f"️  Excess weight: {len(unexpected_weights)}  (Top 5: {unexpected_weights[:5]})")

                    #
                    model.load_state_dict(matched_weights, strict=False)

                except Exception as e:
                    eval_logger.error(f"use safetensors failed to load weights: {e}")
                    #
                    eval_logger.info(" Falling back to standard weight loading method...")
                    missing_keys, unexpected_keys = model.load_state_dict(
                        torch.load(model_file, map_location="cpu"), strict=False
                    )
                    if missing_keys:
                        eval_logger.warning(f"  missing_keys: {len(missing_keys)} ")
                    if unexpected_keys:
                        eval_logger.warning(f"️  unexpected_keys: {len(unexpected_keys)} ")

            return model

        except Exception as e:
            eval_logger.error(f" Rebuilding custom model failed: {e}")
            import traceback
            eval_logger.debug(traceback.format_exc())
            raise

    def _apply_lora_to_model(self, model, finetune_method):

        eval_logger.info(" apply LoRA module...")

        try:
            import lora_lowrank_sparse_adapter

            #
            self._replace_linear_with_adapters(
                model,
                lora_lowrank_sparse_adapter.lora_HybridSparseAdapter,
                scope="all",  #
                rank=8,  #
                alpha=16.0,  #
                sparsity=0.001,
                gamma=0.7,
                sparse_method="svd",
                sparse_svd_rank=256,
                svd_inverse=False,
                cola_silu=False,  #
                cola_init=False
            )
            return model

        except Exception as e:
            eval_logger.warning(f" apply LoRA fail: {e}")
            return model

    def _apply_elsa_to_model(self, model, finetune_method, model_path=None):

        eval_logger.info("apply  ELSA...")

        try:
            import hybrid_lowrank_sparse_adapter

            #
            #
            cola_silu_param = self._infer_cola_silu_from_weights(model, model_path)

            self._replace_linear_with_adapters(
                model,
                hybrid_lowrank_sparse_adapter.HybridSparseAdapter,
                scope="all",
                rank=6,  # ELSA  rank=6
                alpha=12.0,  # ELSA alpha=12
                sparsity=0.001,
                gamma=0.7,
                sparse_method="svd",
                sparse_svd_rank=256,
                svd_inverse=False,
                cola_silu=cola_silu_param,
                cola_init=False
            )
            return model

        except Exception as e:
            eval_logger.warning(f"apply ELSA fail: {e}")
            return model

    def _apply_cola_to_model(self, model, finetune_method):

        eval_logger.info("apply CoLA ...")

        try:
            import cola_lowrank_sparse_adapter

            #
            self._replace_linear_with_adapters(
                model,
                cola_lowrank_sparse_adapter.cola_HybridSparseAdapter,
                scope="all",
                rank=8,  # CoLA rank=8
                alpha=16.0,  # CoLA  alpha=16
                sparsity=0.001,
                gamma=0.7,
                sparse_method="svd",
                sparse_svd_rank=256,
                svd_inverse=False,
                cola_silu=True,  # CoLA SiLU
                cola_init=False
            )
            return model

        except Exception as e:
            eval_logger.warning(f"apply CoLA fail: {e}")
            return model

    def _apply_loro_to_model(self, model, finetune_method):

        eval_logger.info("apply LoRO ...")

        try:
            import lowrank_adpt_module

            #
            lowrank_adpt_module.apply_lowrank_adpt_param(
                model,
                model_type="llama",
                scope="all",  # adapter_scope
                rank=8,  # lora_rank
                alpha=16.0,  # lora_alpha_custom
                init="spec_low"
            )
            return model

        except Exception as e:
            eval_logger.warning(f"apply LoRO fail: {e}")
            return model

    def _infer_cola_silu_from_weights(self, model, model_path=None):
        try:
            #
            if model_path:
                path_str = str(model_path).lower()
                if 'cola_silu_false' in path_str or 'nosilu' in path_str:
                    eval_logger.info("obtain from path cola_silu: False")
                    return False
                elif 'cola_silu_true' in path_str or 'withsilu' in path_str:
                    eval_logger.info("obtain from path cola_silu: True")
                    return True

            #
            eval_logger.info("use default cola_silu: True")
            return True

        except Exception as e:
            eval_logger.warning(f"obtain cola_silu fail: {e}")
            return True

    def _replace_linear_with_adapters(self, model, adapter_class, scope="all", rank=8, alpha=16.0,
                                      sparsity=0.001, gamma=0.7, sparse_method="svd",
                                      sparse_svd_rank=256, svd_inverse=False,
                                      cola_silu=False, cola_init=False):

        eval_logger.info(f"replace Linear to {adapter_class.__name__}...")

        #
        module_names_dict = {
            "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "qkv": ["q_proj", "k_proj", "v_proj"],
            "qv": ["q_proj", "v_proj"],
            "qk": ["q_proj", "k_proj"],
            "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mlp": ["gate_proj", "up_proj", "down_proj"],
        }

        module_names = module_names_dict.get(scope, ["q_proj", "v_proj"])

        #
        model_layers = None
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            model_layers = model.model.layers
        elif hasattr(model, "layers"):
            model_layers = model.layers

        if model_layers is None:
            raise ValueError("cant find modul layer")

        replaced_count = 0

        #
        for i, layer in enumerate(model_layers):
            #
            if hasattr(layer, "self_attn"):
                attn_module = layer.self_attn
                for name, sub_module in list(attn_module.named_children()):
                    if isinstance(sub_module, torch.nn.Linear) and any(n in name for n in module_names):
                        #
                        adapter_kwargs = {
                            'rank': rank,
                            'alpha': alpha,
                            'sparsity': sparsity,
                            'gamma': gamma,
                            'sparse_method': sparse_method,
                            'sparse_svd_rank': sparse_svd_rank,
                            'svd_inverse': svd_inverse,
                            'init': "svd",
                            'cola_silu': cola_silu,
                            'cola_init': cola_init
                        }

                        adapter_module = adapter_class(sub_module, **adapter_kwargs)
                        setattr(attn_module, name, adapter_module)
                        replaced_count += 1

                        eval_logger.debug(f" replace layer.{i}.self_attn.{name}")

            #
            if hasattr(layer, "mlp") and scope in ["mlp", "all"]:
                mlp_module = layer.mlp
                for name, sub_module in list(mlp_module.named_children()):
                    if isinstance(sub_module, torch.nn.Linear) and name in module_names:
                        adapter_kwargs = {
                            'rank': rank,
                            'alpha': alpha,
                            'sparsity': sparsity,
                            'gamma': gamma,
                            'sparse_method': sparse_method,
                            'sparse_svd_rank': sparse_svd_rank,
                            'svd_inverse': svd_inverse,
                            'init': "svd",
                            'cola_silu': cola_silu,
                            'cola_init': cola_init
                        }

                        adapter_module = adapter_class(sub_module, **adapter_kwargs)
                        setattr(mlp_module, name, adapter_module)
                        replaced_count += 1

                        eval_logger.debug(f" replace layer.{i}.mlp.{name}")

        eval_logger.info(f"Successfully replaced {replaced_count}  Linear ")

    def _create_model(self, pretrained: str, **kwargs):

        eval_logger.info(f"Loading models using a custom loader: {pretrained}")

        #
        imported_modules, finetune_method = self._load_custom_modules(pretrained)

        #
        try:
            #
            complete_model_path = Path(pretrained)
            if complete_model_path.exists():
                eval_logger.info("Try loading the full model directly...")

                #
                pytorch_files = list(complete_model_path.glob("pytorch_model*.bin")) + \
                                list(complete_model_path.glob("model*.safetensors"))

                if pytorch_files:
                    eval_logger.info(f" Find the model file: {[f.name for f in pytorch_files]}")

                    #
                    try:
                        eval_logger.info("use HuggingFace standard way to build model...")

                        # 创建配置
                        from transformers import AutoConfig
                        config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)

                        #
                        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                            eval_logger.info("create tokenizer...")
                            revision = kwargs.get('revision', 'main')
                            trust_remote_code = kwargs.get('trust_remote_code', True)
                            use_fast_tokenizer = kwargs.get('use_fast_tokenizer', True)
                            add_bos_token = kwargs.get('add_bos_token', False)

                            self._create_tokenizer(
                                pretrained=pretrained,
                                tokenizer=kwargs.get('tokenizer', None),
                                revision=revision,
                                trust_remote_code=trust_remote_code,
                                use_fast_tokenizer=use_fast_tokenizer,
                                add_bos_token=add_bos_token,
                            )

                        #
                        if "custom" in imported_modules and hasattr(imported_modules["custom"], 'create_model'):
                            #
                            eval_logger.info("Using a custom create_model function...")
                            model = imported_modules["custom"].create_model(config)
                        else:
                            #
                            eval_logger.info("  Rebuilding the model architecture to include custom modules...")

                            model = self._create_custom_model_with_lora(
                                pretrained, config, finetune_method, **kwargs
                            )

                        self._model = model
                        eval_logger.info("successfully load model!")

                        # 检查模型中的自定义模块
                        if finetune_method:
                            self._check_custom_modules_in_model(model, finetune_method)

                        return

                    except Exception as e:
                        eval_logger.warning(f"fail to load full model: {e}")
                        import traceback
                        eval_logger.debug(traceback.format_exc())

        except Exception as e:
            eval_logger.warning(f"custom config fail: {e}")
            import traceback
            eval_logger.debug(traceback.format_exc())

        #
        eval_logger.info("back to standardHuggingFace加载方法...")
        #
        if 'trust_remote_code' not in kwargs:
            kwargs['trust_remote_code'] = True
        super()._create_model(pretrained, **kwargs)

        #
        if finetune_method and hasattr(self, '_model'):
            self._check_custom_modules_in_model(self._model, finetune_method)