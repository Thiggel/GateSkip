import os
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedModel,
    AutoConfig,
    PreTrainedTokenizer,
)
import math
import torch
from torch import nn
from peft import get_peft_model, LoraConfig, TaskType
from experiment.configs import EvaluationConfig, ModelConfig, FinetuneMode
from experiment.models.gating.GateLayer import GateLayer
from ..HasLayers import HasLayers
from experiment.models.gating import ModelGating
from experiment.models.mixture_of_depths import ModelMod
from experiment.models.early_exit import ModelEarlyExit


class ModelAdapter(HasLayers):
    """Handles model initialization and modification with LoRA and gating support"""

    def __init__(
        self,
        config: ModelConfig,
        evaluation_config: EvaluationConfig,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        seed: int,
    ):
        self.config = config
        self.evaluation_config = evaluation_config
        self.tokenizer = tokenizer
        self.device = device
        self.seed = seed

        self.lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = self._initialize_model()

        if not self.config.pretrained:
            self._init_embeddings()

    def _wrap_with_adaptive_compute(self, model: PreTrainedModel) -> PreTrainedModel:
        """Add gating or MoD wrappers to model components"""
        if not (
            self.config.use_gating or self.config.use_mod or self.config.use_early_exit
        ):
            return model

        # Get model dimensions
        d_model = (
            model.config.hidden_size
            if hasattr(model, "config")
            else self._infer_hidden_size(model)
        )

        if self.config.use_mod:
            routing = ModelMod(self.config, d_model)
            # Create MoD routing
            routing = ModelMod(self.config, d_model)

            # Get decoder layers
            layers = self.get_decoder_layers(model)

            # Wrap entire decoder layers
            for i, layer in enumerate(layers):
                wrapped_layer = routing.wrap_module(
                    f"layer_{i}",
                    layer,
                    parent=model,  # Parent is now the model itself
                    layer_idx=i,
                )
                # Replace the layer in the model's layers
                layers[i] = wrapped_layer

            # Add routing module to model
            model.add_module("mod", routing)
        elif self.config.use_gating:
            routing = ModelGating(self.config, d_model)
            # Original gating logic
            attn_gate = (
                GateLayer(d_model, self.config)
                if not self.config.one_gate_per_layer
                else None
            )
            mlp_gate = (
                GateLayer(d_model, self.config)
                if not self.config.one_gate_per_layer
                else None
            )

            layers = self.get_decoder_layers(model)
            for i, layer in enumerate(layers):
                if hasattr(layer, "self_attn"):
                    if self.config.gate_attention:
                        layer.self_attn = routing.wrap_module(
                            f"attn_{i}",
                            layer.self_attn,
                            parent=layer,
                            layer_idx=i,
                            gate=attn_gate,
                            frozen_gate=self.config.frozen_gate,
                        )
                    if self.config.gate_mlp and hasattr(layer, "mlp"):
                        layer.mlp = routing.wrap_module(
                            f"mlp_{i}",
                            layer.mlp,
                            parent=layer,
                            layer_idx=i,
                            gate=mlp_gate,
                            frozen_gate=self.config.frozen_gate,
                        )
                elif hasattr(layer, "attn"):
                    if self.config.gate_attention:
                        layer.attn = routing.wrap_module(
                            f"attn_{i}",
                            layer.attn,
                            parent=layer,
                            layer_idx=i,
                            gate=attn_gate,
                            frozen_gate=self.config.frozen_gate,
                        )
                    if self.config.gate_mlp and hasattr(layer, "ff"):
                        layer.ff = routing.wrap_module(
                            f"mlp_{i}",
                            layer.ff,
                            parent=layer,
                            layer_idx=i,
                            gate=mlp_gate,
                            frozen_gate=self.config.frozen_gate,
                        )

        elif self.config.use_early_exit:
            # Initialize early exit manager
            early_exit = ModelEarlyExit(self.config)

            # Get decoder layers
            layers = self.get_decoder_layers(model)

            # Wrap appropriate modules (usually attention modules)
            for i, layer in enumerate(layers):
                layers[i] = early_exit.wrap_module(
                    f"layer_{i}",
                    layer,
                    parent=model,  # Use the model as parent for output embeddings access
                    layer_idx=i,
                )

            # Add early exit manager to model for easier access
            model.add_module("early_exit", early_exit)

            # Store original layer forward methods to replace them
            original_layer_forwards = {}

            # Monkey patch each decoder layer's forward method to handle hidden state propagation
            for i, layer in enumerate(layers):
                original_layer_forwards[i] = layer.forward

                def make_forward_with_early_exit(layer_idx, orig_forward):
                    def forward_with_early_exit(hidden_states, *args, **kwargs):
                        # If early exit is enabled and we're not in the first layer
                        if hasattr(model, "early_exit") and layer_idx > 0:
                            # Propagate hidden states from earlier exits
                            hidden_states = (
                                model.early_exit.maybe_propagate_hidden_states(
                                    layer_idx, hidden_states
                                )
                            )

                        # Call original forward
                        outputs = orig_forward(hidden_states, *args, **kwargs)

                        return outputs

                    return forward_with_early_exit

                layer.forward = make_forward_with_early_exit(
                    i, original_layer_forwards[i]
                )

            # If this is a generation model, we need to handle the key-value cache updates
            if hasattr(model, "prepare_inputs_for_generation"):
                original_prepare_inputs = model.prepare_inputs_for_generation

                def prepare_inputs_with_early_exit(
                    input_ids, past=None, attention_mask=None, *args, **kwargs
                ):
                    # Call original method
                    inputs = original_prepare_inputs(
                        input_ids,
                        past=past,
                        attention_mask=attention_mask,
                        *args,
                        **kwargs,
                    )

                    # Set generation flag and increment step counter
                    if hasattr(model, "early_exit"):
                        model.early_exit.is_generating = True
                        model.early_exit.current_step += 1

                    # If we have past key-values, ensure they're properly propagated for early-exited tokens
                    if past is not None and hasattr(model, "early_exit"):
                        # For each layer's key-value states in the past
                        updated_past = list(past)

                        for layer_idx, layer_past in enumerate(past):
                            # Find any modules that exited in earlier layers
                            attn_module = None
                            for (
                                name,
                                module,
                            ) in model.early_exit.wrapped_modules.items():
                                if module.layer_idx == layer_idx:
                                    attn_module = module
                                    break

                            if (
                                attn_module is not None
                                and attn_module.current_exit_decision is not None
                            ):
                                # Update KV cache for this layer
                                updated_kv = model.early_exit.update_kv_cache(
                                    layer_idx,
                                    layer_past,
                                    attn_module.current_exit_decision,
                                )
                                updated_past[layer_idx] = updated_kv

                        # Replace with updated past
                        inputs["past_key_values"] = tuple(updated_past)

                    return inputs

                model.prepare_inputs_for_generation = prepare_inputs_with_early_exit

        # Add routing module to model
        if self.config.use_mod:
            model.add_module("mod", routing)
        elif self.config.use_gating:
            model.add_module("gating", routing)

        return model

    def _infer_hidden_size(self, model: PreTrainedModel) -> int:
        """Infer hidden size from model parameters"""
        # Try to find a parameter that would indicate the hidden size
        for param in model.parameters():
            if len(param.shape) >= 2:
                return param.shape[-1]
        raise ValueError("Could not infer hidden size from model parameters")

    def _load_from_checkpoint(self, model: nn.Module) -> nn.Module:
        checkpoint_path = os.path.join(
            os.environ["BASE_CACHE_DIR"],
            f"{self.evaluation_config.load_from_checkpoint}_{self.seed}.pt",
        )
        print("Loading from checkpoint", checkpoint_path)

        checkpoint = torch.load(checkpoint_path, weights_only=False)

        state_dict = (
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        )
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key.replace("model.model", "model").replace(
                "model.lm_head", "lm_head"
            )
            new_state_dict[new_key] = state_dict[key]

        state_dict = new_state_dict

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

        return model

    def _initialize_model(self) -> PreTrainedModel:
        """Initialize the model with appropriate configuration"""
        if self.config.pretrained:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, attn_implementation="eager"
            )
        else:
            config = AutoConfig.from_pretrained(self.config.model_name)
            config.vocab_size = self.tokenizer.vocab_size
            model = AutoModelForCausalLM.from_config(
                config, attn_implementation="eager"
            )

        model.use_cache = False
        model.train()

        model = self._remove_layers(model)

        # First set requires_grad=False for all parameters if not in FULL mode
        if self.config.finetune_mode != FinetuneMode.FULL:
            for param in model.parameters():
                param.requires_grad = False

        if self.evaluation_config.load_from_checkpoint:
            model = self._load_from_checkpoint(model)

        if self.config.use_kl_div_training:
            self.orig_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, attn_implementation="eager"
            )
            self.orig_model.use_cache = False
            self.orig_model.train()
            for param in self.orig_model.parameters():
                param.requires_grad = False

        # Add gating or MoD if needed
        model = self._wrap_with_adaptive_compute(model)

        # Apply LoRA if needed
        if self.config.finetune_mode == FinetuneMode.LORA:
            model = get_peft_model(model, self.lora_config)
            model.print_trainable_parameters()

        if self.evaluation_config.load_from_checkpoint:
            model = self._load_from_checkpoint(model)

        return model

    def _remove_layers(self, model: PreTrainedModel) -> PreTrainedModel:
        if self.config.remove_layers is not None:
            removed_layers = self._get_removed_layers(model)
            layers = self.get_decoder_layers(model)
            layers = nn.ModuleList(
                [layer for idx, layer in enumerate(layers) if idx not in removed_layers]
            )
            model = self.set_decoder_layers(model, layers)
        return model

    def _get_peft_model(self, model: PreTrainedModel) -> PreTrainedModel:
        if self.config.finetune_mode == FinetuneMode.LORA:
            print("Using LoRA")
            model = get_peft_model(model, self.lora_config)
            model.print_trainable_parameters()
            return model

        for param in model.parameters():
            param.requires_grad = False

        if self.config.finetune_mode == FinetuneMode.FULL:
            for param in model.parameters():
                param.requires_grad = True

        return model

    def _init_embeddings(self):
        std = 1.0 / math.sqrt(self.model.config.hidden_size)
        nn.init.normal_(self.model.get_input_embeddings().weight, std=std)
        nn.init.normal_(self.model.get_output_embeddings().weight, std=std)

    def _get_removed_layers(self, model: AutoModel) -> list[int]:
        return self._get_all_layers(model, self.config.remove_layers)
