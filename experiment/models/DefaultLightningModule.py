import math
from lightning import LightningModule
from transformers import PreTrainedTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.optim import AdamW
import torch
import torch.nn.functional as F
from typing import Optional

from experiment.configs import ModelConfig, TrainingConfig, DataConfig, EvaluationConfig
from experiment.configs.ModelConfig import FinetuneMode

from .model_adapter import ModelAdapter
from .MetricsLogger import MetricsLogger
from .HasLayers import HasLayers
from .gating.GatingStatsCollector import GatingStatsCollector



class DefaultLightningModule(LightningModule, HasLayers):
    """Main Lightning Module for language model training"""

    def __init__(
        self,
        config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig,
        evaluation_config: EvaluationConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.config = config
        self.training_config = training_config
        self.data_config = data_config
        self.evaluation_config = evaluation_config
        self.tokenizer = tokenizer

        self.model_adapter = ModelAdapter(
            self.config,
            self.evaluation_config,
            self.tokenizer,
            self.device,
            seed,
        )
        self.model = self.model_adapter.model
        self.old_forward = self.model.forward
        self.model.forward = self.forward
        self.percent_tokens_skipped = []

        self.metrics_logger = MetricsLogger(
            self, self.tokenizer, self.data_config.batch_size
        )

        self.gating_stats_collector = GatingStatsCollector()

    def on_before_optimizer_step(self, _):
        """Log gradient norms before optimization step"""
        capacity = self.config.mod_capacity_factor + (
            1 - self.config.mod_capacity_factor
        ) * (1.0 / (self.global_step / 1500 + 1))

        if self.config.use_mod:
            self.model.mod.update_capacity(capacity)

            self.log("mod_capacity", capacity, sync_dist=True)

        # self.metrics_logger.log_gradient_norms()

    def give_global_step_to_gates(self, input_ids):
        if self.config.use_gating:
            validity_mask = (input_ids != self.tokenizer.pad_token_id) & (
                input_ids != self.tokenizer.eos_token_id
            )
            for module in self.model.gating.wrapped_modules.values():
                module.global_step = self.global_step
                module.current_input_ids = input_ids
                module.current_validity_mask = validity_mask

    def forward(self, input_ids, **kwargs):
        if hasattr(self.model, "early_exit"):
            self.model.early_exit.total_tokens = input_ids.size(-1)
        self.give_global_step_to_gates(input_ids)
        output = self.old_forward(input_ids, **kwargs)

        if self.config.use_gating:
            for name, module in self.model.gating.wrapped_modules.items():
                self.percent_tokens_skipped.append(
                    module.current_percent_tokens_skipped
                )

            if not self.training:
                self.gating_stats_collector.collect(self.model)
        elif self.config.use_mod:
            percent_skipped = []
            for module in self.model.mod.wrapped_modules.values():
                self.percent_tokens_skipped.append(
                    1 - module.current_percent_tokens_processed
                )
        elif self.config.use_early_exit and hasattr(self.model, "early_exit"):
            # Get early exit statistics
            early_exit_stats = self.model.early_exit.compute_early_exit_statistics()
            if "compute_saved" in early_exit_stats:
                self.percent_tokens_skipped.append(early_exit_stats["compute_saved"])

        return output

    def generate(self, *args, **kwargs):
        # Set generation flag and reset statistics
        if self.config.use_early_exit and hasattr(self.model, "early_exit"):
            self.model.early_exit.is_generating = True
            self.model.early_exit.reset_statistics()


        self.metrics_logger.dump_first_batch(kwargs)
        output = self.model.generate(*args, **kwargs)

        print(
            "Output: ",
            self.tokenizer.decode(output[0]),
        )

        print("Tokens skipped: ", torch.mean(torch.tensor(self.percent_tokens_skipped)))

        # Log early exit statistics after generation
        if self.config.use_early_exit and hasattr(self.model, "early_exit"):
            # Update total tokens based on generation output
            if len(output.shape) == 2:
                self.model.early_exit.total_tokens = output.shape[0] * output.shape[1]
            else:
                self.model.early_exit.total_tokens = output.shape[0]

            # Compute and log statistics
            early_exit_stats = self.model.early_exit.compute_early_exit_statistics()
            print(f"Early exit stats: {early_exit_stats}")

            # Update percent_tokens_skipped for consistency with other methods
            if "compute_saved" in early_exit_stats:
                self.percent_tokens_skipped.append(early_exit_stats["compute_saved"])

            # Reset generation flag
            self.model.early_exit.is_generating = False

        return output

    def sample_generate(self):
        string = self.tokenizer.encode(
            "My dog is ",
            return_tensors="pt",
        ).to(self.device)

        generated = self.generate(
            input_ids=string,
            max_length=100,
            max_new_tokens=100,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        print("Sample generation: ", self.tokenizer.decode(generated[0]))

    def on_validation_start(self):
        self.sample_generate()

    def configure_optimizers(self):
        base_lr = self.training_config.learning_rate
        adam_params = {
            "betas": (0.9, 0.95),
            "weight_decay": 0.001,
        }

        model_params = [
            param
            for name, param in self.model.named_parameters()
            if param.requires_grad and "router" not in name and "predictor" not in name
        ]

        mod_params = [
            param
            for name, param in self.model.named_parameters()
            if "router" in name or "predictor" in name
        ]

        parameters = [
            {"params": model_params, "lr": base_lr},
            {"params": mod_params, "lr": 1e-4},
        ]

        if torch.cuda.is_available() and self.training_config.use_deepspeed:
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            optimizer = DeepSpeedCPUAdam(parameters, **adam_params, adamw_mode=True)
        else:
            optimizer = AdamW(parameters, **adam_params)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=self.training_config.lr_decay_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _step(self, batch, _: int, mode: str = "train") -> torch.Tensor:
        """Perform a single training/validation/test step with early exit loss support."""
        if mode == "train":
            self.metrics_logger.dump_first_batch(batch)

        # For early exit training:
        if (
            self.config.use_early_exit
            and hasattr(self.model, "early_exit")
            and mode == "train"
        ):
            # Run forward pass with output_hidden_states=True to get all intermediate representations
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                output_hidden_states=True,
                return_dict=True,
            )

            # Get the hidden states from the output
            hidden_states = outputs.hidden_states

            # If this is an encoder-decoder model, use decoder hidden states
            if (
                hasattr(outputs, "decoder_hidden_states")
                and outputs.decoder_hidden_states is not None
            ):
                hidden_states = outputs.decoder_hidden_states

            # Compute early exit loss
            loss, layer_losses = self.model.early_exit.compute_early_exit_loss(
                hidden_states, self.model.get_output_embeddings(), batch["labels"]
            )

            # Log individual layer losses
            for layer_name, layer_loss in layer_losses.items():
                self.log(
                    f"{mode}_{layer_name}",
                    layer_loss,
                    sync_dist=True,
                    batch_size=batch["labels"].shape[0],
                )
        else:
            # Regular forward pass for validation or non-early-exit training
            outputs = self.model(**batch)
            loss = outputs.loss

            # Regular logging from the existing implementation
            self.log(
                f"{mode}_cross_entropy_loss",
                loss,
                sync_dist=True,
                batch_size=batch["labels"].shape[0],
            )

            self.metrics_logger.log_metrics(loss, outputs, batch["labels"], mode)

            if self.config.use_gating:
                gate_entropy_loss, gate_sparsity_loss = (
                    self.model.gating.compute_gate_loss()
                )

                gate_entropy_loss = self.config.entropy_loss_weight * gate_entropy_loss

                self.log(
                    f"{mode}_gate_sparsity_loss",
                    gate_sparsity_loss,
                    sync_dist=True,
                    batch_size=batch["labels"].shape[0],
                )

                gate_sparsity_loss = (
                    self.config.sparsity_loss_weight * gate_sparsity_loss
                )

                loss += gate_entropy_loss
                loss += gate_sparsity_loss

            elif self.config.use_mod:
                predictor_loss = self.model.mod.compute_predictor_loss(dtype=loss.dtype)

                self.log(
                    f"{mode}_predictor_loss",
                    predictor_loss,
                    sync_dist=True,
                    batch_size=batch["labels"].shape[0],
                )

                loss += predictor_loss * self.config.predictor_loss_weight

        if self.config.use_gating:
            percent_skipped = []
            for name, module in self.model.gating.wrapped_modules.items():
                percent_skipped.append(module.current_percent_tokens_skipped)

            self.log(
                f"{mode}_threshold",
                list(self.model.gating.wrapped_modules.items())[0][1].threshold,
                sync_dist=True,
                batch_size=batch["input_ids"].shape[0],
            )
            percent_skipped = torch.tensor(
                percent_skipped, device=batch["input_ids"].device
            ).mean()
            self.log(
                f"{mode}_percent_tokens_skipped",
                percent_skipped,
                sync_dist=True,
                batch_size=batch["input_ids"].shape[0],
            )
        elif self.config.use_mod:
            percent_skipped = []
            for module in self.model.mod.wrapped_modules.values():
                percent_skipped.append(1 - module.current_percent_tokens_skipped)
            percent_skipped = torch.tensor(
                percent_skipped, device=batch["input_ids"].device
            ).mean()
            self.log(
                f"{mode}_percent_tokens_skipped",
                percent_skipped,
                sync_dist=True,
                batch_size=batch["input_ids"].shape[0],
            )
        elif self.config.use_early_exit and hasattr(self.model, "early_exit"):
            # Get early exit statistics
            early_exit_stats = self.model.early_exit.compute_early_exit_statistics()
            if "compute_saved" in early_exit_stats:
                self.percent_tokens_skipped.append(early_exit_stats["compute_saved"])

        self.metrics_logger.log_loss(loss, mode)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="test")

    def tie_weights(self):
        """Tie the model's weights"""
        self.model.tie_weights()
