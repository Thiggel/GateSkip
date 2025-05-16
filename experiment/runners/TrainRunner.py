from typing import Any
from pathlib import Path
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
import os
from pydantic import BaseModel

from experiment.experiment import Runner
from experiment.datasets import LanguageDataModule
from experiment.experiment import ExperimentConfig
from experiment.configs import ModelConfig, DataConfig, TrainingConfig, EvaluationConfig

from .HasTokenizer import HasTokenizer
from .HasModel import HasModel


class TrainRunner(Runner, HasTokenizer, HasModel):
    """Handles model training using PyTorch Lightning"""

    def __init__(self, configs: dict[str, BaseModel]):
        super().__init__(configs)

        self.tokenizer = self._initialize_tokenizer()

        self.experiment_config: ExperimentConfig = self.configs[
            ExperimentConfig.__name__
        ]
        self.model_config: ModelConfig = self.configs[ModelConfig.__name__]
        self.data_config: DataConfig = self.configs[DataConfig.__name__]
        self.training_config: TrainingConfig = self.configs[TrainingConfig.__name__]
        self.evaluation_config: EvaluationConfig = self.configs[
            EvaluationConfig.__name__
        ]

    def run(self, seed: int) -> dict[str, float]:
        data_module = LanguageDataModule(
            self.data_config,
            self.model_config,
            self.training_config,
            self.evaluation_config.eval_batch_size,
            self.tokenizer,
            seed,
        )

        model = self._load_model(seed, mode="train")

        trainer = self._setup_trainer(seed)

        print(model)

        trainer.fit(model=model, datamodule=data_module)

        if self.evaluation_config.save_to_checkpoint:
            self._save_checkpoint(self.get_checkpoint_path(), seed)

        return {}

    def get_checkpoint_path(self):
        best_checkpoint_path = self.epoch_checkpoint.best_model_path
        last_step_checkpoint = self.step_checkpoint.last_model_path

        if best_checkpoint_path and os.path.exists(best_checkpoint_path):
            return best_checkpoint_path
        elif last_step_checkpoint and os.path.exists(last_step_checkpoint):
            return last_step_checkpoint

        return ""

    def _setup_trainer(self, seed: int) -> Trainer:
        checkpoint_dir = (
            Path(os.environ.get("PYTORCH_LIGHTNING_HOME"))
            / self.experiment_config.experiment_name
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.step_checkpoint = ModelCheckpoint(
            monitor=None,
            every_n_train_steps=self.training_config.validate_every_n_steps,
            dirpath=checkpoint_dir,
            filename=self.experiment_config.experiment_name
            + "_"
            + str(seed)
            + "_step-checkpoint-{step:06d}",
            save_top_k=1,
            verbose=True,
        )

        self.epoch_checkpoint = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            verbose=True,
            save_on_train_epoch_end=True,
            dirpath=checkpoint_dir,
            filename=self.experiment_config.experiment_name
            + "_"
            + str(seed)
            + "_epoch-checkpoint-{epoch:02d}-{val_loss:.2f}",
        )

        callbacks = [
            self.step_checkpoint,
            self.epoch_checkpoint,
            # DeviceStatsMonitor(),
            LearningRateMonitor(logging_interval="step"),
        ]

        if self.training_config.use_early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.training_config.early_stopping_patience,
                    mode="min",
                    min_delta=0.00,
                    verbose=True,
                )
            )

        trainer_args = self._get_trainer_args(callbacks, seed)

        if torch.cuda.is_available() and self.training_config.use_deepspeed:
            trainer_args.update(self._get_cuda_specific_args())
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        return Trainer(**trainer_args)

    def _get_trainer_args(self, callbacks: list, seed: int) -> dict[str, Any]:
        wandb_logger = None
        if self.experiment_config.enable_logging:
            wandb_logger = WandbLogger(
                project=self.experiment_config.project_name,
                name=f"{self.experiment_config.experiment_name}_{seed}",
                group=self.experiment_config.experiment_name,
                save_dir=os.environ["WANDB_DIR"],
            )

        trainer_args = {
            "check_val_every_n_epoch": 1,
            "callbacks": callbacks,
            "enable_checkpointing": True,
            "logger": wandb_logger if self.experiment_config.enable_logging else None,
            "log_every_n_steps": 10,
            "max_epochs": self.training_config.max_epochs,
            "gradient_clip_val": self.training_config.max_grad_norm,
            "accumulate_grad_batches": self.data_config.grad_acc_steps,
            "devices": "auto",
        }

        if self.training_config.max_training_steps is not None:
            trainer_args["max_steps"] = self.training_config.max_training_steps

        if (
            self.training_config.max_epochs is not None
            and self.training_config.max_epochs > 1 and "max_steps" in trainer_args
        ):
            trainer_args.pop("max_steps")

        if self.training_config.max_hours:
            trainer_args["max_time"] = {
                "hours": self.training_config.max_hours,
            }

        if self.training_config.use_profiler:
            trainer_args["profiler"] = "simple"

        if self.training_config.validate_every_n_steps is not None:
            trainer_args["val_check_interval"] = (
                self.training_config.validate_every_n_steps
            )
            trainer_args.pop("max_epochs")

        return trainer_args

    def _get_cuda_specific_args(self) -> dict[str, Any]:
        strategy = DeepSpeedStrategy(
            config={
                "batch_size": self.data_config.batch_size
                * self.data_config.grad_acc_steps
                * torch.cuda.device_count(),
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {"device": "cpu", "pin_memory": True},
                    "overlap_comm": True,
                    #"allgather_bucket_size": 5e8,
                    #"reduce_bucket_size": 5e8,
                    "contiguous_gradients": True,
                },
                "gradient_clipping": self.training_config.max_grad_norm,
                "bf16": {"enabled": False},
                "wall_clock_breakdown": False,
                "zero_allow_untested_optimizer": True,
                "fp16": {
                    "enabled": False
                },
                "flops_profiler": {
                    "enabled": True,
                    "profile_step": 1,
                    "module_depth": -1,
                    "top_modules": 3,
                    "detailed": True,
                },
                # Add specific torch extension configs
                "sparse_attention": {
                    "mode": "dense",  # Use dense attention
                    "block": 32,      # Block size for sparse attention
                    "different_layout_per_head": True
                },
                # Flash attention for newer GPUs
                "flash_attention": True,
                # Tensor cores specific optimizations
                "tensor_core": {"enabled": True, "ds_inference": True},
                "wall_clock_breakdown": False,
                "zero_allow_untested_optimizer": True,
            }
        )
        args = {
            "accelerator": "gpu",
            "default_root_dir": os.environ["PYTORCH_LIGHTNING_HOME"] + "/../",
        }

        args["strategy"] = strategy
        args["default_root_dir"] = os.environ["PYTORCH_LIGHTNING_HOME"]

        return args


    def _save_checkpoint(self, checkpoint_path: str, seed):
        output_path = os.path.join(
            os.environ["BASE_CACHE_DIR"],
            f"{self.evaluation_config.save_to_checkpoint}_{seed}.pt",
        )

        print(f"\n\nSaving checkpoint to {output_path}\n\n")

        if self.training_config.use_deepspeed:
            convert_zero_checkpoint_to_fp32_state_dict(checkpoint_path, output_path)

        else:
            saved_dict = torch.load(checkpoint_path)["state_dict"]

            torch.save(
                saved_dict,
                output_path,
            )
