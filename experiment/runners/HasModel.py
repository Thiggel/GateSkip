from typing import Protocol
import torch
import os
from transformers import PreTrainedTokenizer

from experiment.configs import DataConfig, EvaluationConfig, ModelConfig, TrainingConfig
from experiment.models import DefaultLightningModule


class HasModelProtocol(Protocol):
    model_config: ModelConfig
    training_config: TrainingConfig
    data_config: DataConfig
    evaluation_config: EvaluationConfig
    tokenizer: PreTrainedTokenizer


class HasModel:
    def _load_model(
        self: HasModelProtocol, seed: int, mode: str = "train"
    ) -> DefaultLightningModule:
        model = DefaultLightningModule(
            self.model_config,
            self.training_config,
            self.data_config,
            self.evaluation_config,
            self.tokenizer,
            seed,
        )
        model.setup(mode)

        print(model)

        if self.evaluation_config.load_from_checkpoint:
            checkpoint_path = os.path.join(
                os.environ["BASE_CACHE_DIR"],
                f"{self.evaluation_config.load_from_checkpoint}_{seed}.pt",
            )
            print("Loading from checkpoint", checkpoint_path)

            checkpoint = torch.load(checkpoint_path, weights_only=False)

            state_dict = (
                checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            )
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)

            return model

        return model
