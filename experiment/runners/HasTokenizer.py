from transformers import PreTrainedTokenizer, AutoTokenizer
from pydantic import BaseModel

from experiment.configs import ModelConfig, DataConfig


class HasTokenizer:
    configs: dict[str, BaseModel]

    def _initialize_tokenizer(self) -> PreTrainedTokenizer:
        model_config: ModelConfig = self.configs[ModelConfig.__name__]
        data_config: DataConfig = self.configs[DataConfig.__name__]

        print(f"Using pretrained tokenizer from {model_config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
