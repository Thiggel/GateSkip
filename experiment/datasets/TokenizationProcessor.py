from typing import Any, Optional
import torch
from transformers import PreTrainedTokenizer
from experiment.configs import DataConfig


class TokenizationProcessor:
    """Handles tokenization-related operations"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_config: DataConfig,
        dataset_config: dict[str, Any],
    ):
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.dataset_config = dataset_config

    def tokenize_text(
        self, text: list[str], max_length: Optional[int] = None
    ) -> dict[str, list[Any]]:
        return self.tokenizer(
            text,
            padding="do_not_pad",
            truncation=max_length is not None,
            max_length=max_length,
        )

    def process_sample(
        self,
        sample: str,
    ) -> Optional[dict[str, Any]]:
        q_func = self.dataset_config["q_func"]
        ans_func = self.dataset_config["ans_func"]

        query_text = q_func(sample)
        answer_text = ans_func(sample) + (
            self.tokenizer.eos_token
            if self.dataset_config.get("add_eos_token", False)
            else ""
        )

        query_tokens = self.tokenizer(query_text, add_special_tokens=False)
        answer_tokens = self.tokenizer(answer_text, add_special_tokens=False)
        query_length = len(query_tokens["input_ids"])
        total_len = query_length + len(answer_tokens["input_ids"])

        if (
            self.dataset_config.get("filter_samples_above_max_len", False)
            and total_len > self.data_config.seq_length
        ) or (
            self.dataset_config.get("filter_samples_below_max_len", False)
            and total_len < self.data_config.seq_length
        ):
            return None

        input_ids = query_tokens["input_ids"] + answer_tokens["input_ids"]
        attention_mask = (
            query_tokens["attention_mask"] + answer_tokens["attention_mask"]
        )
        labels = input_ids.copy()
        labels[:query_length] = [-100] * query_length

        if len(input_ids) > self.data_config.seq_length:
            input_ids = input_ids[: self.data_config.seq_length]
            attention_mask = attention_mask[: self.data_config.seq_length]
            labels = labels[: self.data_config.seq_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
