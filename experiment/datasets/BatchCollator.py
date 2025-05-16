import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer
from typing import Any


class BatchCollator:
    """Handles batch collation operations with character to token mask conversion"""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = self._pad_and_truncate(
            [torch.tensor(item["input_ids"]) for item in batch],
            self.tokenizer.pad_token_id or 0,
        )
        attention_mask = self._pad_and_truncate(
            [torch.tensor(item["attention_mask"]) for item in batch], 0
        )
        labels = self._pad_and_truncate(
            [torch.tensor(item["labels"]) for item in batch], -100
        )

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        return result

    def _char_to_token_mask(self, text: str, char_mask: list[int]) -> list[int]:
        """Convert character-level mask to token-level mask.

        A token's mask value is 1 if any character within it has mask value 1.
        This ensures we don't lose any masked positions during tokenization.
        """
        # Get the token to chars mapping
        encoding = self.tokenizer(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        token_mask = []

        # For each token, check if any character in its span has mask value 1
        for start, end in encoding.offset_mapping:
            # If the token is a special token (offset_mapping contains None)
            if start is None or end is None:
                token_mask.append(0)
                continue

            # Check if any character in the token's span has mask value 1
            span_mask = char_mask[start:end]
            token_mask.append(1 if any(m == 1 for m in span_mask) else 0)

        return token_mask

    def _pad_and_truncate(
        self, sequences: list[torch.Tensor], padding_value: int
    ) -> torch.Tensor:
        padded = pad_sequence(sequences, batch_first=True, padding_value=padding_value)
        truncated = padded[:, : self.max_length]

        return truncated
