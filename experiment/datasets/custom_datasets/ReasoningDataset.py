import os
from datasets import Dataset as HFDataset
import json
from typing import Optional, Callable
from torch.utils.data import Dataset
import random


class ReasoningDataset(Dataset):
    def __init__(
        self,
        root_dirs: list[str] = ["csqa_gen", "gsm8k_oai"],
        tokenizer=None,
        process_fn: Optional[Callable] = None,
    ):
        self.root_dirs = root_dirs
        self.process_fn = process_fn

        self.files = [
            os.path.join(root_dir, file)
            for root_dir in self.root_dirs
            for file in os.listdir(root_dir)
            if file.endswith(".json")
        ]
        random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, str]:
        with open(self.files[index], "r") as f:
            data = json.load(f)

        query = data["query"]
        steps = "\n".join(data["steps"])
        solution = data["solution"]

        if self.process_fn is not None:
            return self.process_fn(
                {"query": query, "steps": steps, "solution": solution}
            )

        return {"query": query, "steps": steps, "solution": solution}

    def to_hf_dataset(self):
        """Convert to HuggingFace dataset"""
        # Collect all data in memory
        data = {"input_ids": [], "attention_mask": [], "labels": []}
        for i in range(len(self)):
            item = self[i]
            for key in item:
                if key in data:
                    data[key].append(item[key])

        return HFDataset.from_dict(data)
