import os
from datasets import Dataset as HFDataset, load_dataset
import json
from typing import Optional, Callable
from torch.utils.data import Dataset
import random
from tqdm import tqdm


class HFReasoningDataset(Dataset):
    def __init__(
        self,
        dataset_paths: list[str] = ["multi-domain-reasoning/commonsense_qa", "openai/gsm8k", ], # "multi-domain-reasoning/mmlu"],
        tokenizer=None,
        process_fn: Optional[Callable] = None,
    ):
        self.process_fn = process_fn

        self.datasets = []

        for path in dataset_paths:
            try:
                dataset = load_dataset(path)
            except:
                dataset = load_dataset(path, 'main')

            if "train" in dataset:
                self.datasets.append(dataset["train"])
            elif "auxiliary_train" in dataset:
                self.datasets.append(dataset["auxiliary_train"])
            else:
                raise ValueError(f"No train or auxiliary_train split found in {path}")

        self.random_indices = list(range(len(self)))
        random.shuffle(self.random_indices)

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def _index_to_dataset(self, index: int):
        for dataset in self.datasets:
            if index < len(dataset):
                return dataset, index
            index -= len(dataset)

    def __getitem__(self, index: int) -> dict[str, str]:
        index = self.random_indices[index]
        dataset, index = self._index_to_dataset(index)

        data = dataset[index]

        query = data["question"]
        choices = data["choices"]["text"] if "choices" in data else None

        if choices is not None:
            letters = ["A", "B", "C", "D", "E"]
            query += "\n" + "\n".join([f"{letters[i]}: {choice}" for i, choice in enumerate(choices)])

            steps = data["reasoning_nemotron_70B"].replace("<reasoning>", "").replace("</reasoning>", "")
            answer = data["answerKey"]

        else:
            steps = data["answer"].split("#### ")[0]
            answer = data["answer"].split("#### ")[1]


        item = {"query": query, "steps": steps, "answer": answer}

        if self.process_fn is not None:
            return self.process_fn(item)

        return item

    def to_hf_dataset(self):
        """Convert to HuggingFace dataset"""
        # Collect all data in memory
        data = {"input_ids": [], "attention_mask": [], "labels": []}
        for i in tqdm(range(len(self)), desc="Converting to HF dataset"):
            item = self[i]
            for key in item:
                if key in data:
                    data[key].append(item[key])

        return HFDataset.from_dict(data)
