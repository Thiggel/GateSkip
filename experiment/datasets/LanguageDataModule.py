from typing import Any, List, Optional, Tuple, Union
import os
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from lightning import LightningDataModule
from datasets import Dataset, load_dataset, disable_caching, config, load_from_disk
from torch.utils.data import IterableDataset
from functools import partial

from experiment.configs import DataConfig, ModelConfig, TrainingConfig

from .DatasetSplit import DatasetSplit
from .TokenizationProcessor import TokenizationProcessor
from .DatasetManager import DatasetManager
from .BatchCollator import BatchCollator
from .DatasetConfigurator import DatasetConfigurator
from .custom_datasets import (
    CSQAGen,
    GSM8KGen,
    ReasoningDataset,
    HFReasoningDataset,
)


class LanguageDataModule(LightningDataModule):
    """Main data module for language modeling tasks"""

    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        eval_batch_size: int,
        tokenizer: PreTrainedTokenizer,
        seed: int,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.data_config = data_config
        self.model_config = model_config
        self.training_config = training_config
        self.eval_batch_size = eval_batch_size
        self.tokenizer = tokenizer
        self.seed = seed
        self.dataset_manager = DatasetManager(
            cache_dir
            if cache_dir is not None
            else (
                os.environ.get("BASE_CACHE_DIR", ".")
                if torch.cuda.is_available()
                else "."
            )
        )
        self.dataset_config = DatasetConfigurator.get_dataset_config(
            self.data_config.dataset
        )
        self.token_processor = TokenizationProcessor(tokenizer, self.data_config, self.dataset_config)
        self.batch_collator = BatchCollator(tokenizer, data_config.seq_length)
        self.datasets: Optional[DatasetSplit] = None
        self.setup()

    def prepare_data(self) -> None:
        disable_caching()

    def setup(self, stage: Optional[str] = None) -> None:
        try:

            cache_path = self.dataset_manager.get_cache_path(
                self.data_config, self.model_config, self.seed
            )
            if self.dataset_manager.cached_datasets_exist(cache_path):
                self.datasets = self.dataset_manager.load_cached_datasets(cache_path)
            else:
                self.datasets = self._prepare_datasets()
                self.dataset_manager.save_datasets(self.datasets, cache_path)
        except Exception as e:
            print(e)
            raise ValueError(f"Error while setting up datasets: {e}")

    def _prepare_datasets(self) -> DatasetSplit:
        if self.dataset_config.get("streaming", False):
            return self._prepare_streaming_datasets()
        else:
            return self._prepare_static_datasets()

    def _prepare_static_datasets(self) -> DatasetSplit:
        if "dataset_class" in self.dataset_config:
            cache_dir = os.path.join(
                os.environ.get("BASE_CACHE_DIR", "."),
                self.data_config.dataset,
                self.model_config.model_name,
            )

            if not os.path.exists(cache_dir):

                train_dataset = self._get_dataset_instance(
                    self.dataset_config["dataset_class"],
                    **self.dataset_config.get("dataset_params", {}),
                    tokenizer=self.tokenizer,
                    process_fn=lambda x: self.token_processor.process_sample(x),
                ).to_hf_dataset()
                test_dataset = None

                train_dataset.save_to_disk(cache_dir)

            else:
                train_dataset = load_from_disk(cache_dir)
                test_dataset = None

        else:
            ds = load_dataset(
                self.dataset_config["name"],
                self.dataset_config.get("subset"),
                trust_remote_code=True,
            )

            train_dataset = self._process_split(ds[self.dataset_config["train_field"]])

            if "test_field" in self.dataset_config:
                test_dataset = self._process_split(
                    ds[self.dataset_config["test_field"]]
                )
            elif "test_subset" in self.dataset_config:
                train_dataset, test_dataset = self._split_dataset(
                    train_dataset, self.dataset_config["test_subset"]
                )
            else:
                test_dataset = None

        train_dataset, val_dataset = self._split_dataset(
            train_dataset,
            (
                int(len(train_dataset) * 0.1)
                if "val_subset" not in self.dataset_config
                else int(self.dataset_config["val_subset"])
            ),
        )

        return DatasetSplit(train_dataset, val_dataset, test_dataset)

    def _get_dataset_instance(self, dataset_class_name, **kwargs):
        """Factory method to create dataset instances."""
        dataset_classes = {
            "CSQAGen": CSQAGen,
            "GSM8KGen": GSM8KGen,
            "ReasoningDataset": ReasoningDataset,
            "HFReasoningDataset": HFReasoningDataset,
        }
        return dataset_classes[dataset_class_name](**kwargs)

    def _prepare_streaming_datasets(self) -> DatasetSplit:
        ds = load_dataset(
            self.dataset_config["name"],
            self.dataset_config.get("subset"),
            streaming=True,
            trust_remote_code=True,
        )

        def filter_and_process(sample):
            processed = self.token_processor.process_sample(sample)
            return processed is not None

        train_dataset = (
            ds[self.dataset_config["train_field"]]
            .filter(filter_and_process)
            .map(
                partial(
                    self.token_processor.process_sample,
                ),
                remove_columns=ds[self.dataset_config["train_field"]].column_names,
            )
        )

        val_dataset = train_dataset.take(self.dataset_config["val_subset"])

        return DatasetSplit(train_dataset, val_dataset, None)

    def _process_split(
        self, dataset: Union[Dataset, IterableDataset]
    ) -> Union[Dataset, IterableDataset]:
        return dataset.map(
            lambda samples: self._tokenize(samples),
            remove_columns=dataset.column_names,
            batched=True,
        )

    def _tokenize(self, samples: dict[str, List[Any]]) -> dict[str, List[Any]]:
        samples_list = [dict(zip(samples, i)) for i in zip(*samples.values())]

        processed = []
        for sample in samples_list:
            result = self.token_processor.process_sample(sample)
            if result is not None:
                processed.append(result)

        if not processed:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        return {k: [sample[k] for sample in processed] for k in processed[0].keys()}

    def _split_dataset(
        self, dataset: Dataset, split_size: Union[int, float]
    ) -> Tuple[Dataset, Dataset]:
        split = dataset.train_test_split(
            test_size=split_size, shuffle=True, seed=self.seed
        )
        return split["train"], split["test"]

    def _create_validation_set(self, train_dataset: IterableDataset) -> IterableDataset:
        validation_dataset = train_dataset.filter(
            lambda _, idx: idx < self.data_config.val_dataset_size, with_indices=True
        )
        return validation_dataset

    def _create_dataloader(self, dataset, batch_size, is_train=False):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train and not isinstance(dataset, IterableDataset),
            collate_fn=self.batch_collator,
            num_workers=1,
            drop_last=is_train,
            pin_memory=True,
            persistent_workers=True,
        )

    def train_dataloader(self) -> DataLoader:
        if not self.datasets or not self.datasets.train:
            raise ValueError("Training dataset not initialized")

        return self._create_dataloader(
            self.datasets.train, self.data_config.batch_size, is_train=True
        )

    def val_dataloader(self) -> DataLoader:
        if not self.datasets or not self.datasets.validation:
            raise ValueError("Validation dataset not initialized")

        return self._create_dataloader(
            self.datasets.validation, self.eval_batch_size, is_train=False
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if not self.datasets or not self.datasets.test:
            return None

        return self._create_dataloader(
            self.datasets.test, self.eval_batch_size, is_train=False
        )

    def _get_num_workers(self) -> int:
        if self.dataset_config.get("streaming", False):
            return 1

        if hasattr(os, "sched_getaffinity"):
            try:
                max_workers = len(os.sched_getaffinity(0))
            except Exception:
                max_workers = os.cpu_count() or 1
        else:
            max_workers = os.cpu_count() or 1

        return int(max_workers * 0.75)
