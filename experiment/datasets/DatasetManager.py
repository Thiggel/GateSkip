from pathlib import Path
from datasets import load_from_disk, IterableDataset

from experiment.configs import DataConfig, ModelConfig

from .DatasetSplit import DatasetSplit


class DatasetManager:
    """Handles dataset loading and caching operations"""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)

    def get_cache_path(
        self, data_config: DataConfig, model_config: ModelConfig, seed: int
    ) -> Path:
        return (
            self.cache_dir
            / f"{model_config.model_name}_{data_config.dataset}_{data_config.seq_length}_{data_config.batch_size}_{seed}"
        )

    def cached_datasets_exist(self, cache_path: Path) -> bool:
        return all(
            (cache_path / f"{split}").exists() for split in ["train", "valid", "test"]
        )

    def load_cached_datasets(self, cache_path: Path) -> DatasetSplit:
        return DatasetSplit(
            train=load_from_disk(str(cache_path / "train")),
            validation=load_from_disk(str(cache_path / "valid")),
            test=load_from_disk(str(cache_path / "test")),
        )

    def save_datasets(self, datasets: DatasetSplit, cache_path: Path) -> None:
        if isinstance(datasets.train, IterableDataset):
            return

        if datasets.train:
            datasets.train.save_to_disk(str(cache_path / "train"))
        if datasets.validation:
            datasets.validation.save_to_disk(str(cache_path / "valid"))
        if datasets.test:
            datasets.test.save_to_disk(str(cache_path / "test"))
