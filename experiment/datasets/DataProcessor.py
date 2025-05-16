from typing import Any, Union, Protocol
import torch
from torch.utils.data import Dataset, IterableDataset
from typing_extensions import runtime_checkable


@runtime_checkable
class DataProcessor(Protocol):
    """Protocol for dataset processing operations"""

    def process_split(
        self, dataset: Union[Dataset, IterableDataset], dataset_config: dict[str, Any]
    ) -> Union[Dataset, IterableDataset]: ...

    def tokenize(
        self, samples: dict[str, list[Any]], dataset_config: dict[str, Any]
    ) -> dict[str, list[torch.Tensor]]: ...
