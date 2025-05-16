from dataclasses import dataclass
from typing import Optional, Union
from torch.utils.data import Dataset, IterableDataset


@dataclass
class DatasetSplit:
    """Container for dataset splits"""

    train: Optional[Union[Dataset, IterableDataset]]
    validation: Optional[Union[Dataset, IterableDataset]]
    test: Optional[Union[Dataset, IterableDataset]]
