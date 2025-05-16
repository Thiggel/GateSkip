from typing import Optional, Callable

from .ReasoningDataset import ReasoningDataset


class CSQAGen(ReasoningDataset):
    def __init__(
        self,
        root_dir: str = "csqa_gen",
        tokenizer=None,
        process_fn: Optional[Callable] = None,
    ):
        super().__init__(
            root_dirs=[root_dir], tokenizer=tokenizer, process_fn=process_fn
        )
