from typing import Optional, Callable

from .ReasoningDataset import ReasoningDataset


class GSM8KGen(ReasoningDataset):
    def __init__(
        self,
        root_dir: str = "gsm8k_oai",
        tokenizer=None,
        process_fn: Optional[Callable] = None,
    ):
        super().__init__(
            root_dirs=[root_dir], tokenizer=tokenizer, process_fn=process_fn
        )
