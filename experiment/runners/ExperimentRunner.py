from typing import Dict
import torch
from pydantic import BaseModel
import numpy as np
import random

from experiment.experiment import Runner, ExperimentConfig

from .TrainRunner import TrainRunner
from .EvaluationRunner import EvaluationRunner


class ExperimentRunner(Runner):
    """Main runner that delegates to specific runners based on mode"""

    def __init__(self, configs: dict[str, BaseModel]):
        self.configs = configs
        self.runner = self._create_runner()

    def _create_runner(self) -> Runner:
        experiment_config: ExperimentConfig = self.configs[ExperimentConfig.__name__]

        if experiment_config.mode == "train":
            return TrainRunner(self.configs)
        elif experiment_config.mode == "evaluate":
            return EvaluationRunner(self.configs)
        else:
            raise ValueError(f"Unknown mode: {experiment_config.mode}")

    def set_seed(self, seed: int):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run(self, seed: int) -> Dict[str, float]:
        self.set_seed(seed)
        return self.runner.run(seed)
