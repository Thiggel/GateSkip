from pydantic import BaseModel, Field
from enum import Enum


class ExperimentMode(str, Enum):
    TRAIN = "train"
    EVALUATE = "evaluate"


class ExperimentConfig(BaseModel):
    experiment_name: str = Field("default", description="Name of the experiment")
    seeds: list[int] = Field([1], description="Random seeds to use")
    num_runs: int = Field(1, description="Number of runs to perform")
    enable_logging: bool = Field(False, description="Enable wandb logging")
    continue_on_error: bool = Field(False, description="Continue if a run fails")
    save_results: bool = Field(True, description="Save results to disk")
    mode: ExperimentMode = Field("train", description="Experiment mode")
    project_name: str = Field("letting-nns-think", description="Wandb project name")
