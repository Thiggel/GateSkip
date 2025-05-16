from abc import ABC, abstractmethod
from pydantic import BaseModel


class Runner(ABC):
    def __init__(self, configs: dict[str, BaseModel]):
        self.configs = configs

    @abstractmethod
    def run(self, seed: int) -> dict[str, float]:
        pass
