from experiment.experiment import Experiment
from experiment.configs import TrainingConfig, DataConfig, ModelConfig, EvaluationConfig
from experiment.runners import ExperimentRunner


if __name__ == "__main__":
    experiment = Experiment(
        config_classes=[ModelConfig, TrainingConfig, DataConfig, EvaluationConfig],
        runner_class=ExperimentRunner,
    )

    experiment.run()
