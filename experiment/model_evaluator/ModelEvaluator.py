import sys
from pathlib import Path
import json
from experiment.configs.GatingConfig import GenerationMode
from experiment.models import DefaultLightningModule
from lm_eval.tasks import TaskManager
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import torch
from transformers import PreTrainedTokenizer
import os


class ModelEvaluator:
    """Handles model evaluation using lm-eval-harness with multi-GPU support"""

    def __init__(
        self,
        model: DefaultLightningModule,
        tokenizer: PreTrainedTokenizer,
        eval_batch_size: int = 128,
        num_fewshot: int = 0,
    ):
        self.eval_batch_size = eval_batch_size
        self.num_fewshot = num_fewshot

        self.model = model

        self.tokenizer = tokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)
        self.model.eval()

    def get_gen_kwargs(self, generation_mode: GenerationMode) -> dict[str, any]:
        if generation_mode == GenerationMode.SAMPLING:
            gen_kwargs = {
                "do_sample": True,
                "temperature": 0.2,
                "top_k": 50,
                "top_p": 0.95,
            }

        elif generation_mode == GenerationMode.BEAM:
            gen_kwargs = {
                "num_beams": 3,
                "early_stopping": True,
            }

        else:
            gen_kwargs = {}

        gen_kwargs["max_length"] = 1024
        gen_kwargs["max_new_tokens"] = 1024

        return gen_kwargs

    def dict_to_str(self, d: dict[str, any]) -> str:
        """cononverts dict to e.g. args1=val1,arg2=val2"""
        return ",".join([f"{k}={v}" for k, v in d.items()])

    def evaluate(
        self,
        metrics: list[str],
        seed: int,
        experiment_name: str,
        generation_mode: GenerationMode,
        limit: int = 10000,
    ) -> dict[str, float]:
        gen_kwargs = self.get_gen_kwargs(generation_mode)

        ### TEST
        # from .CustomEvaluator import CustomEvaluator

        # evaluator = CustomEvaluator(
        #    model=self.model,
        #    tokenizer=self.tokenizer,
        #    batch_size=self.eval_batch_size,
        # )

        # results = []
        # for metric in metrics:
        #    metric_results = evaluator.evaluate(
        #        dataset_name=metric,
        #        split="test",
        #        generation_kwargs=gen_kwargs,
        #    )
        #    results.append(metric_results)

        # print(results)
        # return results
        ### /TEST

        wrapped_model = HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
            batch_size=self.eval_batch_size,
            backend="causal",
            device=self.device,
            add_bos_token=True,
        )

        gen_kwargs_str = self.dict_to_str(gen_kwargs) if gen_kwargs != {} else None

        print("gen_kwargs_str:", gen_kwargs_str)

        tm = TaskManager(include_path=os.path.join(sys.path[0], "lm_eval", "tasks"))

        output = evaluator.simple_evaluate(
            model=wrapped_model,
            tasks=metrics or ["commonsense_qa", "gsm8k", "piqa"],
            num_fewshot=self.num_fewshot,
            batch_size=self.eval_batch_size,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
            device=self.device,
            log_samples=True,
            gen_kwargs=gen_kwargs_str,
            task_manager=tm,
            limit=limit,
        )

        # samples = {}
        # for task, task_output in output["samples"].items():
        #     samples[task] = output["samples"][task][0]

        #output["results"]["samples"] = samples

        self._save_results(output["results"], experiment_name)
        self._save_samples(output["samples"], seed, experiment_name)

        return output["results"]

    def _save_results(self, results: dict, experiment_name: str):
        output_dir = Path(os.getenv("BASE_CACHE_DIR") or "") / "evaluation_results"
        output_dir.mkdir(exist_ok=True)

        results_path = output_dir / f"{experiment_name}.json"
        with results_path.open("w") as f:
            json.dump(results, f, indent=2)

    def _save_samples(self, samples: dict, seed, experiment_name):
        try:
            sample_dir = Path(os.environ["BASE_CACHE_DIR"] or "") / "samples"
            sample_dir.mkdir(exist_ok=True)

            sample_path = sample_dir / f"{experiment_name}_{seed}.json"
            sample_path.write_text(json.dumps(samples))
        except Exception as e:
            print(f"Failed to save samples: {e}")
