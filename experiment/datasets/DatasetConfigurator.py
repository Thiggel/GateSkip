from typing import Any


class DatasetConfigurator:
    """Manages dataset configurations"""

    @staticmethod
    def get_dataset_config(dataset_name: str) -> dict[str, Any]:
        configs = DatasetConfigurator.get_all_dataset_configs()
        if dataset_name not in configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return configs[dataset_name]

    @staticmethod
    def get_all_dataset_configs() -> dict[str, dict[str, Any]]:
        base_configs = {
            "wmt16": {
                "name": "wmt/wmt16",
                "subset": "ro-en",
                "q_func": lambda x: f"translate English to Romanian: {x['translation']['en']}\n",
                "ans_func": lambda x: x["translation"]["ro"],
                "train_field": "train",
                "test_field": "test",
                "add_eos_token": True,
                "filter_samples_above_max_len": True,
            },
            "ultrafeedback": {
                "name": "openbmb/UltraFeedback",
                "q_func": lambda x: f"Question: {x['instruction']}\n\nAnswer:",
                "ans_func": lambda x: " " + x["completions"][0]["response"],
                "train_field": "train",
                "test_subset": 1000,
                "custom_filter": lambda ds: ds.filter(
                    lambda x: len(
                        [c for c in x["completions"] if c["model"] == "gpt-4"]
                    )
                    > 0
                ),
                "filter_samples_above_max_len": True,
            },
            "fineweb": {
                "name": "HuggingFaceFW/fineweb",
                "subset": "sample-10BT",
                "q_func": lambda x: "",
                "ans_func": lambda x: x["text"],
                "train_field": "train",
                "streaming": True,
                "val_subset": 1000,
                "filter_samples_below_max_len": True,
            },
            "cot_collection": {
                "name": "kaist-ai/CoT-Collection",
                "q_func": lambda x: f"Question: {x['source']}\nAnswer:",
                "ans_func": lambda x: f" {x['rationale']}\n#### {x['target']}",
                "train_field": "train",
                "streaming": True,
                "val_subset": 500,
                "filter_samples_above_max_len": True,
                "add_eos_token": True,
            },
            "gsm8k": {
                "name": "gsm8k",
                "q_func": lambda x: f"Question: {x['question']}\nAnswer:",
                "ans_func": lambda x: " " + x["answer"],
                "subset": "main",
                "train_field": "train",
                "test_field": "test",
                "add_eos_token": True,
                "filter_samples_above_max_len": True,
            },
            "csqa_gen": {
                "dataset_class": "CSQAGen",
                "q_func": lambda x: f"Question: {x['query']}\nAnswer:",
                "ans_func": lambda x: f" {x['steps']}\n#### {x['solution']}",
                "add_eos_token": True,
                "filter_samples_above_max_len": True,
            },
            "gsm8k_gen": {
                "dataset_class": "GSM8KGen",
                "q_func": lambda x: f"Question: {x['query']}\nAnswer:",
                "ans_func": lambda x: f" {x['steps']}\n#### {x['solution']}",
                "add_eos_token": True,
                "filter_samples_above_max_len": True,
            },
            #"csqa_gsm8k_gen": {
            #    "dataset_class": "ReasoningDataset",
            #    "q_func": lambda x: f"Question: {x['query']}\nAnswer:",
            #    "ans_func": lambda x: f" {x['steps']}\n#### {x['solution']}",
            #    "add_eos_token": True,
            #    "filter_samples_above_max_len": True,
            #},
            "csqa_gsm8k_gen": {
                "dataset_class": "HFReasoningDataset",
                "q_func": lambda x: f"Question: {x['query']}\nAnswer:",
                "ans_func": lambda x: f" {x['steps']}\n#### {x['answer']}",
                "add_eos_token": True,
                "filter_samples_above_max_len": True,
            },
        }

        return base_configs
