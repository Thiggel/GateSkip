import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from experiment.cli_manager import CLIManager
from experiment.configs import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    EvaluationConfig,
)
from experiment.models.model_adapter.ModelAdapter import ModelAdapter
from experiment.models.pruning.shortgpt import compute_block_influence, prune_layers


def load_texts(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]
    return texts


def build_dataloader(texts: list[str], tokenizer, seq_len: int, batch_size: int) -> DataLoader:
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seq_len,
    )
    dataset = TensorDataset(enc["input_ids"])
    return DataLoader(dataset, batch_size=batch_size)


def main():
    parser = argparse.ArgumentParser(description="Apply ShortGPT layer pruning")
    parser.add_argument("model", help="Base model name or path")
    parser.add_argument("dataset", help="Path to text file with one sample per line")
    parser.add_argument("output", help="Directory to save the pruned model")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prune_ratio", type=float, default=0.25)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional path to a .pt checkpoint with model weights",
    )
    args, remaining = parser.parse_known_args()

    # Parse additional model hyperparameters using the same CLI as the main experiment
    sys.argv = [sys.argv[0]] + remaining
    cli = CLIManager(ModelConfig, TrainingConfig, DataConfig, EvaluationConfig)
    cli.parse_args()
    configs = cli.get_all_configs()
    model_config = configs[ModelConfig.__name__]
    training_config = configs[TrainingConfig.__name__]
    data_config = configs[DataConfig.__name__]
    evaluation_config = configs[EvaluationConfig.__name__]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    adapter = ModelAdapter(
        model_config,
        evaluation_config,
        training_config,
        tokenizer,
        device,
        seed=42,
    )
    model = adapter.model

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        new_state_dict = {}
        for key, val in state_dict.items():
            new_key = key.replace("model.model", "model").replace(
                "model.lm_head", "lm_head"
            )
            new_state_dict[new_key] = val
        state_dict = new_state_dict
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    model = model.to(device)

    texts = load_texts(Path(args.dataset))
    dataloader = build_dataloader(texts, tokenizer, args.seq_len, args.batch_size)

    bi_scores = compute_block_influence(model, dataloader, device)
    num_layers = len(bi_scores)
    k = int(num_layers * args.prune_ratio)
    ranked = sorted(range(num_layers), key=lambda i: bi_scores[i])
    to_remove = ranked[:k]

    prune_layers(model, to_remove)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    bi_path = Path(args.output) / "bi_scores.txt"
    with bi_path.open("w") as f:
        for idx, score in enumerate(bi_scores):
            f.write(f"{idx}\t{score}\n")

    remove_path = Path(args.output) / "removed_layers.txt"
    with remove_path.open("w") as f:
        for idx in to_remove:
            f.write(f"{idx}\n")


if __name__ == "__main__":
    main()
