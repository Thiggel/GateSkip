import math
from lightning import LightningModule
import torch
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class MetricsLogger:
    """Handles metric logging logic"""

    def __init__(
        self,
        lightning_module: LightningModule,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
    ):
        self.module = lightning_module
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def dump_first_batch(self, batch: dict[str, torch.Tensor]) -> None:
        try:
            if self.trainer.global_rank > 0:
                return
        except Exception:
            pass

        MAX_DUMPS = 5

        if not hasattr(self, "num_dumped_first_batch"):
            self.num_dumped_first_batch = 0

        if self.num_dumped_first_batch < MAX_DUMPS and self.tokenizer is not None:
            input_ids = batch["input_ids"]
            for i in range(min(len(input_ids), 3)):
                ids = input_ids[i]

                decoded = self.tokenizer.decode(ids)
                print()
                print("Input sample: ", decoded)
                print()

                if "labels" in batch:
                    labels = batch["labels"][i]
                    filtered_input_ids = ids[labels != -100]
                    decoded = self.tokenizer.decode(filtered_input_ids)
                    print(
                        "Target sample (should not include question and padding): ",
                        decoded,
                    )
                    print()

            self.num_dumped_first_batch += 1

    def log_loss(self, loss: torch.Tensor, mode: str):
        self.module.log(
            f"{mode}_loss",
            loss,
            on_step=(mode == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

    def accuracy(
        self,
        outputs: CausalLMOutputWithCrossAttentions,
        labels: torch.Tensor,
        top_k: int = 5,
    ) -> float:
        top_k_predictions = outputs.logits.topk(top_k).indices
        mask = labels != -100
        correct = (
            ((top_k_predictions == labels.unsqueeze(-1)).any(dim=-1) & mask)
            .sum()
            .item()
        )
        total = mask.sum().item()
        accuracy = correct / total if total > 0 else 0

        return accuracy

    def log_metrics(self, loss: torch.Tensor, outputs, labels: torch.Tensor, mode: str):

        if mode == "train":
            metrics = {}
        else:
            try:
                perplexity = math.exp(loss.item())
                topk = 5
                accuracy = self.accuracy(outputs, labels, top_k=topk)

                metrics = {
                    f"{mode}_perplexity": perplexity,
                    f"{mode}_top{topk}_accuracy": accuracy,
                }
            except Exception as e:
                print(f"Error logging metrics: {e}")
                metrics = {}

        # Log gate metrics if gating is used
        if hasattr(self.module.model, "gating"):
            for name, module in self.module.model.gating.wrapped_modules.items():
                if module.current_gate_value is not None:
                    gate_value = module.current_gate_value

                    # Log mean gate value (sparsity indicator)
                    metrics[f"{mode}_gate_value/{name}"] = gate_value.mean().item()

                    # Log entropy
                    eps = 1e-6
                    entropy = (
                        -(
                            gate_value * (gate_value + eps).log()
                            + (1 - gate_value) * (1 - gate_value + eps).log()
                        )
                        .mean()
                        .item()
                    )
                    metrics[f"{mode}_gate_entropy/{name}"] = entropy

        for name, value in metrics.items():
            self.module.log(
                name,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=self.batch_size,
            )

