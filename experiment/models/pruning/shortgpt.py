import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import PreTrainedModel
from typing import Iterable, Sequence

from ..HasLayers import HasLayers


def compute_block_influence(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device,
) -> list[float]:
    """Compute Block Influence (BI) for each decoder layer.

    Args:
        model: The model whose layers will be analysed.
        dataloader: Yields batches of token IDs.
        device: Device to run the computation on.

    Returns:
        List of BI values for each layer (lower means less influential).
    """
    model.eval()
    num_layers = model.config.num_hidden_layers
    bi_scores = torch.zeros(num_layers, dtype=torch.float64)
    token_counts = torch.zeros(num_layers, dtype=torch.long)

    with torch.no_grad():
        for (input_ids,) in dataloader:
            input_ids = input_ids.to(device)
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # tuple of length num_layers + 1
            for i in range(num_layers):
                h_in = hidden_states[i]
                h_out = hidden_states[i + 1]
                cos = F.cosine_similarity(h_in, h_out, dim=-1)
                diff = 1 - cos
                bi_scores[i] += diff.sum().double()
                token_counts[i] += diff.numel()

    bi_scores /= token_counts
    return bi_scores.tolist()


def prune_layers(
    model: PreTrainedModel,
    layers_to_remove: Sequence[int],
) -> PreTrainedModel:
    """Remove the specified layers from the model in place."""
    helper = HasLayers()
    layers = helper.get_decoder_layers(model)
    keep_layers = [
        layer for idx, layer in enumerate(layers) if idx not in layers_to_remove
    ]
    helper.set_decoder_layers(model, torch.nn.ModuleList(keep_layers))
    model.config.num_hidden_layers = len(keep_layers)
    return model

