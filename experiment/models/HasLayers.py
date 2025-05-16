from typing import Annotated, Optional
from transformers import AutoModel
from torch import nn

from experiment.configs import LayerRange


class HasLayers:
    def get_decoder_layers(self, model: AutoModel) -> nn.ModuleList:
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            model = model.base_model.model

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # Gemma2 and similar architectures
            return model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT and similar architectures
            return model.transformer.h
        elif hasattr(model, "decoder") and hasattr(model.decoder, "block"):
            # Models like T5 (encoder-decoder, but using decoder only)
            return model.decoder.block
        elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            return model.gpt_neox.layers
        else:
            raise ValueError(
                "Unable to locate decoder layers. Inspect the model structure."
            )

    def set_decoder_layers(self, model: AutoModel, layers: nn.ModuleList):
        old_model = None

        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            old_model = model
            model = model.base_model.model

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # Gemma2 and similar architectures
            model.model.layers = layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT and similar architectures
            model.transformer.h = layers
        elif hasattr(model, "decoder") and hasattr(model.decoder, "block"):
            # Models like T5 (encoder-decoder, but using decoder only)
            model.decoder.block = layers
        elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            model.gpt_neox.layers = layers
        else:
            raise ValueError(
                "Unable to locate decoder layers. Inspect the model structure."
            )

        if (
            old_model is not None
            and hasattr(old_model, "base_model")
            and hasattr(old_model.base_model, "model")
        ):
            old_model.base_model.model = model
            return old_model

        return model

    def _get_layer_range(
        self, model: AutoModel, layer_range: Optional[Annotated[str, LayerRange]]
    ) -> list[tuple[int, int]]:
        if layer_range == "all":
            return [
                (idx, idx + 1) for idx in range(len(self.get_decoder_layers(model)))
            ]

        if layer_range is None:
            return []

        if ":" in layer_range:
            start, end = map(int, layer_range.split(":"))
        else:
            start = int(layer_range)
            end = start + 1

        layers = self.get_decoder_layers(model)

        if start < 0:
            start = len(layers) + start
        if end <= 0:
            end = len(layers) + end

        return [(start, end)]

    def _get_all_layers(
        self, model: AutoModel, layer_range: Optional[Annotated[str, LayerRange]]
    ) -> list[int]:
        return [
            layer
            for (start, end) in self._get_layer_range(model, layer_range)
            for layer in range(start, end)
        ]
