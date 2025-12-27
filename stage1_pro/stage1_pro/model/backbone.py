import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional, List


class DINOv3Backbone(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        freeze: bool = True,
        use_peft: bool = False,
        peft_type: str = "dora",  # Phase 3: "dora" or "lora"
        peft_rank: int = 16,
        peft_blocks: int = 6,
    ):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.embed_dim = self.model.config.hidden_size

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.use_peft = use_peft
        self.peft_type = peft_type
        self.peft_rank = peft_rank
        self.peft_blocks = peft_blocks
        self.peft_adapters = {}

        # Register PEFT adapters if enabled
        if use_peft:
            self._register_peft_adapters()

    def _register_peft_adapters(self):
        """Register PEFT adapters on specified backbone layers."""
        from .peft import DoRAAdapter, LoRAAdapter

        # Get transformer encoder layers
        layer_count = 0
        for name, module in self.model.named_modules():
            # Target transformer encoder blocks
            if "encoder.layer" in name and layer_count < self.peft_blocks:
                # Find the output projection layer
                if name.endswith("output"):
                    in_features = module.in_features
                    out_features = module.out_features

                    # Create adapter
                    if self.peft_type == "dora":
                        adapter = DoRAAdapter(
                            in_features, out_features, rank=self.peft_rank
                        )
                    else:
                        adapter = LoRAAdapter(
                            in_features, out_features, rank=self.peft_rank
                        )

                    # Register adapter on the module
                    def create_hook(adapter_module):
                        def hook(mod, inp, out):
                            return adapter_module(out)

                        return hook

                    module.register_forward_hook(create_hook(adapter))
                    self.peft_adapters[f"layer_{layer_count}"] = adapter

                    layer_count += 1
                    if layer_count >= self.peft_blocks:
                        break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        return outputs.last_hidden_state[:, 0, :]

    def register_peft_hook(self, layer_name: str, adapter: nn.Module):
        """Register a PEFT adapter on a specific layer (legacy)."""
        if layer_name in self.peft_adapters:
            return

        def hook(module, input, output):
            return adapter(output)

        for name, module in self.model.named_modules():
            if layer_name in name:
                module.register_forward_hook(hook)
                self.peft_adapters[layer_name] = adapter
                break

    def get_peft_parameters(self):
        """Get only PEFT parameters for training."""
        if not self.use_peft:
            return []
        params = []
        for adapter in self.peft_adapters.values():
            params.extend(adapter.parameters())
        return params

    def get_backbone_parameters(self):
        """Get backbone parameters (excluding PEFT)."""
        return self.model.parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.use_peft:
            for name, param in self.model.named_parameters():
                if all(l not in name for l in self.peft_adapters.keys()):
                    param.requires_grad = False
                else:
                    param.requires_grad = mode
        return self
