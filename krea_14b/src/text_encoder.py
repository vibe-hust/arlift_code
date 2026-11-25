import os
from typing import List
import torch
from .wan.modules.t5 import umt5_xxl
from .wan.modules.tokenizers import HuggingfaceTokenizer


class WanTextEncoder(torch.nn.Module):
    def __init__(self, weights) -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.bfloat16,
            device=torch.device('meta')
        ).eval().requires_grad_(False)
        # The official Wan 1.3B repository publishes this shared T5 as .pth.
        self.text_encoder.load_state_dict(torch.load(
            os.path.join(weights, "Wan2.1-T2V-1.3B", "models_t5_umt5-xxl-enc-bf16.pth"),
            map_location="cpu", weights_only=True), assign=True)
        self.text_encoder.to(device=torch.cuda.current_device(), dtype=torch.bfloat16)

        self.tokenizer = HuggingfaceTokenizer(
            name=os.path.join(weights, "Wan2.1-T2V-1.3B", "google", "umt5-xxl/"), seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0
        result = { "prompt_embeds": context }
        return result
