import random

import torch
from transformers import GPT2Config, GPT2LMHeadModel


def tiny_gpt2_config() -> GPT2Config:
    return GPT2Config(
        n_layer=2,
        n_head=4,
        n_embd=64,
        n_positions=64,
        n_ctx=64,
        vocab_size=128,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )


def make_tiny_gpt2_pair(
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> tuple[GPT2LMHeadModel, GPT2LMHeadModel]:
    random.seed(seed)
    torch.manual_seed(seed)
    cfg = tiny_gpt2_config()
    model_a = GPT2LMHeadModel(cfg)

    torch.manual_seed(seed + 1)
    model_b = GPT2LMHeadModel(cfg)

    model_a.to(device=device, dtype=dtype)
    model_b.to(device=device, dtype=dtype)
    model_a.train()
    model_b.train()
    model_a.config.use_cache = False
    model_b.config.use_cache = False
    return model_a, model_b

