from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


ThinkingStateMode = Literal["predictor", "token"]


@dataclass
class BackendResolution:
    requested_backend: str
    actual_backend: str
    import_path: str | None
    implementation_file: str | None
    fallback_used: bool
    fallback_reason: str | None


@dataclass(frozen=True)
class ThinkingStateOutput:
    hidden: torch.Tensor
    thinking_mask: torch.Tensor
    selected_mask: torch.Tensor
    thinking_positions: torch.Tensor
    selected_positions: torch.Tensor
    semantics: str


_BACKEND_RESOLUTIONS: dict[str, BackendResolution] = {}


def _ensure_torchtune_path() -> None:
    root = Path(__file__).resolve().parents[2]
    torchtune_src = root / "torchtune_pkg" / "torchtune"
    if torchtune_src.exists() and str(torchtune_src) not in sys.path:
        sys.path.insert(0, str(torchtune_src))


def _record_resolution(key: str, resolution: BackendResolution) -> None:
    _BACKEND_RESOLUTIONS[key] = resolution


def backend_resolution_snapshot() -> dict:
    return {key: asdict(value) for key, value in sorted(_BACKEND_RESOLUTIONS.items())}


def write_backend_resolution(path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(backend_resolution_snapshot(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_ce_loss_backend():
    requested = "torchtune.modules.loss.CEWithChunkedOutputLoss"
    package_error: Exception | None = None
    try:
        _ensure_torchtune_path()
        module = importlib.import_module("torchtune.modules.loss")
        cls = getattr(module, "CEWithChunkedOutputLoss")
        _record_resolution(
            "ce_loss",
            BackendResolution(
                requested_backend=requested,
                actual_backend="torchtune.modules.loss.CEWithChunkedOutputLoss",
                import_path="torchtune.modules.loss.CEWithChunkedOutputLoss",
                implementation_file=inspect.getsourcefile(cls),
                fallback_used=False,
                fallback_reason=None,
            ),
        )
        return cls
    except Exception as exc:
        package_error = exc

    try:
        root = Path(__file__).resolve().parents[2]
        source_file = root / "torchtune_pkg" / "torchtune" / "torchtune" / "modules" / "loss" / "ce_chunked_output_loss.py"
        if not source_file.exists():
            raise FileNotFoundError(source_file)
        spec = importlib.util.spec_from_file_location("_htext_official_ce_chunked_output_loss", source_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create import spec for {source_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, "CEWithChunkedOutputLoss")
        _record_resolution(
            "ce_loss",
            BackendResolution(
                requested_backend=requested,
                actual_backend="torchtune.modules.loss.ce_chunked_output_loss.CEWithChunkedOutputLoss",
                import_path=str(source_file),
                implementation_file=str(source_file),
                fallback_used=False,
                fallback_reason=None,
            ),
        )
        return cls
    except Exception as source_exc:
        _record_resolution(
            "ce_loss",
            BackendResolution(
                requested_backend=requested,
                actual_backend="htext.functional_cross_entropy",
                import_path=None,
                implementation_file=__file__,
                fallback_used=True,
                fallback_reason=f"package_import={package_error!r}; source_file_import={source_exc!r}",
            ),
        )
        return None


def heima_shifted_thinking_mask(tokens: torch.Tensor, thinking_token_id: int) -> torch.Tensor:
    """Mirror Heima's shifted mask for the hidden state that predicts a thinking token.

    Source reference:
    heima/main_python/2-training-pipeline-main_lora-pure_llm_decoder_lora-split_3_stages-fix_num-progressive.py:1503-1532
    """
    mask = tokens[:, 1:] == thinking_token_id
    pad = torch.zeros((mask.shape[0], 1), dtype=torch.bool, device=mask.device)
    return torch.cat([mask, pad], dim=1)


def direct_thinking_mask(tokens: torch.Tensor, thinking_token_id: int) -> torch.Tensor:
    return tokens == thinking_token_id


def build_predictor_mask(input_ids: torch.Tensor, thinking_token_id: int) -> torch.Tensor:
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be [B, L], got {tuple(input_ids.shape)}")
    selected_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    selected_mask[:, :-1] = input_ids[:, 1:].eq(thinking_token_id)
    return selected_mask


def extract_thinking_state(
    *,
    input_ids: torch.Tensor,
    last_hidden_state: torch.Tensor,
    thinking_token_id: int,
    mode: ThinkingStateMode,
) -> ThinkingStateOutput:
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be [B, L], got {tuple(input_ids.shape)}")
    if last_hidden_state.ndim != 3:
        raise ValueError(f"last_hidden_state must be [B, L, D], got {tuple(last_hidden_state.shape)}")
    if input_ids.shape[:2] != last_hidden_state.shape[:2]:
        raise ValueError("input_ids and hidden-state sequence dimensions differ")
    thinking_mask = input_ids.eq(thinking_token_id)
    per_sample_count = thinking_mask.sum(dim=1)
    if not torch.all(per_sample_count == 1):
        raise ValueError(
            "Strict single-stage Heima mode requires exactly one thinking token per sample; "
            f"counts={per_sample_count.tolist()}"
        )
    if mode == "predictor":
        selected_mask = build_predictor_mask(input_ids, thinking_token_id)
        semantics = "predicts_thinking_token"
    elif mode == "token":
        selected_mask = thinking_mask
        semantics = "contextual_thinking_token_state"
    else:
        raise ValueError(f"Unsupported thinking-state mode: {mode}")
    selected_count = selected_mask.sum(dim=1)
    if not torch.all(selected_count == 1):
        raise ValueError(f"Expected one selected hidden per sample, got {selected_count.tolist()}")
    thinking_positions = thinking_mask.long().argmax(dim=1)
    selected_positions = selected_mask.long().argmax(dim=1)
    if mode == "predictor":
        if torch.any(thinking_positions == 0):
            raise ValueError("<THINKING> cannot be the first token in predictor mode")
        if not torch.equal(selected_positions, thinking_positions - 1):
            raise AssertionError("Predictor-state index is not p-1")
    batch = torch.arange(input_ids.size(0), device=input_ids.device)
    hidden = last_hidden_state[batch, selected_positions, :]
    return ThinkingStateOutput(
        hidden=hidden,
        thinking_mask=thinking_mask,
        selected_mask=selected_mask,
        thinking_positions=thinking_positions,
        selected_positions=selected_positions,
        semantics=semantics,
    )


class HeimaOfficialAbstractProjection(nn.Module):
    """Official-shape projector from torchtune TransformerDecoder.

    Official source:
    torchtune_pkg/torchtune/torchtune/modules/transformer.py:404-416
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


OfficialCompatibleAbstractProjection = HeimaOfficialAbstractProjection


def official_projector_spec(input_dim: int = 4096, output_dim: int = 4096) -> dict:
    return {
        "class": "torch.nn.Sequential",
        "source": "torchtune_pkg/torchtune/torchtune/modules/transformer.py:404-416",
        "layer_order": ["Linear", "ReLU(inplace=True)", "Linear", "Dropout(p=0.0)"],
        "input_dim": input_dim,
        "output_dim": output_dim,
        "bias": True,
        "activation": "ReLU(inplace=True)",
        "normalization": None,
        "initialization": {
            "construction": "torch.nn.Linear default initialization",
            "heima_setup_note": "training script later initializes abstract_projection Linear weights with xavier_uniform_ when no state dict is loaded; biases are not explicitly changed",
            "source": "heima/main_python/2-training-pipeline-main_lora-pure_llm_decoder_lora-split_3_stages-fix_num-progressive.py:780-786",
        },
    }


def official_embedding_replacement(
    token_embeds: torch.Tensor,
    thinking_token: torch.Tensor,
    thinking_token_mask: torch.Tensor,
) -> torch.Tensor:
    """Mirror official Torchtune replacement after token embedding lookup.

    Source reference:
    torchtune_pkg/torchtune/torchtune/modules/transformer.py:669-701
    """
    if thinking_token_mask.dtype is not torch.bool:
        raise TypeError("thinking_token_mask must be bool")
    if thinking_token_mask.shape != token_embeds.shape[:2]:
        raise ValueError("thinking_token_mask shape must match token_embeds [batch, seq]")
    if thinking_token_mask.sum().item() != thinking_token.shape[0] * thinking_token.shape[1]:
        raise ValueError("mask true count must equal thinking_token batch*num_replace_tokens")
    out = token_embeds.clone()
    flat_mask = thinking_token_mask.reshape(-1)
    flat_hidden = out.reshape(-1, out.shape[-1])
    flat_thinking = thinking_token.reshape(-1, out.shape[-1])
    flat_hidden[flat_mask] = flat_thinking
    return flat_hidden.view_as(out)


def prepare_latent_for_decoder(z: torch.Tensor, detach_encoder_latent: bool) -> torch.Tensor:
    return z.detach() if detach_encoder_latent else z


def heima_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Use Torchtune's CEWithChunkedOutputLoss when importable, with HF logits.

    Heima shifts labels before calling its Torchtune loss. This helper keeps the
    same shifted-label convention for standard Hugging Face logits.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if (shift_labels != -100).sum().item() == 0:
        raise ValueError("CE loss requires at least one non-ignored target token")
    cls = resolve_ce_loss_backend()
    if cls is None:
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )
    return cls(num_output_chunks=1, ignore_index=-100)([shift_logits], shift_labels)


def hf_shifted_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if (shift_labels != -100).sum().item() == 0:
        raise ValueError("CE loss requires at least one non-ignored target token")
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )
