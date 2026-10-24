from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.esm.modeling_esm import EsmModel, EsmPreTrainedModel

from .utils import Pooler


def minkowski_inner_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    return -(x[..., 0] * y[..., 0]) + (x[..., 1:] * y[..., 1:]).sum(dim=-1)


def hyperboloid_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    curvature: Optional[Union[torch.Tensor, float]] = None,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    inner = torch.clamp(-minkowski_inner_product(x, y), min=1.0 + epsilon)
    distance = torch.acosh(inner)

    if curvature is None:
        return distance

    curvature_tensor = torch.as_tensor(curvature, device=x.device, dtype=torch.float32)
    scale_factor = 1.0 / torch.sqrt(torch.abs(curvature_tensor).clamp_min(epsilon))
    return distance * scale_factor


class HyperboloidEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.projection = nn.Linear(dim, dim + 1)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.projection(x.to(torch.float32))
        spatial = x_proj[..., 1:]
        time_component = torch.sqrt(1.0 + spatial.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
        return torch.cat([time_component, spatial], dim=-1)


class PairEmbeddingDistance(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.curvature = nn.Parameter(torch.tensor([-1.0], dtype=torch.float32))
        self.embedding = HyperboloidEmbedding(dim)
        nn.init.uniform_(self.curvature, -2.0, -0.5)

    def _apply_constraints(self) -> None:
        with torch.no_grad():
            self.curvature.clamp_(max=-1e-6)

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        self._apply_constraints()
        emb1_h = self.embedding.project(emb1)
        emb2_h = self.embedding.project(emb2)
        return hyperboloid_distance(emb1_h, emb2_h, curvature=self.curvature)


class LucaPhyloHyper(EsmPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config)
        self.model_args = model_kwargs.get("model_args")
        pooler_type = getattr(self.model_args, "pooler_type", "avg")
        self.loss_weighted_method = getattr(self.model_args, "loss_weighted_method", "ols")
        self.loss_weight_eps = getattr(self.model_args, "loss_weight_eps", 1e-2)
        if self.loss_weighted_method not in ("ols", "fm", "be"):
            raise ValueError(
                f"Unsupported loss_weighted_method `{self.loss_weighted_method}`; "
                "expected one of: ols, fm, be."
            )

        self.esm = EsmModel(config, add_pooling_layer=False)
        self.pooler = Pooler(pooler_type)
        self.scale = PairEmbeddingDistance(config.hidden_size)
        self.init_weights()

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.pooler.requires_hidden_states,
            return_dict=True,
        )

    def encode_sequences(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self._encode(input_ids=input_ids, attention_mask=attention_mask)
        return self.pooler(attention_mask, outputs).to(torch.float32)

    def predict_distance_from_embeddings(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self.scale(anchor_embeddings.to(torch.float32), positive_embeddings.to(torch.float32))

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if isinstance(input_ids, (list, tuple)):
            input_ids_1, input_ids_2 = input_ids
            attention_mask_1, attention_mask_2 = attention_mask
        else:
            if input_ids is None or attention_mask is None or input_ids.dim() != 3:
                raise ValueError(
                    "Stage-4 expects either a list of two tensors or tensors with shape "
                    "[batch, 2, sequence_length]."
                )
            input_ids_1 = input_ids[:, 0]
            input_ids_2 = input_ids[:, 1]
            attention_mask_1 = attention_mask[:, 0]
            attention_mask_2 = attention_mask[:, 1]

        z1 = self.encode_sequences(input_ids_1, attention_mask_1)
        z2 = self.encode_sequences(input_ids_2, attention_mask_2)
        distances = self.predict_distance_from_embeddings(z1, z2)

        loss = None
        if labels is not None:
            labels = labels.to(distances.device, dtype=distances.dtype).view_as(distances)
            if self.loss_weighted_method == "ols":
                loss = F.mse_loss(distances, labels)
            else:
                ref = labels.clamp_min(self.loss_weight_eps)
                if self.loss_weighted_method == "fm":
                    weight = 1.0 / ref.pow(2)
                else:
                    weight = 1.0 / ref
                weight = weight / weight.mean()
                loss = ((distances - labels) ** 2 * weight).mean()

        return SequenceClassifierOutput(
            loss=loss,
            logits=distances,
            hidden_states=None,
            attentions=None,
        )
