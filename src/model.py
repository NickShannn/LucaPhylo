import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.esm.modeling_esm import EsmModel, EsmPreTrainedModel

from .utils import Pooler, Similarity, gather_with_grad, ortho_regularization


class LucaPhylo(EsmPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config)

        self.model_args = model_kwargs.get("model_args")
        self.stage = getattr(self.model_args, "training_stage", 2)
        self.pooler = Pooler(getattr(self.model_args, "pooler_type", "avg"))
        self.temperature = getattr(
            self.model_args,
            "base_temp",
            getattr(self.model_args, "temp", 0.07),
        )
        self.ortho_lambda = getattr(self.model_args, "ortho_lambda", 0.0)
        self.separation_weight = getattr(self.model_args, "separation_weight", 0.0)

        self.esm = EsmModel(config, add_pooling_layer=False)
        self.similarity = Similarity(self.temperature)
        self.init_weights()

    def _prepare_pair_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        if input_ids.dim() == 3:
            return input_ids, attention_mask

        if input_ids.dim() == 4:
            outer_batch, pair_batch, pair_size, sequence_length = input_ids.shape
            return (
                input_ids.view(outer_batch * pair_batch, pair_size, sequence_length),
                attention_mask.view(outer_batch * pair_batch, pair_size, sequence_length),
            )

        raise ValueError(
            "Expected contrastive inputs with shape [batch, 2, length] or "
            "[batch, pair_batch, 2, length], "
            f"but received {tuple(input_ids.shape)}."
        )

    def _legacy_stage2_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        if not dist.is_initialized():
            return tensor

        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor.contiguous())
        gathered[dist.get_rank()] = tensor
        return torch.cat(gathered, dim=0)

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.pooler.requires_hidden_states,
            return_dict=True,
        )

    def _compute_similarity(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if self.stage == 3:
            z1 = F.normalize(z1, p=2, dim=1)
            z2 = F.normalize(z2, p=2, dim=1)
            return torch.matmul(z1, z2.transpose(0, 1)) / self.temperature

        return self.similarity(z1.unsqueeze(1), z2.unsqueeze(0))

    def _stage3_regularization(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        ortho_loss = ortho_regularization(z1) + ortho_regularization(z2)
        centroid_similarity = F.cosine_similarity(z1.mean(dim=0), z2.mean(dim=0), dim=0)
        separation_loss = torch.abs(centroid_similarity)
        return self.ortho_lambda * ortho_loss + self.separation_weight * separation_loss

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        input_ids, attention_mask = self._prepare_pair_batch(input_ids, attention_mask)
        batch_size, pair_size, sequence_length = input_ids.shape

        flat_input_ids = input_ids.view(batch_size * pair_size, sequence_length)
        flat_attention_mask = attention_mask.view(batch_size * pair_size, sequence_length)

        outputs = self._encode(flat_input_ids, flat_attention_mask)
        pooled = self.pooler(flat_attention_mask, outputs)
        pooled = pooled.view(batch_size, pair_size, pooled.size(-1))
        z1, z2 = pooled[:, 0], pooled[:, 1]

        if dist.is_initialized() and self.training:
            use_legacy_gather = self.stage == 2 and getattr(
                self.model_args, "legacy_stage2_gather", False
            )
            if use_legacy_gather:
                z1 = self._legacy_stage2_gather(z1)
                z2 = self._legacy_stage2_gather(z2)
            else:
                z1 = gather_with_grad(z1)
                z2 = gather_with_grad(z2)

        logits = self._compute_similarity(z1, z2)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = nn.CrossEntropyLoss()(logits, labels)

        if self.stage == 3:
            loss = loss + self._stage3_regularization(z1, z2)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
