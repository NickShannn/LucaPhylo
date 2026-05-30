import csv
import json
import logging
import math
import os
from datetime import datetime
from typing import Iterable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

logger = logging.getLogger(__name__)


class GatherLayer(torch.autograd.Function):
    """All-gather with backward support for contrastive training."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor):
        if not dist.is_initialized():
            return (tensor,)

        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return tuple(gathered)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor):
        if not dist.is_initialized():
            return grads[0]

        stacked_grads = torch.stack(grads)
        dist.all_reduce(stacked_grads)
        return stacked_grads[dist.get_rank()]


def gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return tensor
    return torch.cat(GatherLayer.apply(tensor), dim=0)


def ortho_regularization(embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.size(0) <= 1:
        return embeddings.new_tensor(0.0)

    normalized = F.normalize(embeddings, p=2, dim=1)
    correlation = normalized @ normalized.transpose(0, 1)
    identity = torch.eye(correlation.size(0), device=embeddings.device, dtype=correlation.dtype)
    return torch.norm(correlation - identity, p="fro")


class Similarity(nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature
        self.cosine = nn.CosineSimilarity(dim=-1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.cosine(x, y) / self.temperature


class Pooler(nn.Module):
    def __init__(self, pooler_type: str):
        super().__init__()
        valid_poolers = {"cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"}
        if pooler_type not in valid_poolers:
            raise ValueError(f"Unsupported pooler_type: {pooler_type}")
        self.pooler_type = pooler_type

    @property
    def requires_hidden_states(self) -> bool:
        return self.pooler_type in {"avg_top2", "avg_first_last"}

    def forward(self, attention_mask: torch.Tensor, outputs) -> torch.Tensor:
        last_hidden = outputs.last_hidden_state

        if self.pooler_type in {"cls", "cls_before_pooler"}:
            return last_hidden[:, 0]

        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        denom = mask.sum(dim=1).clamp_min(1e-12)

        if self.pooler_type == "avg":
            return (last_hidden * mask).sum(dim=1) / denom

        if outputs.hidden_states is None:
            raise ValueError(f"Pooler `{self.pooler_type}` requires hidden states.")

        if self.pooler_type == "avg_first_last":
            hidden = (outputs.hidden_states[1] + outputs.hidden_states[-1]) / 2.0
            return (hidden * mask).sum(dim=1) / denom

        hidden = (outputs.hidden_states[-1] + outputs.hidden_states[-2]) / 2.0
        return (hidden * mask).sum(dim=1) / denom


def infer_text_column(column_names: Iterable[str], preferred: Optional[str] = None) -> str:
    column_names = list(column_names)

    if preferred:
        if preferred not in column_names:
            raise ValueError(f"Column `{preferred}` not found in dataset columns: {column_names}")
        return preferred

    for candidate in ("text", "sequence", "seq"):
        if candidate in column_names:
            return candidate

    if len(column_names) == 1:
        return column_names[0]

    raise ValueError(
        "Could not infer which column contains sequences. "
        f"Please provide `--text_column`. Available columns: {column_names}"
    )


class MetricsLogger(TrainerCallback):
    """Persist a lightweight JSON log for public release runs."""

    def __init__(self, log_dir: str, conventional_epoch_config: Optional[dict] = None):
        self.log_dir = log_dir
        self.metrics_path = os.path.join(log_dir, "training_metrics.json")
        self.history_path = os.path.join(log_dir, "metrics_history.json")
        self.step_metrics_csv_path = os.path.join(log_dir, "step_metrics.csv")
        self.epoch_metrics_csv_path = os.path.join(log_dir, "epoch_eval_metrics.csv")
        self.log_path = os.path.join(log_dir, "training.log")
        self.plots_dir = os.path.join(log_dir, "plots")
        self.conventional_epoch_config = conventional_epoch_config
        self.metrics = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "epoch": [],
            "conventional_epoch_estimate": [],
            "stage4_curvature": [],
            "eval_mse": [],
            "eval_mae": [],
            "eval_mape": [],
            "eval_pearson": [],
        }
        self.history = []
        self.start_time: Optional[datetime] = None
        self._plot_warning_emitted = False
        os.makedirs(log_dir, exist_ok=True)

    def _extract_stage4_curvature(self, model) -> Optional[float]:
        visited = set()
        current = model

        while current is not None and id(current) not in visited:
            visited.add(id(current))

            scale = getattr(current, "scale", None)
            curvature = getattr(scale, "curvature", None) if scale is not None else None
            if curvature is not None:
                value = curvature.detach().float().view(-1)[0].item()
                return float(value)

            next_current = None
            if hasattr(current, "module"):
                next_current = current.module
            elif hasattr(current, "get_base_model"):
                try:
                    next_current = current.get_base_model()
                except Exception:
                    next_current = None
            elif hasattr(current, "model"):
                next_current = current.model
            elif hasattr(current, "base_model"):
                next_current = current.base_model

            current = next_current

        return None

    def _is_main_process(self, state) -> bool:
        return getattr(state, "is_world_process_zero", True)

    def _world_size(self, args) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return max(1, int(getattr(args, "world_size", 1)))

    def _estimate_conventional_epoch(self, args, state) -> Optional[float]:
        if not self.conventional_epoch_config or state.global_step <= 0:
            return None

        train_rows = self.conventional_epoch_config.get("train_rows", 0)
        sequences_per_item = self.conventional_epoch_config.get("sequences_per_item", 0)
        if train_rows <= 0 or sequences_per_item <= 0:
            return None

        per_device_batch = int(getattr(args, "per_device_train_batch_size", 1))
        grad_accum = int(getattr(args, "gradient_accumulation_steps", 1))
        world_size = self._world_size(args)
        sequences_per_optimizer_step = (
            per_device_batch * grad_accum * world_size * sequences_per_item
        )
        return state.global_step * sequences_per_optimizer_step / float(train_rows)

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._is_main_process(state):
            return

        self.start_time = datetime.now()
        with open(self.log_path, "w", encoding="utf-8") as handle:
            handle.write(f"Training started at: {self.start_time.isoformat()}\n")
            if self.conventional_epoch_config:
                train_rows = self.conventional_epoch_config.get("train_rows")
                sequences_per_item = self.conventional_epoch_config.get("sequences_per_item")
                handle.write(
                    "Conventional epoch estimate enabled: "
                    f"train_rows={train_rows}, sequences_per_item={sequences_per_item}\n"
                )

    def _to_float(self, value) -> Optional[float]:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            value = value.detach().cpu().item()

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(numeric_value):
            return None
        return numeric_value

    def _build_record(self, state, logs) -> dict:
        record = {
            "timestamp": datetime.now().isoformat(),
            "global_step": int(state.global_step),
        }

        epoch_value = self._to_float(getattr(state, "epoch", None))
        if epoch_value is not None:
            record["epoch"] = epoch_value

        for key, value in logs.items():
            numeric_value = self._to_float(value)
            if numeric_value is not None:
                record[key] = numeric_value

        if any(key.startswith("eval_") for key in record):
            record["event"] = "eval"
        elif "loss" in record or "train_loss" in record:
            record["event"] = "train"
        else:
            record["event"] = "log"

        return record

    def _append_metric_point(self, metric_name: str, global_step: int, value: float) -> None:
        self.metrics.setdefault(metric_name, []).append((global_step, value))

    def _record_metrics(self, record: dict) -> None:
        metric_aliases = {
            "loss": "train_loss",
            "eval_loss": "eval_loss",
            "learning_rate": "learning_rate",
            "epoch": "epoch",
            "conventional_epoch_estimate": "conventional_epoch_estimate",
            "stage4_curvature": "stage4_curvature",
            "eval_mse": "eval_mse",
            "eval_mae": "eval_mae",
            "eval_mape": "eval_mape",
            "eval_pearson": "eval_pearson",
        }

        global_step = int(record["global_step"])
        for key, value in record.items():
            if key in {"timestamp", "global_step", "event"}:
                continue
            if not isinstance(value, (int, float)):
                continue

            self._append_metric_point(key, global_step, float(value))
            alias = metric_aliases.get(key)
            if alias and alias != key:
                self._append_metric_point(alias, global_step, float(value))

    def _ordered_fieldnames(self, rows) -> list[str]:
        preferred = [
            "timestamp",
            "event",
            "global_step",
            "epoch",
            "train_loss",
            "loss",
            "eval_loss",
            "eval_mse",
            "eval_mae",
            "eval_mape",
            "eval_pearson",
            "learning_rate",
            "stage4_curvature",
            "conventional_epoch_estimate",
        ]
        present = set()
        for row in rows:
            present.update(row.keys())

        fieldnames = [name for name in preferred if name in present]
        fieldnames.extend(sorted(name for name in present if name not in fieldnames))
        return fieldnames

    def _write_csv(self, path: str, rows) -> None:
        if not rows:
            return

        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._ordered_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)

    def _build_epoch_eval_rows(self):
        rows_by_key = {}
        for record in self.history:
            if record.get("event") != "eval":
                continue

            dedupe_key = (
                int(record["global_step"]),
                record.get("epoch"),
            )
            rows_by_key[dedupe_key] = {
                "timestamp": record.get("timestamp"),
                "global_step": int(record["global_step"]),
                "epoch": record.get("epoch"),
                "eval_loss": record.get("eval_loss"),
                "eval_mse": record.get("eval_mse"),
                "eval_mae": record.get("eval_mae"),
                "eval_mape": record.get("eval_mape"),
                "eval_pearson": record.get("eval_pearson"),
                "learning_rate": record.get("learning_rate"),
                "stage4_curvature": record.get("stage4_curvature"),
                "conventional_epoch_estimate": record.get("conventional_epoch_estimate"),
            }

        return sorted(rows_by_key.values(), key=lambda row: (row["global_step"], row.get("epoch") or -1.0))

    def _load_matplotlib(self):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            return plt
        except ImportError:
            if not self._plot_warning_emitted:
                logger.warning(
                    "matplotlib is not installed; skipping metric plot export for `%s`.",
                    self.log_dir,
                )
                self._plot_warning_emitted = True
            return None

    def _plot_series(self, plt, series, title: str, xlabel: str, ylabel: str, output_path: str) -> None:
        if not series:
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        figure, axis = plt.subplots(figsize=(8, 5))
        for label, points in series:
            if not points:
                continue
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            axis.plot(x_values, y_values, marker="o", linewidth=1.8, markersize=3, label=label)

        if not axis.lines:
            plt.close(figure)
            return

        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        if len(axis.lines) > 1:
            axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=200)
        plt.close(figure)

    def _plot_metric_subplots(self, plt, metric_series, title: str, xlabel: str, output_path: str) -> None:
        available_series = [(metric_name, points) for metric_name, points in metric_series if points]
        if not available_series:
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        n_metrics = len(available_series)
        n_cols = 2 if n_metrics > 1 else 1
        n_rows = (n_metrics + n_cols - 1) // n_cols

        figure, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        for axis, (metric_name, points) in zip(axes_flat, available_series):
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            axis.plot(x_values, y_values, marker="o", linewidth=1.8, markersize=3)
            axis.set_title(metric_name)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(metric_name)
            axis.grid(True, alpha=0.3)

        for axis in axes_flat[n_metrics:]:
            axis.set_visible(False)

        figure.suptitle(title)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.savefig(output_path, dpi=200)
        plt.close(figure)

    def _save_plots(self) -> None:
        plt = self._load_matplotlib()
        if plt is None:
            return

        train_loss_points = self.metrics.get("train_loss", [])
        eval_loss_points = self.metrics.get("eval_loss", [])
        self._plot_series(
            plt,
            [
                ("train_loss", train_loss_points),
                ("eval_loss", eval_loss_points),
            ],
            title="Training and Validation Loss",
            xlabel="Global Step",
            ylabel="Loss",
            output_path=os.path.join(self.plots_dir, "loss_curve.png"),
        )

        epoch_eval_rows = self._build_epoch_eval_rows()
        regression_metrics = [
            "eval_mse",
            "eval_mae",
            "eval_mape",
            "eval_pearson",
        ]
        x_key = "epoch" if epoch_eval_rows and any(row.get("epoch") is not None for row in epoch_eval_rows) else "global_step"
        regression_series = []
        for metric_name in regression_metrics:
            points = []
            for row in epoch_eval_rows:
                x_value = row.get(x_key)
                y_value = row.get(metric_name)
                if x_value is None or y_value is None:
                    continue
                points.append((x_value, y_value))
            if points:
                regression_series.append((metric_name, points))

        self._plot_metric_subplots(
            plt,
            regression_series,
            title="Stage-4 Evaluation Metrics",
            xlabel="Epoch" if x_key == "epoch" else "Global Step",
            output_path=os.path.join(self.plots_dir, "eval_regression_metrics.png"),
        )

        self._plot_series(
            plt,
            [("stage4_curvature", self.metrics.get("stage4_curvature", []))],
            title="Stage-4 Curvature",
            xlabel="Global Step",
            ylabel="Curvature",
            output_path=os.path.join(self.plots_dir, "stage4_curvature.png"),
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self._is_main_process(state) or not logs:
            return

        conventional_epoch = self._estimate_conventional_epoch(args, state)
        if conventional_epoch is not None:
            logs["conventional_epoch_estimate"] = conventional_epoch

        curvature = self._extract_stage4_curvature(kwargs.get("model"))
        if curvature is not None:
            logs["stage4_curvature"] = curvature

        record = self._build_record(state, logs)
        self.history.append(record)
        self._record_metrics(record)

        if curvature is not None or conventional_epoch is not None:
            message_parts = [f"Step {state.global_step}:"]
            if curvature is not None:
                message_parts.append(f"stage4_curvature={curvature:.6f}")
            if "epoch" in logs:
                message_parts.append(f"epoch={logs['epoch']}")
            if conventional_epoch is not None:
                message_parts.append(f"conventional_epoch_estimate={conventional_epoch:.4f}")
            message = ", ".join(message_parts)
            logger.info(message)
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(message + "\n")
        self._save_metrics(refresh_plots=record.get("event") == "eval")

    def on_train_end(self, args, state, control, **kwargs):
        if not self._is_main_process(state):
            return

        end_time = datetime.now()
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(f"Training finished at: {end_time.isoformat()}\n")
            if self.start_time is not None:
                handle.write(f"Duration: {end_time - self.start_time}\n")
        self._save_metrics(refresh_plots=True)

    def _save_metrics(self, refresh_plots: bool = False) -> None:
        payload = dict(self.metrics)
        payload["history"] = self.history
        with open(self.metrics_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        with open(self.history_path, "w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2)

        self._write_csv(self.step_metrics_csv_path, self.history)
        self._write_csv(self.epoch_metrics_csv_path, self._build_epoch_eval_rows())

        if refresh_plots:
            self._save_plots()


class SaveTokenizerCallback(TrainerCallback):
    """Save tokenizer files alongside every checkpoint directory."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _is_main_process(self, state) -> bool:
        return getattr(state, "is_world_process_zero", True)

    def on_save(self, args, state, control, **kwargs):
        if not self._is_main_process(state):
            return

        checkpoint_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        self.tokenizer.save_pretrained(checkpoint_dir)
