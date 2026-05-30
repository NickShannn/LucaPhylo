import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process

from src.dataset import LucaContrastiveDataset, LucaLegacyStage2Dataset
from src.model import LucaPhylo
from src.utils import MetricsLogger, SaveTokenizerCallback

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Checkpoint from the previous stage or a compatible ESM model."},
    )
    training_stage: int = field(
        default=2,
        metadata={"help": "Training stage: 2 for label-based contrastive learning, 3 for paired ORF-domain learning."},
    )
    pooler_type: str = field(default="avg")
    base_temp: float = field(default=0.07)
    ortho_lambda: float = field(default=0.3)
    separation_weight: float = field(default=2.0)
    stage2_legacy_mode: bool = field(
        default=True,
        metadata={
            "help": "When training stage 2, use the original class-balanced dynamic batching."
        },
    )
    stage2_pairs_per_item: int = field(
        default=4,
        metadata={"help": "Number of contrastive pairs sampled inside each stage-2 legacy item."},
    )
    stage2_items_per_label: int = field(
        default=10,
        metadata={"help": "Number of stage-2 legacy items to sample per valid label each epoch."},
    )
    legacy_stage2_gather: bool = field(
        default=True,
        metadata={"help": "Use the original stage-2 distributed gather behavior."},
    )
    cache_dir: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    train_file: str = field(
        default=None,
        metadata={"help": "Input CSV file for stage-2 or stage-3 training."},
    )
    max_seq_length: int = field(default=1024)
    validation_split: float = field(
        default=0.1,
        metadata={"help": "Fraction of rows reserved for validation."},
    )
    col_anchor: str = field(
        default="seq",
        metadata={"help": "Column containing the anchor sequence."},
    )
    col_positive: str = field(
        default="domain",
        metadata={"help": "Positive-pair column used by stage 3."},
    )
    col_label: str = field(
        default="label",
        metadata={"help": "Supergroup label column used by stage 2."},
    )


def validate_stage(training_stage: int) -> None:
    if training_stage not in {2, 3}:
        raise ValueError("`training_stage` must be 2 or 3.")


def split_dataframe(dataframe: pd.DataFrame, data_args: DataArguments, training_stage: int):
    if data_args.validation_split <= 0:
        return dataframe.reset_index(drop=True), None

    if training_stage == 2:
        label_counts = dataframe[data_args.col_label].value_counts()
        can_stratify = not label_counts.empty and (label_counts >= 2).all()
        stratify = dataframe[data_args.col_label] if can_stratify else None
        if not can_stratify:
            logger.warning(
                "Falling back to a random split because some labels have fewer than 2 samples."
            )
    else:
        stratify = None

    train_df, val_df = train_test_split(
        dataframe,
        test_size=data_args.validation_split,
        stratify=stratify,
        random_state=42,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def ensure_columns(dataframe: pd.DataFrame, training_stage: int, data_args: DataArguments) -> None:
    required = {data_args.col_anchor}
    if training_stage == 2:
        required.add(data_args.col_label)
    else:
        required.add(data_args.col_positive)

    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required columns in `{data_args.train_file}`: {sorted(missing)}")


def save_split_artifacts(train_df: pd.DataFrame, val_df: Optional[pd.DataFrame], output_dir: str) -> None:
    split_dir = os.path.join(output_dir, "split_data")
    os.makedirs(split_dir, exist_ok=True)
    train_df.to_csv(os.path.join(split_dir, "train.csv"), index=False)
    if val_df is not None:
        val_df.to_csv(os.path.join(split_dir, "val.csv"), index=False)


def resolve_effective_max_seq_length(
    requested_max_seq_length: int, config, training_stage: int
) -> int:
    max_positions = getattr(config, "max_position_embeddings", None)
    if max_positions is None:
        return requested_max_seq_length

    supported_max_seq_length = max_positions - 2
    if requested_max_seq_length > supported_max_seq_length:
        logger.warning(
            "Requested max_seq_length=%s exceeds the stage-%s base checkpoint support (%s). "
            "Inputs will be left-truncated to %s tokens and keep the right-hand sequence region.",
            requested_max_seq_length,
            training_stage,
            supported_max_seq_length,
            supported_max_seq_length,
        )
    return min(requested_max_seq_length, supported_max_seq_length)


class LucaTrainer(Trainer):
    """Trainer with explicit DDP barriers at eval/save boundaries."""

    @staticmethod
    def _barrier():
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        self._barrier()
        return metrics

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        self._barrier()


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    validate_stage(model_args.training_stage)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )
    set_seed(training_args.seed)
    training_args.dataloader_drop_last = True


    dataframe = pd.read_csv(data_args.train_file)
    ensure_columns(dataframe, model_args.training_stage, data_args)
    train_df, val_df = split_dataframe(dataframe, data_args, model_args.training_stage)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    effective_max_seq_length = resolve_effective_max_seq_length(
        data_args.max_seq_length, config, model_args.training_stage
    )
    if model_args.training_stage in {2, 3}:
        if hasattr(config, "hidden_dropout_prob"):
            config.hidden_dropout_prob = 0.15
        if hasattr(config, "attention_probs_dropout_prob"):
            config.attention_probs_dropout_prob = 0.15

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        truncation_side="left",
        model_max_length=effective_max_seq_length,
    )
    model = LucaPhylo.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        model_args=model_args,
    )

    if is_main_process(training_args.local_rank):
        save_split_artifacts(train_df, val_df, training_args.output_dir)

    use_legacy_stage2 = model_args.training_stage == 2 and model_args.stage2_legacy_mode
    if use_legacy_stage2:
        train_dataset = LucaLegacyStage2Dataset(
            train_df,
            tokenizer,
            max_length=effective_max_seq_length,
            col_anchor=data_args.col_anchor,
            col_label=data_args.col_label,
            pairs_per_item=model_args.stage2_pairs_per_item,
            items_per_label=model_args.stage2_items_per_label,
            is_eval=False,
        )
    else:
        train_dataset = LucaContrastiveDataset(
            train_df,
            tokenizer,
            max_length=effective_max_seq_length,
            stage=model_args.training_stage,
            col_anchor=data_args.col_anchor,
            col_positive=data_args.col_positive,
            col_label=data_args.col_label,
        )

    eval_dataset = None
    if val_df is not None and training_args.do_eval:
        if use_legacy_stage2:
            eval_dataset = LucaLegacyStage2Dataset(
                val_df,
                tokenizer,
                max_length=effective_max_seq_length,
                col_anchor=data_args.col_anchor,
                col_label=data_args.col_label,
                pairs_per_item=model_args.stage2_pairs_per_item,
                items_per_label=model_args.stage2_items_per_label,
                is_eval=True,
            )
        else:
            eval_dataset = LucaContrastiveDataset(
                val_df,
                tokenizer,
                max_length=effective_max_seq_length,
                stage=model_args.training_stage,
                col_anchor=data_args.col_anchor,
                col_positive=data_args.col_positive,
                col_label=data_args.col_label,
            )

    conventional_epoch_config = None
    if use_legacy_stage2:
        conventional_epoch_config = {
            "train_rows": len(train_df),
            "sequences_per_item": model_args.stage2_pairs_per_item * 2,
        }

    metrics_logger = MetricsLogger(
        os.path.join(training_args.output_dir, "logs"),
        conventional_epoch_config=conventional_epoch_config,
    )
    save_tokenizer_callback = SaveTokenizerCallback(tokenizer)
    trainer = LucaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[metrics_logger, save_tokenizer_callback],
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        if eval_dataset is None:
            raise ValueError("Evaluation is enabled but no validation split is available.")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)


if __name__ == "__main__":
    main()
