import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process

from src.utils import MetricsLogger, SaveTokenizerCallback, infer_text_column

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="facebook/esm2_t33_650M_UR50D",
        metadata={"help": "Base ESM checkpoint used for stage-1 continued pretraining."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional cache directory for Hugging Face models."},
    )


@dataclass
class DataArguments:
    train_file: str = field(
        default=None,
        metadata={"help": "Training sequences in .txt, .csv, or .json format."},
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "Sequence column name for .csv/.json inputs. Auto-detected when omitted."},
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length after tokenization."},
    )
    masking_probability: float = field(
        default=0.15,
        metadata={"help": "Masking ratio for the MLM objective."},
    )
    validation_split: float = field(
        default=0.2,
        metadata={"help": "Fraction of data reserved for validation."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Optional number of dataset preprocessing workers."},
    )


def load_sequences(data_args: DataArguments):
    extension = os.path.splitext(data_args.train_file)[1].lower()
    if extension not in {".txt", ".csv", ".json"}:
        raise ValueError("Stage-1 training supports .txt, .csv, and .json inputs.")

    dataset_type = "text" if extension == ".txt" else extension.lstrip(".")
    dataset = load_dataset(dataset_type, data_files={"train": data_args.train_file})["train"]
    text_column = infer_text_column(dataset.column_names, data_args.text_column)
    return dataset, text_column


def load_stage1_model(model_args: ModelArguments):
    return AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )


def resolve_effective_max_seq_length(requested_max_seq_length: int, model) -> int:
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is None:
        return requested_max_seq_length

    supported_max_seq_length = max_positions - 2
    if requested_max_seq_length > supported_max_seq_length:
        logger.warning(
            "Requested max_seq_length=%s exceeds the base checkpoint support (%s). "
            "Stage 1 will left-truncate to %s tokens and keep the right-hand sequence region.",
            requested_max_seq_length,
            supported_max_seq_length,
            supported_max_seq_length,
        )
    return min(requested_max_seq_length, supported_max_seq_length)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )
    set_seed(training_args.seed)

    raw_dataset, text_column = load_sequences(data_args)
    split_dataset = raw_dataset.train_test_split(
        test_size=data_args.validation_split,
        seed=training_args.seed,
    )

    model = load_stage1_model(model_args)
    effective_max_seq_length = resolve_effective_max_seq_length(data_args.max_seq_length, model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        truncation_side="left",
        model_max_length=effective_max_seq_length,
    )

    def tokenize_batch(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=effective_max_seq_length,
        )

    tokenized_datasets = split_dataset.map(
        tokenize_batch,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=split_dataset["train"].column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=data_args.masking_probability,
    )
    metrics_logger = MetricsLogger(os.path.join(training_args.output_dir, "logs"))
    save_tokenizer_callback = SaveTokenizerCallback(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"] if training_args.do_eval else None,
        data_collator=data_collator,
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
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)


if __name__ == "__main__":
    main()
