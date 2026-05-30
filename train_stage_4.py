import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process

from src.stage4_data import (
    Stage4DataCollator,
    Stage4DistanceDataset,
    construct_pair_dataframe,
    create_random_split,
    load_tree_distance_matrix,
    normalize_split_ids,
    parse_fasta,
    save_stage4_dataset_artifacts,
    validate_split_ids,
)
from src.stage4_model import LucaPhyloHyper
from src.utils import MetricsLogger, SaveTokenizerCallback

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Checkpoint from stage-2/stage-3 or a compatible ESM model."},
    )
    pooler_type: str = field(default="avg")
    cache_dir: Optional[str] = field(default=None)
    use_lora: bool = field(default=True)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)
    lora_target_modules: str = field(
        default="query,key,value",
        metadata={"help": "Comma-separated ESM attention projections to adapt with LoRA."},
    )


@dataclass
class DataArguments:
    tree_path: str = field(
        default=None,
        metadata={"help": "Backbone tree in Newick format."},
    )
    fasta_path: str = field(
        default=None,
        metadata={"help": "FASTA file whose record IDs match the tree leaves."},
    )
    split_file: Optional[str] = field(
        default=None,
        metadata={"help": "Optional CSV split definition. Supports train/valid/test columns or id/split rows."},
    )
    max_seq_length: int = field(default=2048)
    include_self_pairs: bool = field(default=False)
    tree_distance_backend: str = field(default="ete3")
    tree_distance_workers: int = field(default=0)
    train_ratio: float = field(default=0.7)
    valid_ratio: float = field(default=0.15)
    test_ratio: float = field(default=0.15)


def ensure_context_length(config, max_seq_length: int) -> None:
    max_positions = getattr(config, "max_position_embeddings", None)
    if max_positions is None:
        return

    required_positions = max_seq_length + 2
    if required_positions > max_positions:
        raise ValueError(
            "Stage-4 expects a base checkpoint whose positional embeddings already support the "
            f"requested length. Required {required_positions}, found {max_positions}. "
            "Please start from a stage-2 or stage-3 checkpoint trained at this context length."
        )


def build_stage4_dataframes(data_args: DataArguments, seed: int, include_test_split: bool):
    sequence_by_id = parse_fasta(data_args.fasta_path)
    distance_matrix, taxa, id2index = load_tree_distance_matrix(
        data_args.tree_path,
        backend=data_args.tree_distance_backend,
        num_workers=data_args.tree_distance_workers,
    )

    available_ids = [taxon_id for taxon_id in taxa if taxon_id in sequence_by_id]
    missing_in_fasta = sorted(set(taxa).difference(sequence_by_id.keys()))
    if missing_in_fasta:
        preview = missing_in_fasta[:10]
        raise ValueError(f"Tree leaves missing from FASTA: {preview}")

    if data_args.split_file:
        split_ids = normalize_split_ids(data_args.split_file)
    else:
        split_ids = create_random_split(
            available_ids,
            train_ratio=data_args.train_ratio,
            valid_ratio=data_args.valid_ratio,
            test_ratio=data_args.test_ratio,
            seed=seed,
        )

    if not include_test_split and split_ids["test_id"]:
        logger.info("Ignoring test split because `--do_predict` is disabled.")
        split_ids["test_id"] = []

    validate_split_ids(split_ids, available_ids)

    train_df = construct_pair_dataframe(
        split_ids["train_id"],
        sequence_by_id,
        split_ids["train_id"],
        distance_matrix,
        id2index,
        include_self_pairs=data_args.include_self_pairs,
    )

    valid_parts = []
    if split_ids["valid_id"]:
        valid_parts.append(
            construct_pair_dataframe(
                split_ids["valid_id"],
                sequence_by_id,
                split_ids["train_id"],
                distance_matrix,
                id2index,
                include_self_pairs=data_args.include_self_pairs,
            )
        )
        valid_parts.append(
            construct_pair_dataframe(
                split_ids["valid_id"],
                sequence_by_id,
                split_ids["valid_id"],
                distance_matrix,
                id2index,
                include_self_pairs=data_args.include_self_pairs,
            )
        )
    valid_df = pd.concat(valid_parts, ignore_index=True) if valid_parts else pd.DataFrame(
        columns=["id1", "id2", "seq1", "seq2", "distance"]
    )

    test_parts = []
    if split_ids["test_id"]:
        if split_ids["train_id"]:
            test_parts.append(
                construct_pair_dataframe(
                    split_ids["test_id"],
                    sequence_by_id,
                    split_ids["train_id"],
                    distance_matrix,
                    id2index,
                    include_self_pairs=data_args.include_self_pairs,
                )
            )
        if split_ids["valid_id"]:
            test_parts.append(
                construct_pair_dataframe(
                    split_ids["test_id"],
                    sequence_by_id,
                    split_ids["valid_id"],
                    distance_matrix,
                    id2index,
                    include_self_pairs=data_args.include_self_pairs,
                )
            )
        test_parts.append(
            construct_pair_dataframe(
                split_ids["test_id"],
                sequence_by_id,
                split_ids["test_id"],
                distance_matrix,
                id2index,
                include_self_pairs=data_args.include_self_pairs,
            )
        )
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(
        columns=["id1", "id2", "seq1", "seq2", "distance"]
    )

    metadata = {
        "seed": seed,
        "tree_path": data_args.tree_path,
        "fasta_path": data_args.fasta_path,
        "split_file": data_args.split_file,
        "include_self_pairs": data_args.include_self_pairs,
        "tree_distance_backend": data_args.tree_distance_backend,
        "tree_distance_workers": data_args.tree_distance_workers,
        "pair_counts": {
            "train": int(len(train_df)),
            "valid": int(len(valid_df)),
            "test": int(len(test_df)),
        },
        "split_sizes": {
            "train": int(len(split_ids["train_id"])),
            "valid": int(len(split_ids["valid_id"])),
            "test": int(len(split_ids["test_id"])),
        },
    }
    return (
        split_ids,
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        metadata,
    )


def compute_regression_metrics(eval_prediction):
    predictions = np.asarray(eval_prediction.predictions).reshape(-1)
    labels = np.asarray(eval_prediction.label_ids).reshape(-1)

    mse = float(np.mean((predictions - labels) ** 2))
    mae = float(np.mean(np.abs(predictions - labels)))
    safe_labels = np.where(np.abs(labels) < 1e-8, 1e-8, labels)
    mape = float(np.mean(np.abs((labels - predictions) / safe_labels)) * 100.0)

    if len(predictions) < 2 or np.std(predictions) < 1e-12 or np.std(labels) < 1e-12:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(predictions, labels)[0, 1])
        if np.isnan(pearson):
            pearson = 0.0

    return {
        "mse": mse,
        "mae": mae,
        "mape": mape,
        "pearson": pearson,
    }


def save_stage4_metadata(
    output_dir: str,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> None:
    payload = {
        "model_args": asdict(model_args),
        "data_args": asdict(data_args),
        "training_args": training_args.to_dict(),
        "command": sys.argv,
    }
    with open(os.path.join(output_dir, "stage4_run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with open(os.path.join(output_dir, "base_model_reference.json"), "w", encoding="utf-8") as handle:
        json.dump({"model_name_or_path": model_args.model_name_or_path}, handle, indent=2)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    training_args.remove_unused_columns = False
    set_seed(training_args.seed)

    split_ids, train_df, valid_df, test_df, metadata = build_stage4_dataframes(
        data_args=data_args,
        seed=training_args.seed,
        include_test_split=training_args.do_predict,
    )
    save_stage4_dataset_artifacts(
        output_dir=training_args.output_dir,
        split_ids=split_ids,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        metadata=metadata,
    )
    save_stage4_metadata(training_args.output_dir, model_args, data_args, training_args)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    ensure_context_length(config, data_args.max_seq_length)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        truncation_side="left",
    )
    model = LucaPhyloHyper.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        model_args=model_args,
    )

    if model_args.use_lora:
        target_modules = [module.strip() for module in model_args.lora_target_modules.split(",") if module.strip()]
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            target_modules=target_modules,
            modules_to_save=["scale"],
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        model = get_peft_model(model, lora_config)
        if is_main_process(training_args.local_rank):
            model.print_trainable_parameters()

    train_dataset = Stage4DistanceDataset(
        train_df,
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
    )
    eval_dataset = None
    if not valid_df.empty and training_args.do_eval:
        eval_dataset = Stage4DistanceDataset(
            valid_df,
            tokenizer=tokenizer,
            max_length=data_args.max_seq_length,
        )
    test_dataset = None
    if not test_df.empty and training_args.do_predict:
        test_dataset = Stage4DistanceDataset(
            test_df,
            tokenizer=tokenizer,
            max_length=data_args.max_seq_length,
        )
    data_collator = Stage4DataCollator(
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_regression_metrics,
        callbacks=[
            MetricsLogger(os.path.join(training_args.output_dir, "logs")),
            SaveTokenizerCallback(tokenizer),
        ],
    )

    if training_args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None)
        )
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        if eval_dataset is None:
            raise ValueError("Evaluation is enabled but no validation examples are available.")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    if training_args.do_predict:
        if test_dataset is None:
            raise ValueError("Prediction is enabled but no test examples are available.")
        test_metrics = trainer.predict(test_dataset).metrics
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)


if __name__ == "__main__":
    main()
