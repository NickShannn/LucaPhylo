import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from Bio import Phylo
from torch.utils.data import Dataset


def parse_fasta(path: str) -> "OrderedDict[str, str]":
    sequences: "OrderedDict[str, str]" = OrderedDict()
    current_id = None
    chunks: List[str] = []

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(chunks).upper()
                current_id = line[1:].strip().split()[0]
                chunks = []
                continue

            chunks.append("".join(line.split()))

    if current_id is not None:
        sequences[current_id] = "".join(chunks).upper()

    if not sequences:
        raise ValueError(f"No FASTA records found in `{path}`.")

    return sequences


def _load_tree_distance_matrix_biopython(tree_path: str):
    tree = Phylo.read(tree_path, "newick")
    taxa = [clade.name for clade in tree.get_terminals()]
    if any(name is None for name in taxa):
        raise ValueError("All leaf nodes in the backbone tree must have names.")

    distance_matrix = np.zeros((len(taxa), len(taxa)), dtype=np.float32)
    for i, left_name in enumerate(taxa):
        for j in range(i + 1, len(taxa)):
            distance = float(tree.distance(left_name, taxa[j]))
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    id2index = {taxon_id: index for index, taxon_id in enumerate(taxa)}
    return distance_matrix, taxa, id2index


def _ete3_row_chunk(start_index, end_index, taxa, node_map):
    chunk_results = []
    for i in range(start_index, end_index):
        left_node = node_map[taxa[i]]
        row_values = []
        for j in range(i + 1, len(taxa)):
            row_values.append((j, float(left_node.get_distance(node_map[taxa[j]]))))
        chunk_results.append((i, row_values))
    return chunk_results


def _load_tree_distance_matrix_ete3(tree_path: str, num_workers: int = 0):
    from ete3 import Tree

    tree = Tree(tree_path, format=1)
    taxa = [leaf.name for leaf in tree.get_leaves()]
    if any(name is None for name in taxa):
        raise ValueError("All leaf nodes in the backbone tree must have names.")

    node_map = {leaf.name: leaf for leaf in tree.get_leaves()}
    distance_matrix = np.zeros((len(taxa), len(taxa)), dtype=np.float32)
    worker_count = max(1, num_workers or min(os.cpu_count() or 1, 8))

    if len(taxa) <= 100 or worker_count == 1:
        row_chunks = [_ete3_row_chunk(0, len(taxa), taxa, node_map)]
    else:
        chunk_size = max(1, len(taxa) // worker_count)
        ranges = [(start, min(start + chunk_size, len(taxa))) for start in range(0, len(taxa), chunk_size)]
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            row_chunks = list(
                executor.map(lambda item: _ete3_row_chunk(item[0], item[1], taxa, node_map), ranges)
            )

    for chunk in row_chunks:
        for row_index, row_values in chunk:
            for col_index, distance in row_values:
                distance_matrix[row_index, col_index] = distance
                distance_matrix[col_index, row_index] = distance

    id2index = {taxon_id: index for index, taxon_id in enumerate(taxa)}
    return distance_matrix, taxa, id2index


def load_tree_distance_matrix(tree_path: str, backend: str = "auto", num_workers: int = 0):
    backend = backend.lower()
    if backend not in {"auto", "ete3", "biopython"}:
        raise ValueError("`backend` must be one of: auto, ete3, biopython.")

    if backend in {"auto", "ete3"}:
        try:
            return _load_tree_distance_matrix_ete3(tree_path, num_workers=num_workers)
        except ImportError:
            if backend == "ete3":
                raise

    return _load_tree_distance_matrix_biopython(tree_path)


def normalize_split_ids(
    split_file: str,
    valid_aliases: Iterable[str] = ("valid", "val", "validation"),
) -> Dict[str, List[str]]:
    dataframe = pd.read_csv(split_file)
    columns = {column.lower(): column for column in dataframe.columns}

    valid_column = None
    for alias in valid_aliases:
        if alias in columns:
            valid_column = columns[alias]
            break

    if "train" in columns and (valid_column or "test" in columns):

        return {
            "train_id": dataframe[columns["train"]].dropna().astype(str).tolist(),
            "valid_id": dataframe[valid_column].dropna().astype(str).tolist() if valid_column else [],
            "test_id": dataframe[columns["test"]].dropna().astype(str).tolist() if "test" in columns else [],
        }

    if {"id", "split"}.issubset(columns):
        split_ids = {"train_id": [], "valid_id": [], "test_id": []}
        valid_names = set(valid_aliases)
        for _, row in dataframe.iterrows():
            sample_id = str(row[columns["id"]])
            split_name = str(row[columns["split"]]).strip().lower()
            if split_name == "train":
                split_ids["train_id"].append(sample_id)
            elif split_name in valid_names:
                split_ids["valid_id"].append(sample_id)
            elif split_name == "test":
                split_ids["test_id"].append(sample_id)
            else:
                raise ValueError(f"Unsupported split label `{split_name}` in `{split_file}`.")
        return split_ids

    raise ValueError(
        "Split file must contain a `train` column plus optional `valid`/`test` columns, "
        "or an `id/split` table."
    )


def create_random_split(
    all_ids: Iterable[str],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[str]]:
    ratio_sum = train_ratio + valid_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"`train_ratio + valid_ratio + test_ratio` must equal 1.0, got {ratio_sum:.6f}."
        )

    ids = list(all_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    train_end = int(len(ids) * train_ratio)
    valid_end = train_end + int(len(ids) * valid_ratio)
    return {
        "train_id": ids[:train_end],
        "valid_id": ids[train_end:valid_end],
        "test_id": ids[valid_end:],
    }


def validate_split_ids(split_ids: Dict[str, List[str]], available_ids: Iterable[str]) -> None:
    available_set = set(available_ids)
    train_set = set(split_ids["train_id"])
    valid_set = set(split_ids["valid_id"])
    test_set = set(split_ids["test_id"])

    missing = (train_set | valid_set | test_set).difference(available_set)
    if missing:
        preview = sorted(missing)[:10]
        raise ValueError(f"Split file contains IDs missing from the FASTA/tree inputs: {preview}")

    overlap = {
        "train/valid": train_set & valid_set,
        "train/test": train_set & test_set,
        "valid/test": valid_set & test_set,
    }
    duplicated = {key: value for key, value in overlap.items() if value}
    if duplicated:
        summary = {key: sorted(list(value))[:10] for key, value in duplicated.items()}
        raise ValueError(f"Split IDs overlap across splits: {summary}")


def construct_pair_dataframe(
    anchor_ids: Iterable[str],
    sequence_by_id: Dict[str, str],
    positive_ids: Iterable[str],
    distance_matrix: np.ndarray,
    id2index: Dict[str, int],
    include_self_pairs: bool = False,
) -> pd.DataFrame:
    anchor_ids = list(anchor_ids)
    positive_ids = list(positive_ids)
    same_id_order = anchor_ids == positive_ids
    rows = []

    if same_id_order:
        for anchor_index, left_id in enumerate(anchor_ids):
            end_index = anchor_index + 1 if include_self_pairs else anchor_index
            for positive_index in range(0, end_index):
                right_id = positive_ids[positive_index]
                rows.append(
                    {
                        "id1": left_id,
                        "id2": right_id,
                        "seq1": sequence_by_id[left_id],
                        "seq2": sequence_by_id[right_id],
                        "distance": float(distance_matrix[id2index[left_id], id2index[right_id]]),
                    }
                )
    else:
        for left_id in anchor_ids:
            for right_id in positive_ids:
                rows.append(
                    {
                        "id1": left_id,
                        "id2": right_id,
                        "seq1": sequence_by_id[left_id],
                        "seq2": sequence_by_id[right_id],
                        "distance": float(distance_matrix[id2index[left_id], id2index[right_id]]),
                    }
                )

    return pd.DataFrame(rows, columns=["id1", "id2", "seq1", "seq2", "distance"])


def save_stage4_dataset_artifacts(
    output_dir: str,
    split_ids: Dict[str, List[str]],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metadata: Dict[str, object],
) -> None:
    dataset_dir = os.path.join(output_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    for split_name, ids in split_ids.items():
        file_name = split_name.replace("_id", "_ids.txt")
        with open(os.path.join(dataset_dir, file_name), "w", encoding="utf-8") as handle:
            handle.write("\n".join(ids))

    train_df.to_csv(os.path.join(dataset_dir, "train_pairs.csv"), index=False)
    valid_df.to_csv(os.path.join(dataset_dir, "valid_pairs.csv"), index=False)
    test_df.to_csv(os.path.join(dataset_dir, "test_pairs.csv"), index=False)

    with open(os.path.join(dataset_dir, "split_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def matrix_to_lower_triangle(distance_matrix: np.ndarray) -> List[List[float]]:
    matrix = np.asarray(distance_matrix)
    return [[float(matrix[i, j]) for j in range(i + 1)] for i in range(matrix.shape[0])]


def write_distance_csv(distance_matrix: np.ndarray, labels: List[str], output_path: str) -> None:
    dataframe = pd.DataFrame(distance_matrix, index=labels, columns=labels)
    dataframe.to_csv(output_path)


class Stage4DistanceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length: int = 2048):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

        required_columns = {"seq1", "seq2", "distance"}
        missing = required_columns.difference(self.data.columns)
        if missing:
            raise ValueError(f"Missing required stage-4 columns: {sorted(missing)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        return {
            "seq1": str(row["seq1"]),
            "seq2": str(row["seq2"]),
            "labels": np.float32(row["distance"]),
        }


class Stage4DataCollator:
    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        seq1 = [feature["seq1"] for feature in features]
        seq2 = [feature["seq2"] for feature in features]
        labels = torch.tensor([feature["labels"] for feature in features], dtype=torch.float32)
        batch_size = len(features)

        encoding = self.tokenizer(
            seq1 + seq2,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": torch.stack(
                (encoding["input_ids"][:batch_size], encoding["input_ids"][batch_size:]),
                dim=1,
            ),
            "attention_mask": torch.stack(
                (encoding["attention_mask"][:batch_size], encoding["attention_mask"][batch_size:]),
                dim=1,
            ),
            "labels": labels,
        }
