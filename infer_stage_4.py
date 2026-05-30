import argparse
import hashlib
import json
import logging
import os
from types import SimpleNamespace

import numpy as np
import torch
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, _DistanceMatrix
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from src.stage4_data import (
    load_tree_distance_matrix,
    matrix_to_lower_triangle,
    parse_fasta,
    write_distance_csv,
)
from src.stage4_model import LucaPhyloHyper

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-4 inference with a stage-2/3 base model and LoRA adapter.")
    parser.add_argument("--model_name_or_path", required=True, help="Stage-2 or stage-3 checkpoint used as the base model.")
    parser.add_argument("--stage4_adapter_path", required=False, help="Stage-4 LoRA adapter directory.")
    parser.add_argument("--backbone_tree", required=True, help="Backbone tree in Newick format.")
    parser.add_argument("--backbone_seqs", required=True, help="Backbone FASTA file.")
    parser.add_argument("--query_seqs", required=True, help="Query FASTA file to place onto the backbone.")
    parser.add_argument("--output_dir", required=True, help="Directory for distance matrices and optional NJ trees.")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--pooler_type", default="avg")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--distance_batch_size", type=int, default=32)
    parser.add_argument("--tree_distance_backend", default="auto", choices=["auto", "ete3", "biopython"])
    parser.add_argument("--tree_distance_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache_dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--embedding_cache_dir", default=None, help="Optional on-disk cache for sequence embeddings.")
    parser.add_argument("--save_nj_tree", action="store_true", help="Also write Neighbor-Joining trees from the inferred distances.")
    return parser.parse_args()


def ensure_context_length(config, max_seq_length: int) -> None:
    max_positions = getattr(config, "max_position_embeddings", None)
    if max_positions is None:
        return

    required_positions = max_seq_length + 2
    if required_positions > max_positions:
        raise ValueError(
            f"Requested max_seq_length={max_seq_length}, but the base checkpoint only supports "
            f"{max_positions - 2}. Please use a stage-2/stage-3 checkpoint trained at the target context length."
        )


def load_model(args):
    logger.info("Loading stage-4 base model from `%s`.", args.model_name_or_path)
    logger.info(
        "The base checkpoint does not contain the stage-4 hyperbolic head, so newly initialized "
        "`scale.*` weights are expected before any LoRA adapter is loaded."
    )
    config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    ensure_context_length(config, args.max_seq_length)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        truncation_side="left",
    )
    model_args = SimpleNamespace(pooler_type=args.pooler_type)
    model = LucaPhyloHyper.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir,
        model_args=model_args,
    )

    if args.stage4_adapter_path:
        logger.info("Loading stage-4 LoRA adapter from `%s`.", args.stage4_adapter_path)
        model = PeftModel.from_pretrained(model, args.stage4_adapter_path)
    else:
        logger.warning(
            "No `--stage4_adapter_path` was provided. Inference will use the randomly initialized "
            "stage-4 head from the base model, which is usually not meaningful."
        )

    model = model.to(args.device)
    model.eval()
    return model, tokenizer


def get_stage4_module(model):
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def log_stage4_parameter_status(model, adapter_path: str | None) -> None:
    stage4_model = get_stage4_module(model)
    scale = getattr(stage4_model, "scale", None)
    if scale is None:
        logger.warning("Could not find the stage-4 `scale` module after model loading.")
        return

    curvature = getattr(scale, "curvature", None)
    projection = getattr(getattr(scale, "embedding", None), "projection", None)

    payload = {
        "adapter_loaded": bool(adapter_path),
        "stage4_adapter_path": adapter_path,
    }
    if curvature is not None:
        payload["scale.curvature"] = float(curvature.detach().float().view(-1)[0].item())
    if projection is not None:
        payload["scale.embedding.projection.weight_shape"] = list(projection.weight.shape)
        payload["scale.embedding.projection.weight_norm"] = float(projection.weight.detach().float().norm().item())
        if projection.bias is not None:
            payload["scale.embedding.projection.bias_norm"] = float(
                projection.bias.detach().float().norm().item()
            )

    logger.info("Active stage-4 parameter status: %s", json.dumps(payload, ensure_ascii=False))


def get_embedding_cache_root(args):
    cache_root = args.embedding_cache_dir or os.path.join(args.output_dir, "embedding_cache")
    os.makedirs(cache_root, exist_ok=True)
    return cache_root


def embedding_cache_key(model_fingerprint: str, record_id: str, sequence: str, max_seq_length: int) -> str:
    payload = f"{model_fingerprint}\n{record_id}\n{sequence[-max_seq_length:]}\n{max_seq_length}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def encode_sequences_with_cache(
    sequences,
    model,
    tokenizer,
    device,
    batch_size,
    max_seq_length,
    cache_dir,
    model_fingerprint,
):
    os.makedirs(cache_dir, exist_ok=True)
    stage4_model = get_stage4_module(model)
    record_ids = list(sequences.keys())
    embeddings = [None] * len(record_ids)
    pending = []

    for index, record_id in enumerate(record_ids):
        sequence = sequences[record_id]
        cache_name = embedding_cache_key(model_fingerprint, record_id, sequence, max_seq_length)
        cache_path = os.path.join(cache_dir, f"{cache_name}.pt")
        if os.path.exists(cache_path):
            embeddings[index] = torch.load(cache_path, map_location="cpu")
        else:
            pending.append((index, sequence, cache_path))

    for start in tqdm(range(0, len(pending), batch_size), desc="Encoding sequences"):
        batch = pending[start : start + batch_size]
        batch_sequences = [sequence for _, sequence, _ in batch]
        features = tokenizer(
            batch_sequences,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        features = {name: value.to(device) for name, value in features.items()}
        with torch.no_grad():
            pooled = stage4_model.encode_sequences(
                input_ids=features["input_ids"],
                attention_mask=features["attention_mask"],
            ).cpu()

        for batch_index, (target_index, _, cache_path) in enumerate(batch):
            embedding = pooled[batch_index]
            torch.save(embedding, cache_path)
            embeddings[target_index] = embedding

    if any(embedding is None for embedding in embeddings):
        raise RuntimeError("Failed to compute all sequence embeddings for stage-4 inference.")

    return torch.stack(embeddings), record_ids


def compute_pairwise_distances(stage4_model, left_embeddings, right_embeddings, device, batch_size):
    rows = []
    right_embeddings = right_embeddings.to(device)
    with torch.no_grad():
        for start in tqdm(range(0, left_embeddings.size(0), batch_size), desc="Computing distances"):
            left_batch = left_embeddings[start : start + batch_size].to(device)
            distances = stage4_model.predict_distance_from_embeddings(
                left_batch.unsqueeze(1),
                right_embeddings.unsqueeze(0),
            )
            rows.append(distances.cpu())
    return torch.cat(rows, dim=0).numpy()


def save_nj_tree(distance_matrix: np.ndarray, labels, output_path: str) -> None:
    lower_triangle = matrix_to_lower_triangle(distance_matrix)
    dm = _DistanceMatrix(names=list(labels), matrix=lower_triangle)
    tree = DistanceTreeConstructor().nj(dm)
    for clade in tree.find_clades():
        if getattr(clade, "branch_length", None) is not None and clade.branch_length < 0:
            clade.branch_length = 0.0
    Phylo.write(tree, output_path, "newick")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    model, tokenizer = load_model(args)
    stage4_model = get_stage4_module(model)
    log_stage4_parameter_status(model, args.stage4_adapter_path)

    backbone_sequences = parse_fasta(args.backbone_seqs)
    query_sequences = parse_fasta(args.query_seqs)
    backbone_distances, backbone_ids, backbone_id2index = load_tree_distance_matrix(
        args.backbone_tree,
        backend=args.tree_distance_backend,
        num_workers=args.tree_distance_workers,
    )

    missing_backbone_ids = [record_id for record_id in backbone_ids if record_id not in backbone_sequences]
    if missing_backbone_ids:
        raise ValueError(f"Backbone FASTA is missing IDs present in the backbone tree: {missing_backbone_ids[:10]}")

    cache_root = get_embedding_cache_root(args)
    model_fingerprint = f"{os.path.abspath(args.model_name_or_path)}|{os.path.abspath(args.stage4_adapter_path or '')}"
    backbone_embeddings, backbone_order = encode_sequences_with_cache(
        backbone_sequences,
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        cache_dir=os.path.join(cache_root, "backbone"),
        model_fingerprint=model_fingerprint,
    )
    query_embeddings, query_order = encode_sequences_with_cache(
        query_sequences,
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        cache_dir=os.path.join(cache_root, "query"),
        model_fingerprint=model_fingerprint,
    )

    query_backbone_distances = compute_pairwise_distances(
        stage4_model=stage4_model,
        left_embeddings=query_embeddings,
        right_embeddings=backbone_embeddings,
        device=args.device,
        batch_size=args.distance_batch_size,
    )
    query_query_distances = compute_pairwise_distances(
        stage4_model=stage4_model,
        left_embeddings=query_embeddings,
        right_embeddings=query_embeddings,
        device=args.device,
        batch_size=args.distance_batch_size,
    )

    backbone_count = len(backbone_order)
    query_count = len(query_order)
    all_labels = backbone_order + query_order
    full_distance_matrix = np.zeros((backbone_count + query_count, backbone_count + query_count), dtype=np.float32)

    for i, left_id in enumerate(backbone_order):
        for j, right_id in enumerate(backbone_order):
            full_distance_matrix[i, j] = backbone_distances[backbone_id2index[left_id], backbone_id2index[right_id]]

    full_distance_matrix[backbone_count:, :backbone_count] = query_backbone_distances
    full_distance_matrix[:backbone_count, backbone_count:] = query_backbone_distances.T
    full_distance_matrix[backbone_count:, backbone_count:] = query_query_distances
    np.fill_diagonal(full_distance_matrix, 0.0)
    np.fill_diagonal(query_query_distances, 0.0)

    full_matrix_path = os.path.join(args.output_dir, "stage4_full_distance_matrix.csv")
    query_matrix_path = os.path.join(args.output_dir, "stage4_query_distance_matrix.csv")
    write_distance_csv(full_distance_matrix, all_labels, full_matrix_path)
    write_distance_csv(query_query_distances, query_order, query_matrix_path)

    if args.save_nj_tree:
        save_nj_tree(
            full_distance_matrix,
            all_labels,
            os.path.join(args.output_dir, "stage4_full_distance_matrix.nwk"),
        )
        save_nj_tree(
            query_query_distances,
            query_order,
            os.path.join(args.output_dir, "stage4_query_distance_matrix.nwk"),
        )

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "stage4_adapter_path": args.stage4_adapter_path,
        "backbone_tree": args.backbone_tree,
        "backbone_seqs": args.backbone_seqs,
        "query_seqs": args.query_seqs,
        "max_seq_length": args.max_seq_length,
        "tree_distance_backend": args.tree_distance_backend,
        "tree_distance_workers": args.tree_distance_workers,
        "backbone_count": backbone_count,
        "query_count": query_count,
        "output_files": {
            "full_distance_matrix": full_matrix_path,
            "query_distance_matrix": query_matrix_path,
        },
    }
    with open(os.path.join(args.output_dir, "stage4_infer_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
