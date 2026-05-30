import random

from torch.utils.data import Dataset


class LucaContrastiveDataset(Dataset):
    """Minimal dataset wrapper for stage-2 and stage-3 contrastive training."""

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length=2048,
        stage=2,
        col_anchor="seq",
        col_positive="domain",
        col_label="label",
    ):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stage = stage
        self.col_anchor = col_anchor
        self.col_positive = col_positive
        self.col_label = col_label

        required_columns = {self.col_anchor}
        if self.stage == 2:
            required_columns.add(self.col_label)
        elif self.stage == 3:
            required_columns.add(self.col_positive)
        else:
            raise ValueError("`stage` must be 2 or 3.")

        missing = required_columns.difference(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if self.stage == 2:
            grouped = self.data.groupby(self.col_label)[self.col_anchor].apply(list)
            self.label_to_sequences = grouped.to_dict()

    def __len__(self):
        return len(self.data)

    def _sample_positive(self, anchor_sequence, label):
        candidates = self.label_to_sequences[label]
        if len(candidates) == 1:
            return candidates[0]

        positive_sequence = random.choice(candidates)
        while positive_sequence == anchor_sequence and len(candidates) > 1:
            positive_sequence = random.choice(candidates)
        return positive_sequence

    def __getitem__(self, index):
        row = self.data.iloc[index]
        anchor_sequence = row[self.col_anchor]

        if self.stage == 2:
            positive_sequence = self._sample_positive(
                anchor_sequence=anchor_sequence,
                label=row[self.col_label],
            )
        else:
            positive_sequence = row[self.col_positive]

        encoding = self.tokenizer(
            [anchor_sequence, positive_sequence],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }


class LucaLegacyStage2Dataset(Dataset):
    """Stage-2 dataset that mirrors the original class-balanced dynamic batching."""

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length=2048,
        col_anchor="seq",
        col_label="label",
        pairs_per_item=4,
        items_per_label=10,
        is_eval=False,
    ):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.col_anchor = col_anchor
        self.col_label = col_label
        self.pairs_per_item = pairs_per_item
        self.items_per_label = items_per_label
        self.is_eval = is_eval

        required_columns = {self.col_anchor, self.col_label}
        missing = required_columns.difference(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        grouped = self.data.groupby(self.col_label)[self.col_anchor].apply(list)
        self.class_samples = grouped.to_dict()
        self.valid_labels = [
            label for label, sequences in self.class_samples.items() if len(sequences) >= 2
        ]
        if not self.valid_labels:
            raise ValueError("Stage-2 legacy sampling requires at least one label with 2 sequences.")

        if not self.is_eval and len(self.valid_labels) < self.pairs_per_item:
            raise ValueError(
                "Stage-2 legacy sampling requires at least "
                f"{self.pairs_per_item} labels with 2 sequences each, "
                f"but found {len(self.valid_labels)}."
            )

    def __len__(self):
        if self.is_eval:
            return len(self.valid_labels)
        return len(self.valid_labels) * self.items_per_label

    def __getitem__(self, index):
        if self.is_eval:
            label = self.valid_labels[index % len(self.valid_labels)]
            texts = random.sample(self.class_samples[label], 2)
            pair_count = 1
        else:
            selected_labels = random.sample(self.valid_labels, self.pairs_per_item)
            texts = []
            for label in selected_labels:
                texts.extend(random.sample(self.class_samples[label], 2))
            pair_count = self.pairs_per_item

        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].view(pair_count, 2, -1),
            "attention_mask": encoding["attention_mask"].view(pair_count, 2, -1),
        }
