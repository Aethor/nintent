from __future__ import annotations
from typing import List, Generator

import torch
from transformers import BertTokenizer

from tree import IntentTree


class Dataset:
    def __init__(self, trees: List[IntentTree], tokenizer: BertTokenizer):
        self.trees = trees
        self.tokenizer = tokenizer

    @classmethod
    def from_file(cls, filename: str, tokenizer: BertTokenizer) -> Dataset:
        dataset = Dataset([], tokenizer)
        with open(filename) as f:
            for line in f:
                dataset.trees.append(IntentTree.from_str(line.split("\t")[-1]))
        return dataset

    def batches_nb(self, batch_size):
        return len(self.trees) // batch_size

    def batches(
        self, batch_size: int
    ) -> Generator[Tuple[torch.Tensor, List[IntentTree]], None, None]:
        """
        :param batch_size:
        :yield: Tuple[
                sentence : torch.Tensor(batch_size, seq_size),
                tree :     List[IntentTree](batch_size)
            ]
        """
        batch_nb = len(self.trees) // batch_size
        for i in range(batch_nb):
            batch_trees = self.trees[i * batch_size : (i + 1) * batch_size]
            yield (
                torch.tensor(
                    [
                        self.tokenizer.encode("[CLS] " + tree.tokens + " [SEP]")
                        for tree in batch_trees
                    ]
                ),
                batch_trees,
            )
