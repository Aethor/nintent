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

    def pad(self, tensors: List[torch.Tensor], **kwargs) -> torch.Tensor:
        """
        :param tensors: list(tensors(var seq_size))(batch_size)
        :return: (batch_size, max_seq_size)
        """
        if len(tensors) == 0:
            raise Exception("empty batch")
        max_seq_size = max([t.shape[0] for t in tensors])
        batch_tensor = torch.zeros(len(tensors), max_seq_size, **kwargs)
        for i, t in enumerate(tensors):
            cur_seq_size = t.shape[0]
            batch_tensor[i, 0:cur_seq_size] = t
            for j in range(max_seq_size - cur_seq_size):
                batch_tensor[i, cur_seq_size + j] = 0
        return batch_tensor

    def batches(
        self, batch_size: int, device: torch.device
    ) -> Generator[Tuple[torch.Tensor, List[IntentTree]], None, None]:
        """
        :param batch_size:
        :param device:
        :return: trees List[IntentTree](batch_size)
        """
        batch_nb = len(self.trees) // batch_size
        for i in range(batch_nb):
            batch_trees = self.trees[i * batch_size : (i + 1) * batch_size]
            yield batch_trees
