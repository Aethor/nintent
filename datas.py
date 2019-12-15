from __future__ import annotations
from typing import List, Generator

from tree import IntentTree


class Dataset:
    def __init__(self, trees: List[IntentTree]):
        self.trees = trees

    @classmethod
    def from_file(cls, filename: str) -> Dataset:
        dataset = Dataset([])
        with open(filename) as f:
            for line in f:
                dataset.trees.append(IntentTree.from_str(line.split("\t")[-1]))
        return dataset

    def batches(self, batch_size: int) -> Generator[List[IntentTree], None, None]:
        batch_nb = len(self.trees) // batch_size
        for i in range(batch_nb):
            yield self.trees[i * batch_size : (i + 1) * batch_size]
