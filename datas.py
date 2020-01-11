from typing import List, Generator, Iterable, Tuple, Optional, Mapping

import torch
from transformers import BertTokenizer
from tqdm import tqdm

from tree import IntentTree, Intent, Slot


class Dataset:
    def __init__(self, trees: List[IntentTree]):
        """
        :param trees: a list of IntentTree
        """
        self.trees = [
            tree for tree in trees if not tree.node_type.type.startswith("UN")
        ]

    @classmethod
    def from_file(cls, filename: str, usage_ratio: Optional[float] = None):
        """
        :return: a Dataset
        """
        trees = list()
        with open(filename) as f:
            for i, _ in enumerate(f):
                pass
            if not usage_ratio is None:
                max_line = int(usage_ratio * i)
            f.seek(0)
            for i, line in enumerate(f):
                if usage_ratio is None or i < max_line:
                    trees.append(IntentTree.from_str(line.split("\t")[-1]))
        return Dataset(trees)

    @classmethod
    def from_files(
        cls, filenames: Iterable[str], usage_ratios: Optional[List[float]] = None
    ) -> Tuple:
        """
        :return: a Tuple of Dataset
        """
        datasets = list()
        if usage_ratios is None:
            for filename in tqdm(filenames):
                datasets.append(Dataset.from_file(filename))
        else:
            for filename, usage_ratio in tqdm(
                zip(filenames, usage_ratios), total=len(filenames)
            ):
                datasets.append(Dataset.from_file(filename, usage_ratio))

        return tuple(datasets)

    def class_weights(self) -> Tuple[List[float]]:
        """
        :return: intent_weights, slot_weights
        """
        intent_occ = [0] * Intent.intent_types_nb()
        slot_occ = [0] * Slot.slot_types_nb()
        for tree in self.trees:
            for flat_node in tree.flat():
                if type(flat_node.node_type) == Intent:
                    intent_occ[Intent.stoi(flat_node.node_type.type)] += 1
                elif type(flat_node.node_type) == Slot:
                    slot_occ[Slot.stoi(flat_node.node_type.type)] += 1
        max_intent_occ = max(intent_occ)
        max_slot_occ = max(slot_occ)
        intent_weights = [
            (max_intent_occ / occ) if occ != 0 else 1 for occ in intent_occ
        ]
        slot_weights = [(max_slot_occ / occ) if occ != 0 else 1 for occ in slot_occ]
        return intent_weights, slot_weights

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
        batch_nb = self.batches_nb(batch_size)
        for i in range(batch_nb):
            batch_trees = self.trees[i * batch_size : (i + 1) * batch_size]
            yield batch_trees
