from __future__ import annotations
from typing import Union, Tuple, Optional, List, Type, Mapping, Iterable
from collections import namedtuple

from transformers import BertTokenizer
import torch


class Intent:
    intent_types: Mapping[int, str] = {
        0: "COMBINE",
        1: "GET_EVENT_ATTENDEE_AMOUNT",
        2: "GET_EVENT",
        3: "NEGATION",
        4: "GET_LOCATION_SCHOOL",
        5: "GET_DIRECTIONS",
        6: "GET_ESTIMATED_DEPARTURE",
        7: "GET_LOCATION_HOME",
        8: "GET_ESTIMATED_DURATION",
        9: "GET_INFO_TRAFFIC",
        10: "UNSUPPORTED_EVENT",
        11: "GET_ESTIMATED_ARRIVAL",
        12: "GET_INFO_ROUTE",
        13: "GET_LOCATION_WORK",
        14: "UNSUPPORTED_NAVIGATION",
        15: "GET_EVENT_ATTENDEE",
        16: "UNSUPPORTED",
        17: "GET_LOCATION_HOMETOWN",
        18: "GET_DISTANCE",
        19: "GET_EVENT_ORGANIZER",
        20: "UPDATE_DIRECTIONS",
        21: "UNINTELLIGIBLE",
        22: "GET_LOCATION",
        23: "GET_INFO_ROAD_CONDITION",
        24: "GET_CONTACT",
    }

    def __init__(self, intent_type: Union[str, int]):
        if isinstance(intent_type, int):
            self.type = Intent.itos(intent_type)
        else:
            if not intent_type in {v: k for k, v in Intent.intent_types.items()}:
                raise KeyError(f"Invalid intent type {intent_type}")
            self.type = intent_type

    @classmethod
    def itos(cls, i: int):
        return Intent.intent_types[i]

    @classmethod
    def stoi(cls, s: str):
        return {v: k for k, v in Intent.intent_types.items()}[s]

    @classmethod
    def intent_types_nb(cls) -> int:
        return len(list(cls.intent_types))

    def __eq__(self, other: Intent):
        if not isinstance(other, Intent):
            return False
        return other.type == self.type

    def __str__(self):
        return "INTENT : " + self.type


class Slot:
    slot_types: Mapping[int, str] = {
        0: "WAYPOINT_AVOID",
        1: "COMBINE",
        2: "ROAD_CONDITION_AVOID",
        3: "CONTACT",
        4: "PATH_AVOID",
        5: "TYPE_RELATION",
        6: "LOCATION_CURRENT",
        7: "AMOUNT",
        8: "NAME_EVENT",
        9: "DATE_TIME",
        10: "DATE_TIME_ARRIVAL",
        11: "DATE_TIME_DEPARTURE",
        12: "SEARCH_RADIUS",
        13: "OBSTRUCTION_AVOID",
        14: "POINT_ON_MAP",
        15: "GROUP",
        16: "SOURCE",
        17: "DESTINATION",
        18: "CATEGORY_LOCATION",
        19: "METHOD_TRAVEL",
        20: "ORDINAL",
        21: "OBSTRUCTION",
        22: "CONTACT_RELATED",
        23: "UNIT_DISTANCE",
        24: "ATTRIBUTE_EVENT",
        25: "WAYPOINT",
        26: "LOCATION_USER",
        27: "LOCATION",
        28: "ATTENDEE_EVENT",
        29: "ORGANIZER_EVENT",
        30: "ROAD_CONDITION",
        31: "PATH",
        32: "LOCATION_MODIFIER",
        33: "LOCATION_WORK",
        34: "WAYPOINT_ADDED",
        35: "CATEGORY_EVENT",
    }

    def __init__(self, slot_type: Union[str, int]):
        if isinstance(slot_type, int):
            self.type = Slot.itos(slot_type)
        else:
            if not slot_type in {v: k for k, v in Slot.slot_types.items()}:
                raise KeyError(f"Invalid slot type {slot_type}")
            self.type = slot_type

    @classmethod
    def itos(cls, i: int):
        return Slot.slot_types[i]

    @classmethod
    def stoi(cls, s: str):
        return {v: k for k, v in Slot.slot_types.items()}[s]

    @classmethod
    def slot_types_nb(cls) -> int:
        return len(list(cls.slot_types))

    def __eq__(self, other: Slot):
        if not isinstance(other, Slot):
            return False
        return other.type == self.type

    def __str__(self):
        return "SLOT : " + self.type


class IntentTree:

    node_types: Mapping[int, Optional[Type]] = {0: Intent, 1: Slot}
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    def __init__(
        self,
        tokens: Union[str, List[str]],
        node_type: Optional[Union[Intent, Slot]],
        span_coords: Optional[List[int]] = None,
    ):
        if isinstance(tokens, str):
            self.tokens: List[str] = IntentTree.tokenizer.tokenize(tokens)
        elif isinstance(tokens, list):
            self.tokens = tokens
        else:
            raise Exception(f"can't handle tokens type : {type(tokens)}")
        self.node_type = node_type
        self.span_coords = span_coords
        self.children = []

    def add_child_(self, child: IntentTree):
        self.children.append(child)

    def add_children_(self, children: Iterable[IntentTree]):
        for child in children:
            self.add_child_(child)

    def children_spans(self) -> List[List[int]]:
        return [child.span_coords for child in self.children]

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @classmethod
    def tokens_as_tensor(
        cls, tokens: Union[str, List[str]], device: Optional[torch.device]
    ) -> torch.Tensor:
        if device is None:
            return torch.tensor(IntentTree.tokenizer.encode(tokens))
        return torch.tensor(IntentTree.tokenizer.encode(tokens)).to(device)

    @classmethod
    def from_str(cls, sent: str) -> IntentTree:
        stack: List[IntentTree] = []
        token_idx = 0

        for token in sent.split():

            if token.startswith("["):
                label, typ = token[1:].split(":")
                if label == "IN":
                    stack.append(IntentTree([], Intent(typ), [token_idx, token_idx]))
                elif label == "SL":
                    stack.append(IntentTree([], Slot(typ), [token_idx, token_idx]))
                else:
                    raise Exception(f"Unknown node type : {label}")
                if len(stack) >= 2:
                    stack[-2].add_child_(stack[-1])

            elif token.startswith("]"):
                last_popped = stack.pop()

            else:
                tokens = IntentTree.tokenizer.tokenize(token)
                token_idx += len(tokens)
                for node in stack:
                    node.tokens += tokens
                    node.span_coords[1] += len(tokens)

        return last_popped

    def flat(self) -> List[namedtuple]:
        FlatNode = namedtuple("FlatNode", ["node_type", "tokens", "span_coords"])
        cur_flat_tree = [FlatNode(self.node_type, self.tokens, self.span_coords)]
        for child in self.children:
            cur_flat_tree += child.flat()
        return cur_flat_tree

    def labeled_bracketing_similarity(self, other: IntentTree) -> Tuple[float]:
        """
        :param other: pred tree (self tree is considered as gold)
        :return: precision, recall, F1
        """
        other_flat = other.flat()
        self_flat = self.flat()

        precision_list = list()
        for flat_node in other_flat:
            if flat_node in self_flat:
                precision_list.append(1)
            else:
                precision_list.append(0)
        if len(precision_list) > 0:
            precision = sum(precision_list) / len(precision_list)
        else:
            precision = 0

        recall_list = list()
        for flat_node in self_flat:
            if flat_node in other_flat:
                recall_list.append(1)
            else:
                recall_list.append(0)
        if len(recall_list) > 0:
            recall = sum(recall_list) / len(recall_list)
        else:
            recall = 0

        if precision + recall == 0:
            return precision, recall, 0
        f1 = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1

    @classmethod
    def exact_accuracy_metric(
        cls, pred_trees: List[IntentTree], gold_trees: List[IntentTree]
    ) -> float:
        if len(pred_trees) != len(gold_trees):
            raise Exception
        exact_accuracy_list = list()
        for pred_tree, gold_tree in zip(pred_trees, gold_trees):
            if pred_tree == gold_tree:
                exact_accuracy_list.append(1)
            else:
                exact_accuracy_list.append(0)
        if len(exact_accuracy_list) != 0:
            return sum(exact_accuracy_list) / len(exact_accuracy_list)
        else:
            return 0

    @classmethod
    def labeled_bracketed_metric(
        cls, pred_trees: List[IntentTree], gold_trees: List[IntentTree]
    ) -> Tuple[float]:
        if len(pred_trees) != len(gold_trees):
            raise Exception
        precision_list = list()
        recall_list = list()
        f1_list = list()

        for pred_tree, gold_tree in zip(pred_trees, gold_trees):
            precision, recall, f1 = gold_tree.labeled_bracketing_similarity(pred_tree)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        metrics = list()
        for metric_list in [precision_list, recall_list, f1_list]:
            if len(metric_list) > 0:
                metrics.append(sum(metric_list) / len(metric_list))
            else:
                metric_list.append(0)
        return tuple(metrics)

    def __eq__(self, other: IntentTree) -> bool:
        if not isinstance(other, IntentTree):
            return False
        if len(self.children) != len(other.children):
            return False
        if len(self.children) == 0 and len(other.children) == 0:
            return self.tokens == other.tokens and self.node_type == other.node_type
        for child, other_child in zip(self.children, other.children):
            if child != other_child:
                return False
        return True

    def __str__(
        self, indent: str = "", is_last: bool = True, is_root: bool = True
    ) -> str:
        string = (
            indent
            + (("└──" if is_last else "├──") if not is_root else "")
            + "[{}][{}][{}]\n".format(
                str(self.node_type), str(self.span_coords), " ".join(self.tokens)
            )
        )
        indent += "   " if is_last else "│  "
        for child in self.children:
            string += child.__str__(
                indent=indent, is_last=child is self.children[-1], is_root=False
            )
        return string

    @classmethod
    def node_types_idx(cls, ntype: Type) -> int:
        return {v: k for k, v in cls.node_types.items()}[ntype]

    @classmethod
    def node_types_nb(cls) -> int:
        return len(list(cls.node_types))
