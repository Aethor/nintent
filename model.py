from typing import Optional, List, Tuple

import torch
from transformers import BertModel, BertTokenizer

from tree import IntentTree, Intent, Slot


class SpanEncoder(torch.nn.Module):
    def __init__(self):
        super(SpanEncoder, self).__init__()
        self.bert_encoder = BertModel.from_pretrained("models/spanbert")
        self.hidden_size = self.bert_encoder.config.hidden_size

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        :param seqs: (batch_size, seq_size)
        :return:     (batch_size, seq_size, hidden_size)
        """
        return self.bert_encoder(seqs)[0]


class SpanScorer(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super(SpanScorer, self).__init__()
        self.scorer = torch.nn.Linear(2 * hidden_size, 1)

    def forward(self, span_repr: torch.Tensor) -> torch.Tensor:
        """
        :param span_repr: (batch_size, 2, hidden_size)
        :return:          (batch_size, 1)
        """
        # (batch_size, spans_nb, 2 * hidden_size)
        flat_spans = torch.flatten(span_repr, start_dim=1).float()
        return torch.sigmoid(self.scorer(flat_spans))


class Selector(torch.nn.Module):
    def __init__(self, hidden_size: int, labels_nb: int):
        super().__init__()
        self.selector = torch.nn.Linear(2 * hidden_size, labels_nb)

    def forward(self, span_repr: torch.Tensor) -> torch.Tensor:
        """
        :param span_repr: (batch_size, 2, hidden_size)
        :return:          (batch_size, labels_nb)
        """
        # (batch_size, spans_nb, 2 * hidden_size)
        flat_span = torch.flatten(span_repr, start_dim=1)
        return torch.softmax(self.selector(flat_span), dim=1)


class TreeScorer(torch.nn.Module):
    def __init__(self, tokenizer: BertTokenizer):
        super().__init__()

        self.tokenizer = tokenizer

        self.span_encoder = SpanEncoder()
        self.hidden_size = self.span_encoder.hidden_size

        self.span_scorer = SpanScorer(self.hidden_size)

        self.node_type_selector = Selector(self.hidden_size, IntentTree.node_types_nb())
        self.intent_type_selector = Selector(self.hidden_size, Intent.intent_types_nb())
        self.slot_type_selector = Selector(self.hidden_size, Slot.slot_types_nb())

        # 1 reprensents positive class by convention
        self.is_terminal_selector = Selector(self.hidden_size, 2)

    def forward(self, tree: IntentTree, device: torch.device) -> torch.Tensor:
        """
        :param tree:
        :return: (1)
        """
        # (batch_size, seq_size, hidden_size)
        tokens_repr = self.span_encoder(
            torch.tensor(self.tokenizer.encode(tree.tokens)).unsqueeze(0).to(device)
        )
        # (batch_size, 2, hidden_size)
        span_repr = torch.cat((tokens_repr[:, 0, :], tokens_repr[:, -1, :]), dim=1)

        span_score = self.span_scorer(span_repr).squeeze()

        node_type_score = self.node_type_selector(span_repr)[
            0,
            IntentTree.node_types_idx(
                None if tree.node_type is None else type(tree.node_type)
            ),
        ].squeeze()

        intent_type_score = torch.tensor(0, dtype=torch.float).to(device)
        slot_type_score = torch.tensor(0, dtype=torch.float).to(device)
        if type(tree.node_type) == Intent:
            intent_type_score = self.intent_type_selector(span_repr)[
                0, Intent.stoi(tree.node_type.type)
            ].squeeze()
        elif type(tree.node_type) == Slot:
            slot_type_score = self.slot_type_selector(span_repr)[
                0, Slot.stoi(tree.node_type.type)
            ].squeeze()
        elif tree.node_type is None:
            pass
        else:
            raise Exception(f"unknown node type : {type(tree.node_type)}")

        is_terminal_score = self.is_terminal_selector(span_repr)[
            0, 1 if tree.is_leaf() else 0
        ].squeeze()

        children_score = torch.tensor(0, dtype=torch.float).to(device)
        for child in tree.children:
            children_score += self(child, device)

        return (
            span_score
            + node_type_score
            + intent_type_score
            + slot_type_score
            + is_terminal_score
            + children_score
        )

    # TODO: enforce that level 0 should be an intent
    # TODO: enforce tree correctness ??
    def make_tree(self, tokens: torch.Tensor) -> IntentTree:
        """
        :param tokens: (batch_size, seq_size)
        :note: only works with batch size equals to 1
        """
        # (batch_size, seq_size, hidden_size)
        tokens_repr = self.span_encoder(tokens)
        # (batch_size, 2, hidden_size)
        span_repr = torch.cat((tokens_repr[:, 0, :], tokens_repr[:, -1, :]), dim=1)

        node_type = IntentTree.node_types[
            torch.max(self.node_type_selector(span_repr), 1).indices.item()
        ]

        decoded_tokens = self.tokenizer.decode(tokens)
        if node_type == Intent:
            intent_type = torch.max(
                self.intent_type_selector(span_repr), 1
            ).indices.item()
            cur_tree = IntentTree(decoded_tokens, Intent(intent_type))
        elif node_type == Slot:
            slot_type = torch.max(self.slot_type_selector(span_repr), 1).indices.item()
            cur_tree = IntentTree(decoded_tokens, Slot(slot_type))
        elif node_type is None:
            cur_tree = IntentTree(decoded_tokens, None)

        # Check if the current tree should be terminal
        if (
            torch.max(self.is_terminal_selector(span_repr), 1).indices.item() == 1
            or len(tokens) == 1
        ):
            return cur_tree

        children: List[Tuple[IntentTree]] = []
        for i in range(tokens.shape[1] - 1):
            ltree = self.make_tree(tokens[:, : i + 1])
            rtree = self.make_tree(tokens[:, i + 1 :])
            children.append((ltree, rtree))

        best_children_pair = max(children, key=lambda e: self(e[0]) + self(e[1]))

        cur_tree.add_children_(best_children_pair)

        return cur_tree
