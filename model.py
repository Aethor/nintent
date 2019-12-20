from typing import Optional

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

        self.node_type_selector = Selector(self.hidden_size, IntentTree.node_types_nb())
        self.intent_type_selector = Selector(self.hidden_size, Intent.intent_types_nb())
        self.slot_type_selector = Selector(self.hidden_size, Slot.slot_types_nb())

        # 1 reprensents positive class by convention
        self.is_terminal_selector = Selector(self.hidden_size, 2)

    def forward(self, tree: IntentTree) -> torch.Tensor:
        """
        :param tree:
        :return: (1)
        """
        # (batch_size, seq_size, hidden_size)
        tokens_repr = self.span_encoder(
            torch.tensor(self.tokenizer.encode(tree.tokens)).unsqueeze(0)
        )
        # (batch_size, 2, hidden_size)
        span_repr = torch.cat((tokens_repr[:, 0, :], tokens_repr[:, -1, :]), dim=1)

        span_score = self.span_scorer(span_repr)

        node_type_score = self.node_type_selector(span_repr)[
            IntentTree.node_types_idx(type(tree.node_type))
        ]

        if type(tree.node_type) == Intent:
            intent_type_score = self.intent_type_selector(span_repr)[
                Intent.stoi(tree.node_type)
            ]
            slot_type_score = torch.tensor(0)
        elif type(tree.node_type) == Slot:
            slot_type_score = self.slot_type_selector(span_repr)[
                Slot.stoi(tree.node_type)
            ]
            intent_type_score = torch.tensor(0)

        is_terminal_score = self.is_terminal_selector(span_repr)[
            1 if tree.is_leaf() else 0
        ]

        children_score = torch.tensor(0)
        for child in tree.children:
            children_score += self(child)

        return (
            span_score
            + node_type_score
            + intent_type_score
            + slot_type_score
            + is_terminal_score
            + children_score
        )

    def make_tree_(self, tree: IntentTree):
        tokens_repr = self.span_encoder(
            torch.tensor(self.tokenizer.encode(tree.tokens)).unsqueeze(0)
        )
        # TODO:
