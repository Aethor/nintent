from typing import Optional, List, Tuple, Union
import copy

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

        self.is_slot_selector = Selector(self.hidden_size, 2)
        self.is_intent_selector = Selector(self.hidden_size, 2)
        self.intent_type_selector = Selector(self.hidden_size, Intent.intent_types_nb())
        self.slot_type_selector = Selector(self.hidden_size, Slot.slot_types_nb())

    def forward(self, gold_tree: IntentTree, device: torch.device,) -> torch.Tensor:
        """
        :param tokens: (batch_size(1), seq_size)
        :param tokens_repr: (batch_size(1), seq_size, hidden_size)
        :return: (intent tree, loss)
        """
        tokens = (
            torch.tensor(IntentTree.tokenizer.encode(gold_tree.tokens))
            .unsqueeze(0)
            .to(device)
        )
        # (batch_size, seq_size, hidden_size)
        tokens_repr = self.span_encoder(tokens)[:, 1:-1, :]

        loss = torch.tensor([0], dtype=torch.float).to(device)

        if type(gold_tree.node_type) == Intent:

            for span_size in range(tokens_repr.shape[1] - 1, 0, -1):
                for span_start in range(0, tokens_repr.shape[1] - span_size + 1):

                    span_end = span_start + span_size - 1

                    span_repr = torch.cat(
                        (tokens_repr[:, span_start, :], tokens_repr[:, span_end, :]),
                        dim=1,
                    )
                    is_slot_pred = self.is_slot_selector(span_repr)[0]
                    is_slot = torch.max(is_slot_pred, 0).indices.item() == 1

                if (span_start, span_end) in gold_tree.children_spans():
                    loss += is_slot_pred[0]
                else:
                    loss += is_slot_pred[1]

            for child in gold_tree.children:
                loss += self(child, device,)

        elif type(gold_tree.node_type) == Slot:
            span_repr = torch.cat((tokens_repr[:, 0, :], tokens_repr[:, -1, :]), dim=1,)
            is_intent_pred = self.is_intent_selector(span_repr)[0]
            is_intent = torch.max(is_intent_pred, 0).indices.item() == 1
            if gold_tree.is_leaf():
                loss += is_intent_pred[0]
            else:
                loss += is_intent_pred[1]

            # FIXME:
            assert len(gold_tree.children) <= 1

            if gold_tree.is_leaf():
                return loss

            intent_node = gold_tree.children[0]
            loss += self(intent_node, device,)

        return loss

