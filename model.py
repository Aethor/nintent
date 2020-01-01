from typing import Optional, List, Tuple, Union, Type
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

    def span_repr(self, span: torch.Tensor) -> torch.Tensor:
        """
        :param span: (batch_size, seq_size, hidden_size)
        :param coords: (batch_size, 2)
        :return: (batch_size, 2, hidden_size)
        """
        return torch.cat((span[:, 0, :], span[:, -1, :]), dim=1)

    def make_tree(
        self,
        tokens_str: List[str],
        device: torch.device,
        cur_type: Type,
        coords_offset: int = 0,
    ) -> IntentTree:
        """
        """
        tokens = (
            torch.tensor(IntentTree.tokenizer.encode(tokens_str))
            .unsqueeze(0)
            .to(device)
        )
        tokens_repr = self.span_encoder(tokens)[:, 1:-1, :]

        if cur_type == Intent:
            intent_type = Intent.itos(
                torch.max(
                    self.intent_type_selector(self.span_repr(tokens_repr)), 1
                ).indices.item()
            )
            cur_tree = IntentTree(
                tokens_str,
                Intent(intent_type),
                (0 + coords_offset, coords_offset + len(tokens_str)),
            )

            for span_size in range(tokens_repr.shape[1] - 1, 0, -1):
                for span_start in range(0, tokens_repr.shape[1] - span_size + 1):
                    span_end = span_start + span_size

                    is_overlapping = False
                    for child_span in cur_tree.children_spans():
                        if span_start >= child_span[0] and span_start < child_span[1]:
                            is_overlapping = True
                        if span_end > child_span[0] and span_end <= child_span[1]:
                            is_overlapping = True
                    if is_overlapping:
                        continue

                    span_repr = self.span_repr(tokens_repr[:, span_start:span_end, :])
                    is_slot_pred = self.is_slot_selector(span_repr)[0]
                    is_slot = torch.max(is_slot_pred, 0).indices.item() == 1

                    if is_slot:
                        cur_tree.add_child_(
                            self.make_tree(
                                tokens_str[span_start:span_end],
                                device,
                                Slot,
                                coords_offset + span_start,
                            )
                        )

        elif cur_type == Slot:
            slot_type = Slot.itos(
                torch.max(
                    self.slot_type_selector(self.span_repr(tokens_repr)), 1
                ).indices.item()
            )
            cur_tree = IntentTree(
                tokens_str,
                Slot(slot_type),
                (0 + coords_offset, coords_offset + len(tokens_str)),
            )
            is_intent_pred = self.is_intent_selector(self.span_repr(tokens_repr))[0]
            is_intent = torch.max(is_intent_pred, 0).indices.item() == 1

            if is_intent:
                cur_tree.add_child_(
                    self.make_tree(tokens_str, device, Intent, coords_offset)
                )

        return cur_tree

    def forward(self, gold_tree: IntentTree, device: torch.device) -> torch.Tensor:
        """
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

            intent_type_pred = self.intent_type_selector(self.span_repr(tokens_repr))[0]
            intent_type_idx = torch.max(intent_type_pred, 0).indices.item()
            gold_tree_intent_idx = Intent.stoi(gold_tree.node_type.type)
            loss += (
                intent_type_pred[intent_type_idx]
                - intent_type_pred[gold_tree_intent_idx]
            )

            candidate_span_nb = sum(range(2, tokens_repr.shape[1] + 1))
            span_loss = torch.tensor([0], dtype=torch.float).to(device)
            for span_size in range(tokens_repr.shape[1] - 1, 0, -1):
                for span_start in range(0, tokens_repr.shape[1] - span_size + 1):
                    span_end = span_start + span_size

                    span_repr = self.span_repr(tokens_repr[:, span_start:span_end, :])
                    is_slot_pred = self.is_slot_selector(span_repr)[0]
                    is_slot = torch.max(is_slot_pred, 0).indices.item() == 1

                if [span_start, span_end] in gold_tree.children_spans():
                    # FIXME: experimental weight
                    negative_weight = candidate_span_nb / len(
                        gold_tree.children_spans()
                    )
                    span_loss += negative_weight * is_slot_pred[0]
                else:
                    span_loss += is_slot_pred[1]
            if candidate_span_nb > 0:
                loss += span_loss / candidate_span_nb

            for child in gold_tree.children:
                loss += self(child, device,)

        elif type(gold_tree.node_type) == Slot:

            slot_type_pred = self.slot_type_selector(self.span_repr(tokens_repr))[0]
            slot_type_idx = torch.max(slot_type_pred, 0).indices.item()
            gold_tree_slot_idx = Slot.stoi(gold_tree.node_type.type)
            loss += slot_type_pred[slot_type_idx] - slot_type_pred[gold_tree_slot_idx]

            is_intent_pred = self.is_intent_selector(self.span_repr(tokens_repr))[0]
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
            loss += self(intent_node, device)

        return loss

