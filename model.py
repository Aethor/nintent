from typing import Optional, List, Tuple, Union, Type
import copy

import torch
from torch.nn.functional import binary_cross_entropy, cross_entropy
from transformers import BertModel, BertTokenizer

from tree import IntentTree, Intent, Slot


def are_spans_overlapping(span1, span2):
    return (
        (span1[0] >= span2[0] and span1[0] < span2[1])
        or (span1[1] > span2[0] and span1[1] <= span2[1])
        or (span1[0] <= span2[0] and span1[1] >= span2[1])
    )


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


class TreeMaker(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.span_encoder = SpanEncoder()
        self.hidden_size = self.span_encoder.hidden_size

        self.is_slot_selectors = list()
        for _ in Intent.intent_types:
            self.is_slot_selectors.append(Selector(self.hidden_size, 2))

        self.is_intent_selector = Selector(self.hidden_size, 2)

        self.intent_type_selector = Selector(self.hidden_size, Intent.intent_types_nb())
        self.intent_type_loss = torch.nn.CrossEntropyLoss()

        self.slot_type_selectors = list()
        for _ in range(Intent.intent_types_nb()):
            self.slot_type_selectors.append(
                Selector(self.hidden_size, Slot.slot_types_nb())
            )
        self.slot_type_loss = torch.nn.CrossEntropyLoss()

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
        span_coords: List[int],
        coords_offset: int = 0,
        cur_intent: Optional[Intent] = None,
    ) -> IntentTree:
        """
        """
        tokens = (
            torch.tensor(IntentTree.tokenizer.encode(tokens_str))
            .unsqueeze(0)
            .to(device)
        )
        tokens_repr = self.span_encoder(tokens)[
            :, span_coords[0] + 1 : span_coords[1] + 1, :
        ]

        if cur_type == Intent:
            intent_type = Intent.itos(
                torch.max(
                    self.intent_type_selector(self.span_repr(tokens_repr)), 1
                ).indices.item()
            )
            cur_tree = IntentTree(
                tokens_str[span_coords[0] : span_coords[1]],
                Intent(intent_type),
                [span_coords[0], span_coords[1]],
            )

            selected_spans: List[Tuple[Tuple[int], float]] = list()
            for span_size in range(1, tokens_repr.shape[1]):
                for span_start in range(0, tokens_repr.shape[1] - span_size + 1):
                    span_end = span_start + span_size

                    span_repr = self.span_repr(tokens_repr[:, span_start:span_end, :])
                    is_slot_pred = self.is_slot_selectors[Intent.stoi(intent_type)](
                        span_repr
                    )[0]
                    is_slot = torch.max(is_slot_pred, 0).indices.item() == 1

                    if is_slot:
                        overlapping_spans = list()
                        for span_coords, span_score in selected_spans:
                            if are_spans_overlapping(
                                (span_start, span_end), span_coords
                            ):
                                overlapping_spans.append((span_coords, span_score))
                        if len(overlapping_spans) > 0:
                            h_mean = len(overlapping_spans) / sum(
                                [1 / s[1] for s in overlapping_spans]
                            )
                        else:
                            h_mean = 0
                        if is_slot_pred[1] > h_mean:
                            for overlapping_span in overlapping_spans:
                                selected_spans.remove(overlapping_span)
                            selected_spans.append(
                                ((span_start, span_end), is_slot_pred[1])
                            )

            for selected_span in selected_spans:
                cur_tree.add_child_(
                    self.make_tree(
                        # tokens_str[selected_span[0][0] : selected_span[0][1]],
                        tokens_str,
                        device,
                        Slot,
                        [selected_span[0][0], selected_span[0][1]],
                        coords_offset + selected_span[0][0],
                        intent_type,
                    )
                )

        elif cur_type == Slot:
            slot_type = Slot.itos(
                torch.max(
                    self.slot_type_selectors[Intent.stoi(cur_intent)](
                        self.span_repr(tokens_repr)
                    ),
                    1,
                ).indices.item()
            )
            cur_tree = IntentTree(
                tokens_str[span_coords[0] : span_coords[1]],
                Slot(slot_type),
                [span_coords[0], span_coords[1]],
            )
            is_intent_pred = self.is_intent_selector(self.span_repr(tokens_repr))[0]
            is_intent = torch.max(is_intent_pred, 0).indices.item() == 1

            if is_intent:
                cur_tree.add_child_(
                    self.make_tree(
                        tokens_str, device, Intent, span_coords, coords_offset
                    )
                )

        return cur_tree

    def forward(
        self,
        gold_tree: IntentTree,
        device: torch.device,
        top_tokens: List[str],
        span_coords: List[int],
        cur_intent: Optional[Intent] = None,
    ) -> torch.Tensor:
        """
        :return: (intent tree, loss)
        """
        tokens = (
            torch.tensor(IntentTree.tokenizer.encode(top_tokens))
            .unsqueeze(0)
            .to(device)
        )
        # (batch_size, seq_size, hidden_size)
        tokens_repr = self.span_encoder(tokens)[
            :, span_coords[0] + 1 : span_coords[1] + 1, :
        ]

        loss = torch.tensor([0], dtype=torch.float).to(device)

        if type(gold_tree.node_type) == Intent:

            intent_type_pred = self.intent_type_selector(self.span_repr(tokens_repr))
            gold_tree_intent_idx = torch.tensor(
                [Intent.stoi(gold_tree.node_type.type)]
            ).to(device)
            loss += self.intent_type_loss(intent_type_pred, gold_tree_intent_idx)

            sentence_len = tokens_repr.shape[1]
            gold_spans = gold_tree.children_spans()
            if len(gold_spans) > 0:
                span_weight = torch.tensor(
                    [(sentence_len * (sentence_len - 1) / 2) / len(gold_spans), 1],
                    dtype=torch.float,
                ).to(device)
            else:
                span_weight = torch.tensor([1, 1], dtype=torch.float).to(device)
            # span_weight = torch.tensor([1, 1], dtype=torch.float).to(device)

            for span_size in range(sentence_len - 1, 0, -1):
                for span_start in range(0, sentence_len - span_size + 1):
                    span_end = span_start + span_size

                    span_repr = self.span_repr(tokens_repr[:, span_start:span_end, :])
                    is_slot_pred = self.is_slot_selectors[gold_tree_intent_idx.item()](
                        span_repr
                    )

                    gold_pred = torch.tensor(
                        [1] if [span_start, span_end] in gold_spans else [0]
                    ).to(device)

                    loss += cross_entropy(is_slot_pred, gold_pred, weight=span_weight)

            for child in gold_tree.children:
                loss += self(
                    child, device, top_tokens, child.span_coords, gold_tree.node_type
                )

        elif type(gold_tree.node_type) == Slot:

            slot_type_pred = self.slot_type_selectors[Intent.stoi(cur_intent.type)](
                self.span_repr(tokens_repr)
            )
            gold_tree_slot_idx = torch.tensor([Slot.stoi(gold_tree.node_type.type)]).to(
                device
            )
            loss += self.slot_type_loss(slot_type_pred, gold_tree_slot_idx)

            is_intent_pred = self.is_intent_selector(self.span_repr(tokens_repr))
            is_intent = torch.max(is_intent_pred[0], 0).indices.item() == 1
            if gold_tree.is_leaf():
                target = torch.tensor([0]).to(device)
                loss += cross_entropy(is_intent_pred, target)
            else:
                target = torch.tensor([1]).to(device)
                loss += cross_entropy(is_intent_pred, target)

            if gold_tree.is_leaf():
                return loss

            intent_node = gold_tree.children[0]
            loss += self(intent_node, device, top_tokens, intent_node.span_coords)

        return loss

    def to(self, device: torch.device):
        super().to(device)
        for is_slot_selector in self.is_slot_selectors:
            is_slot_selector = is_slot_selector.to(device)
        for slot_type_selector in self.slot_type_selectors:
            slot_type_selector = slot_type_selector.to(device)
