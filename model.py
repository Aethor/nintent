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
        return self.bert_encoder(seqs)


class SpanScorer(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super(SpanScorer, self).__init__()
        self.scorer = torch.nn.Linear(2 * hidden_size, 1)

    def forward(self, spans_repr: torch.Tensor) -> torch.Tensor:
        """
        :param spans_repr: (batch_size, spans_nb, 2, hidden_size)
        :return:           (batch_size, spans_nb)
        """
        # (batch_size, spans_nb, 2 * hidden_size)
        flat_spans = torch.flatten(spans_repr, start_dim=2)

        return torch.softmax(self.scorer(flat_spans).squeeze(), dim=1)


class LabelSelector(torch.nn.Module):
    def __init__(self, hidden_size: int, labels_nb: int):
        super(LabelSelector, self).__init__()
        self.selector = torch.nn.Linear(2 * hidden_size, labels_nb)

    def forward(self, spans_repr: torch.Tensor) -> torch.Tensor:
        """
        :param spans_repr: (batch_size, 2, hidden_size)
        :return:           (batch_size, labels_nb)
        """
        # (batch_size, spans_nb, 2 * hidden_size)
        flat_spans = torch.flatten(spans_repr, start_dim=1)
        return torch.softmax(self.selector(flat_spans), dim=1)


class TreeMaker(torch.nn.Module):
    def __init__(self):
        super(TreeMaker, self).__init__()
        self.span_encoder = SpanEncoder()
        self.span_scorer = SpanScorer(self.span_encoder.hidden_size)
        self.intent_selector = LabelSelector(
            self.span_encoder.hidden_size, len(list(Intent.intent_types))
        )
        self.slot_selector = LabelSelector(
            self.span_encoder.hidden_size, len(list(Slot.slot_types))
        )

    def forward(
        self, tokens: torch.Tensor, tokenizer: BertTokenizer
    ) -> (IntentTree, float):
        """
        :param tokens: (batch_size?, seq_size)
        """
        if tokens.shape[1] == 0:
            raise Exception(
                f"[error] len(tokens) must be > 0 (current : {len(tokens)})"
            )

        # (batch_size?, seq_size, hidden_size)
        # TODO: optimize by not calling at each level
        tokens_repr = self.span_encoder(tokens)

        if tokens.shape[1] == 1:
            selectors_input = torch.cat(
                (tokens_repr[:, 0, :], tokens_repr[:, 0, :]), dim=1
            )
        else:
            selectors_input = torch.cat(
                (tokens_repr[:, 0, :], tokens_repr[:, -1, :]), dim=1
            )

        max_intent = torch.max(self.intent_selector(selectors_input), 1)
        intent_type = Intent.intent_types[max_intent.indices[0].item()]

        max_slot = torch.max(self.slot_selector(selectors_input), 1)
        slot_type = Slot.slot_types[max_slot.indices[0].item()]

        cur_tree = IntentTree(
            tokenizer.decode([t.item() for t in tokens[0]]), (intent_type, slot_type)
        )
        label_score = max_intent.values[0].item() + max_slot.value[0].item()

        if tokens.shape[1] == 1:
            return (cur_tree, label_score)

        span_score = self.span_encoder(tokens)

        max_split_score = -1
        max_split_children: Optional[list] = None
        for i in range(tokens.shape[1] - 1):  # for all splits
            lsplit, rsplit = (tokens[:, :i], tokens[:, i:])
            ltree, lscore = self(lsplit)
            rtree, rscore = self(rsplit)

            if lscore + rscore > max_split_score:
                max_split_score = lscore + rscore
                max_split_children = [ltree, rtree]

        assert not max_split_children is None

        for child in max_split_children:
            cur_tree.add_child(child)

        return (cur_tree, label_score + span_score + max_split_score)

