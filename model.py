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

    def forward(self, spans_repr: torch.Tensor) -> torch.Tensor:
        """
        :param spans_repr: (batch_size, 2, hidden_size)
        :return:           (batch_size, 1)
        """
        # (batch_size, spans_nb, 2 * hidden_size)
        flat_spans = torch.flatten(spans_repr, start_dim=1).float()
        return self.scorer(flat_spans)


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
    def __init__(self, tokenizer: BertTokenizer):
        super(TreeMaker, self).__init__()
        self.tokenizer = tokenizer
        self.span_encoder = SpanEncoder()
        self.span_scorer = SpanScorer(self.span_encoder.hidden_size)
        self.intent_selector = LabelSelector(
            self.span_encoder.hidden_size, len(list(Intent.intent_types))
        )
        self.slot_selector = LabelSelector(
            self.span_encoder.hidden_size, len(list(Slot.slot_types))
        )

    def forward(
        self, tokens: torch.Tensor, encoded_tokens: Optional[torch.Tensor] = None
    ) -> (IntentTree, float):
        """
        :param tokens: (batch_size?, seq_size)
        :param encoded_tokens: (batch_size?, seq_size, hidden_size)
        """
        if tokens.shape[1] == 0:
            raise Exception(
                f"[error] len(tokens) must be > 0 (tokens shape : {tokens.shape})"
            )
        if not encoded_tokens is None and encoded_tokens.shape[1] == 0:
            raise Exception(
                f"[error] len(encoded_tokens) must be > 0 (encoded_tokens shape : {encoded_tokens.shape})"
            )

        # (batch_size?, seq_size, hidden_size)
        if encoded_tokens is None:
            encoded_tokens = self.span_encoder(tokens)

        tokens_repr = torch.cat(
            (encoded_tokens[:, 0, :], encoded_tokens[:, -1, :]), dim=1
        )

        max_intent = torch.max(self.intent_selector(tokens_repr), 1)
        intent_type = Intent.intent_types[max_intent.indices[0].item()]

        max_slot = torch.max(self.slot_selector(tokens_repr), 1)
        slot_type = Slot.slot_types[max_slot.indices[0].item()]

        node_text = self.tokenizer.decode([t.item() for t in tokens[0]])
        label_score = max_intent.values[0].item() + max_slot.values[0].item()

        if intent_type == "NOT_INTENT" and slot_type == "NOT_INTENT":
            cur_tree = IntentTree(node_text, None)
            return cur_tree, label_score
        elif intent_type == "NOT_INTENT":
            cur_tree = IntentTree(node_text, Slot(slot_type))
        elif slot_type == "NOT_INTENT":
            cur_tree = IntentTree(node_text, Intent(intent_type))
        else:
            slot_tree = IntentTree(node_text, Slot(slot_type))
            cur_tree = IntentTree(node_text, Intent(intent_type))
            slot_tree.add_child(cur_tree)

        if tokens.shape[1] == 1:
            return (cur_tree, label_score)

        span_score = self.span_scorer(tokens_repr)[0][0].item()

        max_split_score = -1
        max_split_children: Optional[list] = None
        for i in range(tokens.shape[1] - 1):  # for all splits
            ltree, lscore = self(tokens[:, : i + 1], encoded_tokens[:, : i + 1, :])
            rtree, rscore = self(tokens[:, i + 1 :], encoded_tokens[:, i + 1 :, :])

            if lscore + rscore > max_split_score:
                max_split_score = lscore + rscore
                max_split_children = [ltree, rtree]

        assert not max_split_children is None

        for child in max_split_children:
            cur_tree.add_child(child)

        return (cur_tree, label_score + span_score + max_split_score)

    def score_tree(self, tree: IntentTree) -> torch.Tensor:
        tokens_repr = self.span_encoder(
            torch.tensor(self.tokenizer.encode(tree.tokens)).unsqueeze(0)
        )
        span_repr = torch.cat((tokens_repr[:, 0, :], tokens_repr[:, -1, :]), dim=1)
        span_score = self.span_scorer(span_repr)

        if len(tree.children) == 0:
            return span_score

        if tree.node_type is None:
            label_score = (
                self.intent_selector(span_repr)[Intent.stoi("NOT_INTENT")]
                * self.slot_selector(span_repr)[Slot.stoi("NOT_SLOT")]
            )
        elif isinstance(tree.node_type, Intent):
            label_score = self.intent_selector(span_repr)[0][
                Intent.stoi(tree.node_type.type)
            ]
        elif isinstance(tree.node_type, Slot):
            label_score = self.slot_selector(span_repr)[0][
                Slot.stoi(tree.node_type.type)
            ]

        children_scores = torch.zeros(len(tree.children))
        for i, child in enumerate(tree.children):
            children_scores[i] = self.score_tree(child)

        return span_score + label_score + torch.sum(children_scores)
