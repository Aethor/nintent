from typing import Optional, List, Tuple, Mapping
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

        self.node_type_selector = Selector(self.hidden_size, IntentTree.node_types_nb())
        self.intent_type_selector = Selector(self.hidden_size, Intent.intent_types_nb())
        self.slot_type_selector = Selector(self.hidden_size, Slot.slot_types_nb())

        # 1 reprensents positive class by convention
        self.is_terminal_selector = Selector(self.hidden_size, 2)

        self.clear_tree_cache_()

    def clear_tree_cache_(self):
        self.tree_cache: Mapping[str, IntentTree] = dict()

    def _hash_tensor(self, t: torch.Tensor) -> str:
        """
        :param t: (n)
        :return: a string as a hash
        """
        return " ".join([str(e.item()) for e in t])

    def new_train(
        self,
        tokens: torch.Tensor,
        tokens_repr: Optional[torch.Tensor],
        gold_tree: IntentTree,
    ) -> (IntentTree, torch.Tensor):
        """
        :param tokens:      (batch_size, seq_size)
        :param tokens_repr: (batch_size, seq_size, hidden_size)
        :return: (intent tree, loss)
        """
        self.train()
        if tokens_repr is None:
            # (batch_size, seq_size, hidden_size)
            tokens_repr = self.span_encoder(tokens)
        span_repr = torch.cat((tokens_repr[:, 0, :], tokens_repr[:, -1, :]), dim=1)

        loss = torch.tensor([0]).to(tokens.device)

        node_type_pred = self.node_type_selector(span_repr)
        node_type = IntentTree.node_types[torch.max(node_type_pred, 1).indices.item()]
        loss += (
            1
            - torch.max(node_type_pred, 1).indices
            + node_type_pred[IntentTree.node_types_idx(type(gold_tree.node_type))]
        )
        if node_type == Intent:
            intent_type_pred = self.intent_type_selector(span_repr)
            loss += (
                1
                - torch.max(intent_type_pred, 1).indices
                + intent_type_pred[Intent.stoi(gold_tree.node_type)]
            )
        elif node_type == Slot:
            slot_type_pred = self.slot_type_selector(span_repr)
            loss += (
                1
                - torch.max(slot_type_pred, 1).indices
                + slot_type_pred[Slot.stoi(gold_tree.node_type)]
            )
        elif node_type is None:
            # FIXME: check that
            loss += 1

        splits_nb = tokens.shape[1] - 1
        best_split_score = torch.tensor([0])
        for i in range(splits_nb):
            l_span_score = self.span_scorer(
                torch.cat((tokens_repr[:, i + 1, :], tokens_repr[:, -1, :]), dim=1)
            )[0]
            r_span_score = self.span_scorer(
                torch.cat((tokens_repr[:, 0, :], tokens_repr[:, i + 1, :]), dim=1)
            )[0]
            split_score = l_span_score + r_span_score
            if split_score > best_split_score:
                best_split_score = split_score

        # TODO: make trees have directly tensors ar tokens
        gold_tree_tokens = (
            self.tokenizer.encode(gold_tree.tokens).unsqueeze(0).to(device)
        )
        gold_tree_split_score = torch.tensor([0])
        for child in gold_tree.children:
            gold_tree_split_tokens_repr = self.span_encoder(gold_tree_tokens)
            gold_tree_split_span_repr = torch.cat(
                (
                    gold_tree_split_span_repr[:, 0, :],
                    gold_tree_split_span_repr[:, -1, :],
                )
            )
            gold_tree_split_score += self.span_scorer(gold_tree_split_span_repr)[0]

        loss += 1 - best_split_score + gold_tree_split_score

        # TODO: mechanism to know tree split points

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
    def make_tree(
        self, tokens: torch.Tensor, tokens_repr: Optional[torch.Tensor]
    ) -> IntentTree:
        """
        :param tokens: (batch_size, seq_size)
        :span_repr:    (batch_size, seq_size, hidden_size)
        :note: only works with batch size equals to 1
        """
        if tokens_repr is None:
            # (batch_size, seq_size, hidden_size)
            tokens_repr = self.span_encoder(tokens)
        # (batch_size, 2, hidden_size)
        span_repr = torch.cat((tokens_repr[:, 0, :], tokens_repr[:, -1, :]), dim=1)

        tokens_hash = self._hash_tensor(tokens[0])
        if tokens_hash in self.tree_cache:
            return copy.deepcopy(self.tree_cache[tokens_hash])

        node_type = IntentTree.node_types[
            torch.max(self.node_type_selector(span_repr), 1).indices.item()
        ]

        decoded_tokens = self.tokenizer.decode([t.item() for t in tokens[0]])
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

        self.tree_cache[tokens_hash] = cur_tree

        # Check if the current tree should be terminal
        if (
            torch.max(self.is_terminal_selector(span_repr), 1).indices.item() == 1
            or len(tokens[0]) == 1
        ):
            return cur_tree

        children: List[Tuple[IntentTree]] = []
        for i in range(tokens.shape[1] - 1):
            ltree = self.make_tree(tokens[:, : i + 1], tokens_repr[:, : i + 1, :])
            rtree = self.make_tree(tokens[:, i + 1 :], tokens_repr[:, i + 1 :, :])
            children.append((ltree, rtree))

        best_children_pair = max(
            children,
            key=lambda e: self(e[0], tokens.device) + self(e[1], tokens.device),
        )

        cur_tree.add_children_(best_children_pair)

        return cur_tree
