import torch
from transformers import BertModel


class SpanEncoder(torch.nn.Module):
    def __init__(self):
        super(SpanEncoder, self).__init__()
        # TODO: include spanbert
        self.bert_encoder = BertModel.from_pretrained("models/spanbert")
        self.hidden_size = self.bert_encoder.config.hidden_size

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        :param seqs: (batch_size, seq_size)
        :return:     (batch_size, seq_size, hidden_size)
        """
        return self.bert_encoder(seqs)


class SpanSelector(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super(SpanSelector, self).__init__()
        self.selector = torch.nn.Linear(2 * hidden_size, 1)

    def forward(self, spans_repr: torch.Tensor) -> torch.Tensor:
        """
        :param spans_repr: (batch_size, spans_nb, 2, hidden_size)
        :return:           (batch_size, spans_nb)
        """
        # (batch_size, spans_nb, 2 * hidden_size)
        flat_spans = torch.flatten(spans_repr, start_dim=2)

        return self.selector(flat_spans).squeeze()


class LabelSelector(torch.nn.Module):
    def __init__(self, hidden_size: int, labels_nb: int):
        super(LabelSelector, self).__init__()
        self.selector = torch.nn.Linear(2 * hidden_size, labels_nb)

    def forward(self, spans_repr: torch.Tensor) -> torch.Tensor:
        """
        :param spans_repr: (batch_size, spans_nb, 2, hidden_size)
        :return:           (batch_size, spans_nbs, labels_nb)
        """
        # (batch_size, spans_nb, 2 * hidden_size)
        flat_spans = torch.flatten(spans_repr, start_dim=2)
        return self.selector(flat_spans)


# TODO: research tree structure
class TreeScorer(torch.nn.Module):
    def __init__(self, labels_nb: int):
        super(TreeScorer, self).__init__()
        self.span_encoder = SpanEncoder()
        self.span_selector = SpanSelector(self.span_encoder.hidden_size)
        self.label_selector = LabelSelector(self.span_encoder.hidden_size, labels_nb)

    def forward(self, span: torch.Tensor, split: int):
        """
        :param span: (batch_size, seq_size, hidden_size)
        """
        if span.shape[1] == 1:
            span_repr = torch.cat((span[:, 0, :], span[:, 0, :]))
            return torch.softmax(self.label_selector(span_repr))
        span_repr = torch.cat((span[:, 0, :], span[:, -1, :]))
        return torch.softmax(self.label_selector(span_repr))
