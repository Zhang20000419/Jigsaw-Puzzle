from torch.nn import CrossEntropyLoss, Module


class LitCrossEntropyLoss(Module):
    def __init__(self, labels):
        super(LitCrossEntropyLoss, self).__init__()
        self.labels = labels
        self.loss = CrossEntropyLoss(reduction='sum')

    def forward(self, logits, y):
        logits = logits.view(-1, self.labels)
        y = y.view(-1)
        return self.loss(logits, y) / y.shape[0]

