import torch.nn as nn
import torch.nn.functional as F

class SelfDistillation(nn.Module):
    def __init__(self, dim, depth, num_cls, T, alpha, beta):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.CE = nn.CrossEntropyLoss()
        self.L2 = nn.MSELoss()
        self.classifiers = nn.ModuleList(nn.Linear(dim, num_cls) for _ in range(depth))

    def forward(self, op_last, steps, labels):
        outputs = []
        total_ce = 0
        total_kl = 0
        total_l2 = 0
        for i in range(len(steps)):
            outputs.append(self.classifiers[i](steps[i]))
            total_ce = total_ce + self.CE(outputs[-1], labels)
            kl = F.kl_div(
                F.log_softmax(outputs[-1] / self.T, dim=1),
                F.softmax(op_last / self.T, dim=1),
                reduction='batchmean'
            )
            total_kl = kl + total_kl
            l2 = self.L2(steps[i], steps[-1])
            total_l2 = total_l2 + l2
        total_loss = (1 - self.alpha) * total_ce + self.alpha * total_kl + self.beta * total_l2
        return total_loss