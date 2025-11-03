import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.log_softmax(inputs)
        targets = F.one_hot(targets, self.num_classes).float()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings, targets):
        embeddings = embeddings.float()
        dist_mat = euclidean_dist(embeddings, embeddings)
        n = dist_mat.size(0)

        is_pos = targets.expand(n, n).eq(targets.expand(n, n).t())
        is_neg = targets.expand(n, n).ne(targets.expand(n, n).t())

        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist_mat[i][is_pos[i]].max().unsqueeze(0))
            dist_an.append(dist_mat[i][is_neg[i]].min().unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class CombinedLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 label_smooth_epsilon=0.1,
                 use_triplet=True,
                 triplet_margin=0.3,
                 id_weight=1.0,
                 triplet_weight=1.0):
        super().__init__()
        self.id_loss = CrossEntropyLabelSmooth(num_classes, epsilon=label_smooth_epsilon)
        self.use_triplet = use_triplet
        self.id_weight = id_weight
        self.triplet_weight = triplet_weight
        if self.use_triplet:
            self.triplet_loss = TripletLoss(margin=triplet_margin)

    def forward(self, cls_score, global_feat, feat, labels):
        id_loss_val = self.id_loss(cls_score, labels)
        loss = self.id_weight * id_loss_val
        loss_dict = {"id_loss": id_loss_val.detach()}

        if self.use_triplet:
            triplet_loss_val = self.triplet_loss(global_feat, labels)
            loss = loss + self.triplet_weight * triplet_loss_val
            loss_dict["triplet_loss"] = triplet_loss_val.detach()

        return loss, loss_dict
