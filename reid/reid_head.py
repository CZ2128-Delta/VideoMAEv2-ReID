import torch.nn as nn
import torch.nn.functional as F


class ReIDHead(nn.Module):
    """BNNeck-based classification head for ReID."""

    def __init__(self, in_dim, num_classes, embed_dim=512, neck_feat='after'):
        super().__init__()
        self.num_classes = num_classes
        self.neck_feat = neck_feat
        self.embed_dim = embed_dim

        self.feat_proj = nn.Linear(in_dim, embed_dim)
        self.bottleneck = nn.BatchNorm1d(embed_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)

        self._init_params()

    def _init_params(self):
        nn.init.kaiming_normal_(self.feat_proj.weight, mode='fan_out')
        nn.init.constant_(self.feat_proj.bias, 0)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, features, label=None):
        feat_proj = self.feat_proj(features)
        bn_feat = self.bottleneck(feat_proj)

        if self.neck_feat == 'after':
            feat = F.normalize(bn_feat, p=2, dim=1)
        else:
            feat = F.normalize(feat_proj, p=2, dim=1)

        if self.training:
            cls_score = self.classifier(bn_feat)
            return cls_score, feat_proj, feat
        else:
            return feat
