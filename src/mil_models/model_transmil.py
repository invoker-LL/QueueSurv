import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from .components import Attn_Net, Attn_Net_Gated, create_mlp, process_surv, process_clf
from .model_configs import ABMILConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, config, mode):
        super(TransMIL, self).__init__()
        self.input_size = config.in_dim
        feat_size = config.embed_dim
        self.pos_layer = PPEG(dim=feat_size)
        self._fc1 = nn.Sequential(nn.Linear(self.input_size, feat_size), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, feat_size))
        self.n_classes = config.n_classes
        self.layer1 = TransLayer(dim=feat_size)
        self.layer2 = TransLayer(dim=feat_size)
        self.norm = nn.LayerNorm(feat_size)
        self._fc2 = nn.Linear(feat_size, self.n_classes)
        self.mode=mode

    def forward_no_loss(self, x, attn_mask=None):
        h = x.float()
        h = self._fc1(h)  # [B, n, 512]

        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        # import pdb;pdb.set_trace()
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        h = self.norm(h)[:, 0]

        # ---->predict
        logits = self._fc2(h)  # [B, n_classes]

        return {'logits':logits}

    def forward(self, h, model_kwargs={}):

        if self.mode == 'classification':
            attn_mask = model_kwargs['attn_mask']
            label = model_kwargs['label']
            loss_fn = model_kwargs['loss_fn']

            out = self.forward_no_loss(h, attn_mask=attn_mask)
            logits = out['logits']

            results_dict, log_dict = process_clf(logits, label, loss_fn)
        elif self.mode == 'survival':
            attn_mask = model_kwargs['attn_mask']
            label = model_kwargs['label']
            censorship = model_kwargs['censorship']
            loss_fn = model_kwargs['loss_fn']

            out = self.forward_no_loss(h, attn_mask=attn_mask)
            logits = out['logits']

            results_dict, log_dict = process_surv(logits, label, censorship, loss_fn)
        else:
            raise NotImplementedError("Not Implemented!")

        return results_dict, log_dict

# class ABMILSurv(ABMIL):

#     def __init__(self, config: ABMILConfig):
#         super().__init__(config)

#     def forward(self, h, attn_mask=None, label=None, censorship=None, loss_fn=None):
#         out = self.forward_no_loss(h, attn_mask=attn_mask)
#         logits = out['logits']

#         results_dict, log_dict = process_surv(logits, label, censorship, loss_fn)

#         return results_dict, log_dict