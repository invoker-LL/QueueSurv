import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from .components import Attn_Net, Attn_Net_Gated, create_mlp, process_surv, process_clf
from .model_configs import ABMILConfig


class MIL_Enc(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mlp = create_mlp(in_dim=config.in_dim,
                              hid_dims=[config.embed_dim] *
                              (config.n_fc_layers - 1),
                              dropout=config.dropout,
                              out_dim=config.embed_dim,
                              end_with_fc=False)

        if config.gate:
            self.attention_net = Attn_Net_Gated(L=self.config.embed_dim,
                                                D=config.attn_dim,
                                                dropout=config.dropout,
                                                n_classes=1)
        else:
            self.attention_net = Attn_Net(L=config.embed_dim,
                                          D=config.attn_dim,
                                          dropout=config.dropout,
                                          n_classes=1)


    def forward_attention(self, h, attn_only=False):
        # B: batch size
        # N: number of instances per WSI
        # L: input dimension
        # K: number of attention heads (K = 1 for ABMIL)
        # h is B x N x L
        # import pdb;pdb.set_trace()
        h = self.mlp(h.float())
        # h is B x N x D
        A = self.attention_net(h)  # B x N x K
        A = torch.transpose(A, -2, -1)  # B x K x N
        if attn_only:
            return A
        else:
            return h, A

    def forward_feature(self, h, attn_mask=None):
        # return h.mean(1).float()
        h, A = self.forward_attention(h)
        # A is B x K x N
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.bmm(A, h).squeeze(dim=1)  # B x K x C --> B x C
        return M

from collections import deque

class DequeQueue:
    def __init__(self):
        self.items = deque()

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if self.is_empty():
            raise IndexError("队列为空，无法出队")
        return self.items.popleft()

    def peek(self):
        if self.is_empty():
            raise IndexError("队列为空，无法查看")
        return self.items[0]

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

    def show_all(self):
        return list(self.items)

import random
class ABMIL(nn.Module):
    def __init__(self, config, mode):
        super().__init__()
        self.config = config
        self.encoder_k = MIL_Enc(config)

        self.classifier = nn.Linear(config.embed_dim, config.n_classes)
        self.n_classes = config.n_classes

        self.mode = mode

        self.encoder_q = MIL_Enc(config)
        self.queue_emb = DequeQueue()
        self.queue_label = DequeQueue()

    def forward_head(self,M):
        return self.classifier(M)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        self.m = 0.9
        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        # for param_q, param_k in zip(
        #         self.encoder_q.parameters(), self.encoder_k.parameters()
        # ):
        #     param_q.data = param_k.data

    def get_queue(self, M_q, label, censorship):

        # random.shuffle(all_q_label)
        # random.shuffle(all_q_emb)

        max_q_len = 32
        while self.queue_emb.size()>=max_q_len-1:
            self.queue_emb.dequeue() # 30
            self.queue_label.dequeue()
        self.queue_emb.enqueue(M_q) #31
        self.queue_label.enqueue(torch.cat([label, censorship],dim=-1))
        #
        all_q_emb = self.queue_emb.show_all()
        all_q_label = self.queue_label.show_all()



        return all_q_emb, all_q_label


    def forward_no_loss(self, h, attn_mask=None,label=None,censorship=None):
        # if self.training:
        #     M = self.encoder_q.forward_feature(h)
        # else:
        #     M = self.encoder_k.forward_feature(h)
        M = self.encoder_q.forward_feature(h)

        with torch.no_grad():
            if self.training:
            # if False:
                self._momentum_update_key_encoder()
                M_k = self.encoder_k.forward_feature(h)
                all_q_emb, all_q_label = self.get_queue(M_k, label, censorship)
                all_q_emb.append(M)
                all_q_label.append(torch.cat([label, censorship],dim=-1))
                M = torch.cat(all_q_emb)
                label = torch.cat(all_q_label)
                label, censorship = label[:,:1], label[:,1:]
                # import pdb;pdb.set_trace()

        # import pdb;pdb.set_trace()
        logits = self.classifier(M)
        # print(logits.shape)
        out = {'logits': logits, 'censorship': censorship, 'feats': None, 'feats_agg': None, 'label':label}

        return out

    def forward(self, h, model_kwargs={}):

        if self.mode == 'survival':
            attn_mask = model_kwargs['attn_mask']
            label = model_kwargs['label']
            censorship = model_kwargs['censorship']
            loss_fn = model_kwargs['loss_fn']

            out = self.forward_no_loss(h, attn_mask=attn_mask,label=label,censorship=censorship)
            logits = out['logits']
            label = out['label']
            # print(label.shape)
            censorship = out['censorship']
            # import pdb;pdb.set_trace()
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