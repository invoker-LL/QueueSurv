import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from .components import Attn_Net, Attn_Net_Gated, create_mlp, process_surv, process_clf
from .model_configs import ABMILConfig

QK_TIMES = 1
import math


def position_encoding_1d(d_model: int, length: int, base: float = 10000):
    assert d_model % 2 == 0, f"Cannot use sin/cos positional encoding with odd dim (got dim={d_model})"

    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(base) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


def apply_rotary_position_embeddings_nystrom(sinusoidal: torch.Tensor, *tensors):
    assert len(tensors) > 0, "at least one input tensor"
    N = sinusoidal.shape[0]
    # pdb.set_trace()
    cos_pos = sinusoidal[..., 1::2].repeat_interleave(2, 1).view(1, N, -1)
    sin_pos = sinusoidal[..., 0::2].repeat_interleave(2, 1).view(1, N, -1)
    cos_pos = cos_pos.expand_as(tensors[0])
    sin_pos = sin_pos.expand_as(tensors[0])

    outputs = []
    for t in tensors:
        t_r = torch.empty_like(t)
        t_r[..., 0::2] = -t[..., 1::2]
        t_r[..., 1::2] = t[..., 0::2]
        outputs.append(t * cos_pos + t_r * sin_pos)

    return outputs if len(tensors) > 1 else outputs[0]


def apply_rotary_position_embeddings(sinusoidal: torch.Tensor, *tensors):
    assert len(tensors) > 0, "at least one input tensor"
    N = sinusoidal.shape[0]

    cos_pos = sinusoidal[..., 1::2].repeat_interleave(2, 1).view(1, N, 1, -1)
    sin_pos = sinusoidal[..., 0::2].repeat_interleave(2, 1).view(1, N, 1, -1)
    cos_pos = cos_pos.expand_as(tensors[0])
    sin_pos = sin_pos.expand_as(tensors[0])

    outputs = []
    for t in tensors:
        t_r = torch.empty_like(t)
        t_r[..., 0::2] = -t[..., 1::2]
        t_r[..., 1::2] = t[..., 0::2]
        outputs.append(t * cos_pos + t_r * sin_pos)

    return outputs if len(tensors) > 1 else outputs[0]


class Rotary2D:

    def __init__(self, dim: int, base: float = 10000):
        self.dim = dim
        self.base = base
        self.pos_cached = None
        self.w_size_cached = None
        self.h_size_cached = None

    def forward(self, x_shape):
        H, W = int(x_shape[0].item()), int(x_shape[1].item())
        # pdb.set_trace()
        if self.pos_cached is None or self.w_size_cached != W or self.h_size_cached != H:
            # pdb.set_trace()
            print('forward')
            self.h_size_cached = H
            self.w_size_cached = W

            position_x = position_encoding_1d(H, self.dim // 2, self.base)
            position_y = position_encoding_1d(W, self.dim // 2, self.base)

            position_x = position_x.reshape(H, -1, 2)
            position_y = position_y.reshape(W, -1, 2)

            self.pos_cached = torch.empty(H * W, self.dim, dtype=torch.float).cuda()
            for i in range(H):
                for j in range(W):
                    emb = torch.cat([
                        position_x[i, 0::2],
                        position_y[j, 0::2],
                        position_x[i, 1::2],
                        position_y[j, 1::2]
                    ], 0).flatten(-2)
                    self.pos_cached[i * W + j] = emb.to(torch.float).cuda()
        return self.pos_cached


def exists(val):
    return val is not None


def precompute_freqs_cis(dim: int, end: int, pos_idx, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # t = pos_idx.cpu()
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    # pdb.set_trace()
    return freqs_cis[pos_idx.long()]


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, ):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


from torch import autograd
import copy


# class ABMIL(nn.Module):  # longmil
#     def __init__(self, config, mode, n_classes=4, input_size=1024, size2=1024):
#         super(ABMIL, self).__init__()
#         self.n_heads = 12
#         self.input_size = input_size
#         feat_size = size2
#         self.feat_size = feat_size
#         self._fc1 = nn.Sequential(nn.Linear(input_size, feat_size), nn.GELU())
#         self.mode = mode
#         from mil_models.modeling_finetune import WsiTransformer
#         self.vit2 = WsiTransformer(embed_dim=1024, num_classes=n_classes, depth=6, num_heads=16, mlp_ratio=4,
#                                    qkv_bias=True)
#
#         self.rotary = Rotary2D(dim=feat_size * QK_TIMES)
#         # self.alibi = torch.load('./alibi_tensor_core.pt').cuda()
#         # self.alibi = torch.load('./alibi_tensor_core_w20.pt').cuda()
#         # self.alibi_mask = self.alibi==-100
#         # self.slope_m = get_slopes(4).to(torch.float16)
#         # from models.ABMIL import ConvModel
#         from mil_models.simpleConv import ConvModel
#         self.spconv = ConvModel().cuda()
#         ckpt_path = '/mnt/Xsky/lhl/wsi_ft_git/beit2/ckpts/conv_1ckpts_fzEncoder_pathgen_Var_1_2_4_7_14_8192_3decoder_ConvPt/checkpoint.pth'
#         # ckpt_path = 'beit2/ckpts/conv_1ckpts_fzEncoder_pathGEN_Var_1_2_4_7_14_8192_ConvDecoder4/checkpoint.pth'
#         state_dict = torch.load(ckpt_path)['model']
#
#         def load_state_dict_temp(state_dict, name_str='spconv.'):
#             new_state_dict = {}
#             # name_str =
#             for k, v in state_dict.items():
#                 if k.find(name_str) >= 0:
#                     k = k[len(name_str):]
#                     # if k.find('blocks.') >= 0:
#                     #     text = k[len('blocks.'):]
#                     #     block_pos = text.find('.')
#                     #     block_num_idx = int(text[:block_pos])
#                     new_state_dict[k] = v
#                 # if k.find()
#             # pdb.set_trace()
#
#             return new_state_dict
#
#         conv_state_dict = load_state_dict_temp(state_dict)
#         self.spconv.load_state_dict(conv_state_dict)
#
#         ckpt_vit = '/mnt/Xsky/lhl/wsi_ft_git/beit2/pretrain_ckpt/wsi_pt_2dROPE_bs64_vqIDs_pathGen+BRACS_1e-3_5e-6_FzConv4_ep20/checkpoint.pth'
#
#         state_dict = torch.load(ckpt_vit)['model']
#
#         # import pdb;pdb.set_trace()
#
#         load_vit_pt = False
#
#         conv_state_dict = load_state_dict_temp(state_dict, 'conv_embed.')
#         self.spconv.load_state_dict(conv_state_dict)
#
#         use_lora = True
#         if load_vit_pt:
#             self.vit2.load_state_dict(state_dict, strict=False)
#         # else:
#         use_lora = False
#         import loralib as lora
#         def replace_linear_with_lora(model, rank=16):
#             for name, module in model.named_children():
#                 if 'head' in name:
#                     continue
#                 if isinstance(module, torch.nn.Linear):
#                     # 替换为 LoRA 线性层
#                     lora_layer = lora.Linear(
#                         in_features=module.in_features,  # 输入维度
#                         out_features=module.out_features,  # 输出维度
#                         r=rank  # LoRA 的秩
#                     )
#                     # 复制原始权重和偏置
#                     lora_layer.weight = module.weight
#                     lora_layer.bias = module.bias
#                     setattr(model, name, lora_layer)
#                 else:
#                     # 递归处理子模块
#                     replace_linear_with_lora(module, rank)
#
#         if use_lora:
#             replace_linear_with_lora(self.vit2)
#             for name, param in self.vit2.named_parameters():
#                 if 'lora' in name:  # 只训练 LoRA 参数
#                     param.requires_grad = True
#                     print(name)
#                 elif 'head' in name:
#                     pass
#                 else:
#                     param.requires_grad = False
#
#     def forward_no_loss(self, x):
#         max_len = 30000
#         x_feat, pos = x
#         x_feat = x_feat[:, :max_len]
#         pos = pos[:, :max_len]
#         # print(x_feat.shape)
#         pos = pos.squeeze()
#         x_feat = x_feat.cuda()
#         if False:
#             # with torch.no_grad():
#             if True:
#                 # import pdb;pdb.set_trace()
#                 # if False:
#                 # B,N,D = x_feat.shape
#                 # h = self.spconv(x_feat.transpose(2,1).view(B,D,7,7)).squeeze().unsqueeze(0)
#                 # h = self.spconv(x_feat.squeeze(0)).squeeze().unsqueeze(0)
#                 h = self.spconv(x_feat.to(torch.float16).squeeze()).squeeze().unsqueeze(0)
#         else:
#             # x_feat = x_feat.unsqueeze(0)  # .permute(0,2,1)
#             h = x_feat.to(torch.float16)
#             h = self._fc1(h)
#         freqs_cis, _, _ = self.positional_embedding(pos, use_rope=True)
#         # import pdb;pdb.set_trace()
#         logits = self.vit2(h, rope=freqs_cis.to(torch.float16))
#         # import pdb;pdb.set_trace()
#         # Y_hat = torch.argmax(logits, dim=-1)
#         # Y_prob = F.softmax(logits, dim=-1)
#
#         out = {'logits': logits}
#         return out
#
#     def forward(self, h, model_kwargs={}):
#
#         if self.mode == 'classification':
#             attn_mask = model_kwargs['attn_mask']
#             label = model_kwargs['label']
#             loss_fn = model_kwargs['loss_fn']
#
#             out = self.forward_no_loss(h, attn_mask=attn_mask)
#             logits = out['logits']
#
#             results_dict, log_dict = process_clf(logits, label, loss_fn)
#         elif self.mode == 'survival':
#             attn_mask = model_kwargs['attn_mask']
#             label = model_kwargs['label']
#             censorship = model_kwargs['censorship']
#             loss_fn = model_kwargs['loss_fn']
#
#             out = self.forward_no_loss([h, model_kwargs['coords']])
#             logits = out['logits']
#
#             results_dict, log_dict = process_surv(logits, label, censorship, loss_fn)
#         else:
#             raise NotImplementedError("Not Implemented!")
#
#         return results_dict, log_dict
#
#     def positional_embedding(self, x, use_alibi=False, use_rope=False):
#         # scale = 1  # for 20x 224 with 112 overlap (or 40x 224)
#         scale = 2  # for 20x 224 with 0 overlap
#         shape = 112  # or 128
#         # shape = 128
#         freqs_cis = None
#         alibi_bias = None
#         alibi_bias2 = None
#         # import pdb;pdb.set_trace()
#         #
#         if use_rope or use_alibi:
#             abs_pos = x[::, -2:]
#             # pdb.set_trace()
#             # print(abs_pos)
#             x_pos, y_pos = abs_pos[:, 0], abs_pos[:, 1]
#             x_pos = torch.round((x_pos - x_pos.min()) / (shape * scale) / 4)
#             y_pos = torch.round((y_pos - y_pos.min()) / (shape * scale) / 4)
#             H, W = 600 // scale, 600 // scale
#             selected_idx = (x_pos * W + y_pos).to(torch.int)
#
#             if use_rope:
#                 pos_cached = self.rotary.forward(torch.tensor([H, W]))
#                 freqs_cis = pos_cached[selected_idx].cuda()
#             if use_alibi:
#                 alibi_bias = self.alibi[selected_idx, :][:, selected_idx]
#                 # alibi_bias = alibi_bias.to(torch.float)
#                 # alibi_bias2 = torch.detach_copy(alibi_bias)
#                 # alibi_bias[torch.where(alibi_bias == -100)] = -torch.inf
#                 # pdb.set_trace()
#                 alibi_bias = alibi_bias[:, :, None] * self.slope_m[None, None, :].cuda()
#                 # pdb.set_trace()
#                 alibi_bias = alibi_bias.permute(2, 0, 1).unsqueeze(0)  # .float()
#
#                 shape3 = alibi_bias.shape[3]
#                 pad_num = 8 - shape3 % 8  # to tackle FlashAttention problems
#                 padding_bias = torch.zeros(1, alibi_bias.shape[1], alibi_bias.shape[2], pad_num).cuda()
#                 alibi_bias = torch.cat([alibi_bias, padding_bias], dim=-1)
#                 alibi_bias = autograd.Variable(alibi_bias.contiguous())[:, :, :, :shape3]
#                 # pdb.set_trace()
#                 alibi_bias2 = copy.deepcopy(alibi_bias)
#                 temp_min = alibi_bias2.min()
#                 indices_inf = torch.where(alibi_bias2 == temp_min)
#                 indices_local = torch.where(alibi_bias2 != temp_min)
#
#                 alibi_bias2[indices_inf] = -torch.inf
#                 alibi_bias2[indices_local] = 0
#                 #
#                 # pdb.set_trace()
#
#         return freqs_cis, alibi_bias2, alibi_bias


class ABMIL(nn.Module):
    def __init__(self, config, mode):
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

        self.classifier = nn.Linear(config.embed_dim, config.n_classes)
        self.n_classes = config.n_classes

        self.mode = mode

        ckpt_path = '/mnt/Xsky/lhl/wsi_ft_git/beit2/ckpts/conv_1ckpts_fzEncoder_pathgen_Var_1_2_4_7_14_8192_3decoder_ConvPt/checkpoint.pth'
        ckpt_path = '/mnt/Xsky/lhl/wsi_ft_git/beit2/ckpts/conchV1_5_ckpts_fzEncoder_pathGEN_Var_1_2_4_7_14_8192_3decoder/checkpoint.pth'
        state_dict = torch.load(ckpt_path)['model']

        def load_state_dict_temp(state_dict, name_str='decoder.'):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.find(name_str) >= 0:
                    k = k[len(name_str):]
                    new_state_dict[k] = v
            # pdb.set_trace()
            return new_state_dict

        conv_state_dict = load_state_dict_temp(state_dict, name_str='spconv.')
        # conv_state_dict = load_state_dict_temp(state_dict)
        from mil_models.simpleConv import ConvModel
        self.spconv = ConvModel()
        self.spconv.load_state_dict(conv_state_dict)

    def forward_attention(self, h, attn_only=False):
        # B: batch size
        # N: number of instances per WSI
        # L: input dimension
        # K: number of attention heads (K = 1 for ABMIL)
        # h is B x N x L
        # import pdb;pdb.set_trace()
        # h = self.spconv(h.squeeze()).squeeze().unsqueeze(0)

        h = self.mlp(h)
        # h is B x N x D
        A = self.attention_net(h)  # B x N x K
        A = torch.transpose(A, -2, -1)  # B x K x N
        if attn_only:
            return A
        else:
            return h, A

    def forward_no_loss(self, h, attn_mask=None):
        h, A = self.forward_attention(h)
        A_raw = A
        # A is B x K x N
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.bmm(A, h).squeeze(dim=1)  # B x K x C --> B x C
        logits = self.classifier(M)

        out = {'logits': logits, 'attn': A, 'feats': h, 'feats_agg': M}

        return out

    def forward_feature(self, h, attn_mask=None):
        h, A = self.forward_attention(h)
        # A is B x K x N
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.bmm(A, h).squeeze(dim=1)  # B x K x C --> B x C
        return M

    def forward_head(self, M):
        return self.classifier(M)

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