import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from fastai.basics import *


BN1d = nn.InstanceNorm1d   ## partial(Norm, ndim=1, norm='Instance') 


class Concat(Module):
    def __init__(self, dim=1): self.dim = dim
    def forward(self, *x): return torch.cat(*x, dim=self.dim)
    def __repr__(self): return f'{self.__class__.__name__}(dim={self.dim})'


def Conv1d(ni, nf, kernel_size=None, ks=None, stride=1, padding='same', dilation=1, init='auto', bias_std=0.01, **kwargs):
    "conv1d layer with padding='same', 'causal', 'valid', or any integer (defaults to 'same')"
    assert not (kernel_size and ks), 'use kernel_size or ks but not both simultaneously'
    assert kernel_size is not None or ks is not None, 'you need to pass a ks'
    kernel_size = kernel_size or ks
    if padding == 'same': 
        if kernel_size%2==1: 
            conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, padding=kernel_size//2 * dilation, dilation=dilation, **kwargs)
        else:
            conv = SameConv1d(ni, nf, kernel_size, stride=stride, dilation=dilation, **kwargs)
    elif padding == 'causal': conv = CausalConv1d(ni, nf, kernel_size, stride=stride, dilation=dilation, **kwargs)
    elif padding == 'valid': conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, padding=0, dilation=dilation, **kwargs)
    else: conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, padding=padding, dilation=dilation, **kwargs)
    init_linear(conv, None, init=init, bias_std=bias_std)
    return conv


class InceptionModule(Module):
    def __init__(self, ni, nf, ks=40, bottleneck=True):
        ks = [ks // (2**i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]  # ensure odd ks
        bottleneck = bottleneck if ni > 1 else False
        self.bottleneck = Conv1d(ni, nf, 1, bias=False) if bottleneck else noop
        self.convs = nn.ModuleList([Conv1d(nf if bottleneck else ni, nf, k, bias=False) for k in ks])
        self.maxconvpool = nn.Sequential(*[nn.MaxPool1d(3, stride=1, padding=1), Conv1d(ni, nf, 1, bias=False)])
        self.concat = Concat()
        self.bn = BN1d(nf * 4)
        self.act = nn.GELU()  ## Raw is ReLU

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        x = self.concat([l(x) for l in self.convs] + [self.maxconvpool(input_tensor)])
        return self.act(self.bn(x))
       

class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())

        return combined

    def expert_to_gates(self):
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gelu = nn.GELU()
        self.out_drop = nn.Dropout(p=0.15)

    def forward(self, x):
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.out_drop(out)
        out = self.fc2(out)
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.gamma * self.scale


class MoE_Block(nn.Module):
    def __init__(self, input_size, output_size, num_experts, hidden_size, k=1):
        super(MoE_Block, self).__init__()
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k

        self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.rmsnorm = RMSNorm(dim=self.output_size)
        self.act = nn.GELU()

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)


    def top_k_gating(self, x):
        logits = x @ self.w_gate
        logits = self.softmax(logits)
        
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  # normalization

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        load = self._gates_to_load(gates)
      
        return gates, load

    def forward(self, x):
        batch_size, num_patches, feature_size = x.shape  # x: (batch_size, num_patches, input_size)
        x_flat = x.reshape(batch_size * num_patches, feature_size)  # Flatten patches for processing
        gates, load = self.top_k_gating(x_flat)

        # calculate importance loss
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x_flat)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]

        y = x_flat + dispatcher.combine(expert_outputs)
        y = self.rmsnorm(y.view((batch_size, num_patches, feature_size)))
        
        return self.act(y), loss


class ShapeEmbedLayer(nn.Module):  
    def __init__(self, seq_len, shape_size=8, in_chans=1, embed_dim=128, stride=4):
        super().__init__()
        stride = stride
        num_patches = int((seq_len - shape_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=shape_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out

############################################################################
class MultiScaleShapeEmbed(nn.Module):
    def __init__(self, seq_len, shape_sizes=(8, 16, 32), in_chans=1, embed_dim=128, stride=4, fusion="concat"):
        super().__init__()
        assert fusion in ["concat", "sum"]
        self.fusion = fusion
        self.convs = nn.ModuleList([
            nn.Conv1d(in_chans, embed_dim, kernel_size=k, stride=stride)
            for k in shape_sizes
        ])

        # 统一 token 长度：不同 kernel 得到的 N 不同，这里对齐到最小 N（裁剪）
        # 也可以改成插值对齐到最大 N
        if fusion == "concat":
            self.out_proj = nn.Linear(embed_dim * len(shape_sizes), embed_dim)
        else:
            self.out_proj = nn.Identity()

        # 预估一个 num_patches（以最小 kernel 为准）
        k_min = min(shape_sizes)
        self.num_patches = int((seq_len - k_min) / stride + 1)

    def forward(self, x):
        outs = []
        Ns = []
        for conv in self.convs:
            y = conv(x)                # [B, D, N_i]
            Ns.append(y.shape[-1])
            outs.append(y)
        N_min = min(Ns)
        outs = [o[..., :N_min] for o in outs]  # 裁剪对齐到最小长度
        if self.fusion == "sum":
            y = torch.stack(outs, dim=0).sum(dim=0)  # [B, D, N]
            tokens = y.transpose(1, 2)               # [B, N, D]
            return tokens
        else:
            y = torch.cat(outs, dim=1)               # [B, D*K, N]
            tokens = y.transpose(1, 2)               # [B, N, D*K]
            tokens = self.out_proj(tokens)           # [B, N, D]
            return tokens
#################################################################################


def coml_index(input_indx, dim):
    full_idx = torch.arange(dim)
    mask = torch.ones(input_indx.size(0), dim, dtype=torch.bool)

    for i in range(input_indx.size(0)):
        mask[i, input_indx[i]] = False

    complement_idx = torch.stack([full_idx[mask[i]] for i in range(input_indx.size(0))])
    return complement_idx


# class SoftShapeNet_layer(nn.Module): 
#     def __init__(self, dim, moe_nets=None, atten_head=None):
#         super().__init__()

#         self.norm1 = RMSNorm(dim)
#         self.norm2 = RMSNorm(dim)
#         self.attention_head = atten_head
#         self.out_drop = nn.Dropout(p=0.15)
#         self.moe = moe_nets
#         self.inception = InceptionModule(dim, 32)
#         self.act = nn.GELU()

#     def forward(self, x, end_depth=False, remain_ratio=1.0):

#         if remain_ratio < 1.0:
#             x = self.norm1(x)
#             attn_x_score = self.attention_head(x)

#             left_patch_tokens = math.ceil(remain_ratio * x.shape[1])
#             _, left_idx = torch.topk(attn_x_score, left_patch_tokens, dim=1, largest=True, sorted=True)

#             compl_left_indx = coml_index(input_indx=left_idx.squeeze(-1), dim=x.shape[1])
#             compl_left_indx = compl_left_indx.unsqueeze(2)

#             sorted_left_idx, _ = torch.sort(left_idx, dim=1)
#             left_index = sorted_left_idx.expand(-1, -1, x.shape[2])
#             compl = compl_left_indx.to(left_index.device)
#             non_topk = torch.gather(x * attn_x_score, dim=1, index=compl.expand(-1, -1, x.shape[2]))  # [B, N-1-left_tokens, C]
#             extra_token = torch.sum(non_topk, dim=1, keepdim=True)  # [B, 1, C]
#             left_x = torch.gather(x * attn_x_score, dim=1, index=left_index)  # [B, left_tokens, C]

#             x = torch.cat([left_x, extra_token], dim=1)           
#             incep_x = self.inception(self.norm2(x).permute(0, 2, 1))
#             reshape_incep_x = incep_x.permute(0, 2, 1)
#             temp_moe_x, moe_loss = self.moe(self.norm2(x))
#             x = x + temp_moe_x + reshape_incep_x
#         else:
#             x = self.norm1(x)
#             attn_x_score = self.attention_head(x)
#             x = x * attn_x_score            
#             incep_x = self.inception(self.norm2(x).permute(0, 2, 1))
#             reshape_incep_x = incep_x.permute(0, 2, 1)
#             moe_loss = 0.0
#             x = x + reshape_incep_x

#         end_attn_x_score = None
#         if end_depth:
#             x = self.out_drop(x)
#             end_attn_x_score = self.attention_head(x)

#         return self.act(x), moe_loss, end_attn_x_score


# =========================================================
# HyperGraph Neural Network (HGNN) modules
# 说明：
# - 节点：patch tokens (N 个)
# - 超边：以每个节点为中心，取其 kNN 形成一个超边（E=N）
# - 超边特征：超边内节点特征均值/加权均值
# - 节点更新：聚合“自己为中心的超边”特征（高效，避免构造巨大 incidence matrix）
# =========================================================

class HyperEdgeBuilder(nn.Module):
    """
    基于特征相似度构建超边：每个节点 -> 一个超边（包含自己 + k 个近邻）
    """
    def __init__(self, k=8, metric="cosine"):
        super().__init__()
        self.k = k
        self.metric = metric

    @torch.no_grad()
    def forward(self, x):
        """
        x: (B, N, C)
        return:
          idx: (B, N, K)  每个节点对应的超边成员索引（K = k+1, 含自身）
        """
        B, N, C = x.shape
        K = min(self.k + 1, N)

        if self.metric == "cosine":
            xn = F.normalize(x, dim=-1)  # (B,N,C)
            sim = torch.matmul(xn, xn.transpose(1, 2))  # (B,N,N)
            # 取相似度 topK（含自身）
            _, idx = torch.topk(sim, k=K, dim=-1, largest=True, sorted=False)  # (B,N,K)
        else:
            # euclidean：距离越小越近 -> 取 smallest
            dist = torch.cdist(x, x)  # (B,N,N)
            _, idx = torch.topk(dist, k=K, dim=-1, largest=False, sorted=False)  # (B,N,K)

        return idx


class HyperGraphConv(nn.Module):
    """
    简化 HGNN：以“每个节点为中心的超边”做一次超边聚合，再投影回节点空间
    """
    def __init__(self, dim, k=8, metric="cosine", edge_drop=0.0):
        super().__init__()
        self.builder = HyperEdgeBuilder(k=k, metric=metric)
        self.edge_drop = float(edge_drop)

        self.pre = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU()
        )
        self.post = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU()
        )
        self.norm = RMSNorm(dim)

    def forward(self, x):
        """
        x: (B, N, C)
        return: (B, N, C)
        """
        B, N, C = x.shape
        x0 = x

        x = self.pre(x)
        idx = self.builder(x)  # (B,N,K)

        # gather: (B, N, K, C)
        idx_expand = idx.unsqueeze(-1).expand(-1, -1, -1, C)
        x_expand = x.unsqueeze(1).expand(-1, N, -1, -1)  # (B,N,N,C)
        members = torch.gather(x_expand, dim=2, index=idx_expand)  # (B,N,K,C)

        # 超边特征：均值（也可换成 max / attention）
        e = members.mean(dim=2)  # (B,N,C)

        # 可选：超边 dropout（提升泛化）
        if self.edge_drop > 0 and self.training:
            drop_mask = (torch.rand(B, N, 1, device=x.device) > self.edge_drop).float()
            e = e * drop_mask

        out = self.post(e)
        out = self.norm(out + x0)  # 残差 + RMSNorm
        return out


class FusionGate(nn.Module):
    """
    用可学习门控融合多个分支：避免直接相加导致训练不稳
    """
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x, a, b):
        """
        x: base
        a,b: two branches
        return: x + gate*a + (1-gate)*b
        """
        gate = self.g(x)
        return x + gate * a + (1.0 - gate) * b


# =========================================================
# 覆盖 SoftShapeNet_layer：加入 HyperGraphConv 分支
# 注意：不动你上面的定义，这里同名覆盖即可
# =========================================================
class SoftShapeNet_layer(nn.Module):
    def __init__(self, dim, moe_nets=None, atten_head=None,
                 hg_k=8, hg_metric="cosine", hg_edge_drop=0.0,
                 hg_weight=1.0):
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attention_head = atten_head
        self.out_drop = nn.Dropout(p=0.15)

        self.moe = moe_nets
        self.inception = InceptionModule(dim, 32)  # 保持你原实现不改

        # 新增：超图卷积分支
        self.hg = HyperGraphConv(dim=dim, k=hg_k, metric=hg_metric, edge_drop=hg_edge_drop)
        self.hg_weight = float(hg_weight)

        # 门控融合：MoE vs Inception；再把 HG 作为第三分支注入
        self.fuse_moe_incep = FusionGate(dim)
        self.fuse_hg = nn.Parameter(torch.tensor(0.0))  # learnable scalar gate (初始化偏保守)

        self.act = nn.GELU()

    def forward(self, x, end_depth=False, remain_ratio=1.0):

        if remain_ratio < 1.0:
            x = self.norm1(x)
            attn_x_score = self.attention_head(x)

            left_patch_tokens = math.ceil(remain_ratio * x.shape[1])
            _, left_idx = torch.topk(attn_x_score, left_patch_tokens, dim=1, largest=True, sorted=True)

            compl_left_indx = coml_index(input_indx=left_idx.squeeze(-1), dim=x.shape[1])
            compl_left_indx = compl_left_indx.unsqueeze(2)

            sorted_left_idx, _ = torch.sort(left_idx, dim=1)
            left_index = sorted_left_idx.expand(-1, -1, x.shape[2])
            compl = compl_left_indx.to(left_index.device)

            non_topk = torch.gather(x * attn_x_score, dim=1, index=compl.expand(-1, -1, x.shape[2]))
            extra_token = torch.sum(non_topk, dim=1, keepdim=True)
            left_x = torch.gather(x * attn_x_score, dim=1, index=left_index)

            x = torch.cat([left_x, extra_token], dim=1)

            x2 = self.norm2(x)

            # Inception 分支
            incep_x = self.inception(x2.permute(0, 2, 1)).permute(0, 2, 1)

            # MoE 分支
            moe_x, moe_loss = self.moe(x2)

            # 先门控融合 MoE 和 Inception
            x = self.fuse_moe_incep(x, moe_x, incep_x)

            # HyperGraph 分支（第三分支注入）
            hg_x = self.hg(x2)
            x = x + self.hg_weight * torch.sigmoid(self.fuse_hg) * hg_x

        else:
            x = self.norm1(x)
            attn_x_score = self.attention_head(x)
            x = x * attn_x_score

            x2 = self.norm2(x)

            # Inception
            incep_x = self.inception(x2.permute(0, 2, 1)).permute(0, 2, 1)

            # MoE：你原来这里 moe_loss=0，我建议仍然用 MoE 让专家学起来
            moe_x, moe_loss = self.moe(x2)

            # 融合
            x = self.fuse_moe_incep(x, moe_x, incep_x)

            # HyperGraph 注入
            hg_x = self.hg(x2)
            x = x + self.hg_weight * torch.sigmoid(self.fuse_hg) * hg_x

        end_attn_x_score = None
        if end_depth:
            x = self.out_drop(x)
            end_attn_x_score = self.attention_head(x)

        return self.act(x), moe_loss, end_attn_x_score


# =========================================================

    ####################################
def add_pos_embed(x, pos_embed):
        """
        x: [B, N, C]
        pos_embed: [1, N0, C]
        return: [B, N, C]
        """
        B, N, C = x.shape
        N0 = pos_embed.shape[1]

        if N0 == N:
            pe = pos_embed
        else:
            # 用 1D 插值把 pos_embed 从 N0 -> N
            pe = F.interpolate(
                pos_embed.transpose(1, 2),  # [1, C, N0]
                size=N,
                mode="linear",
                align_corners=False
            ).transpose(1, 2)               # [1, N, C]
        return x + pe
    ##########################################
# 覆盖 SoftShapeNet：把超图参数传入每个 block
# =========================================================
class H-SER(nn.Module):
    def __init__(self, seq_len, shape_size, num_channels, emb_dim, sparse_rate, depth, num_classes,
                 num_experts=8, stride=2,
                 hg_k=8, hg_metric="cosine", hg_edge_drop=0.0, hg_weight=1.0):
        super().__init__()

        self.seq_len = seq_len
        self.shape_size = shape_size
        self.num_channels = num_channels
        self.emb_dim = emb_dim
        self.sparse_rate = sparse_rate
        self.depth = depth
        self.num_classes = num_classes

        # self.shape_embed = ShapeEmbedLayer(
        #     seq_len=self.seq_len, shape_size=self.shape_size,
        #     in_chans=self.num_channels, embed_dim=self.emb_dim, stride=stride
        # )

        ###################################
        self.shape_embed = MultiScaleShapeEmbed(
            seq_len=self.seq_len, shape_sizes=(8, 16, 32),
            in_chans=self.num_channels, embed_dim=self.emb_dim, stride=stride, fusion="concat"
        )
        # Multi-Scale Shape Embedding
        # self.shape_embed = MultiScaleShapeEmbedLayer(
        #     seq_len=self.seq_len, in_chans=self.num_channels, embed_dim=self.emb_dim, stride=stride, k_list=(8, 16, 32)
        # )

        ##########################################

        # 保持你原 attention head 不改（Sigmoid）
        self.attention_head = nn.Sequential(
            nn.Linear(self.emb_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        self.moe = MoE_Block(input_size=self.emb_dim, output_size=self.emb_dim,
                             num_experts=num_experts, hidden_size=self.emb_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.shape_embed.num_patches, self.emb_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=0.15)
        self.sparse_ratio_d = [x.item() for x in torch.linspace(0, self.sparse_rate, self.depth)]

        # 每层加入 HGNN
        self.shape_blocks = nn.ModuleList([
            SoftShapeNet_layer(
                dim=self.emb_dim, moe_nets=self.moe, atten_head=self.attention_head,
                hg_k=hg_k, hg_metric=hg_metric, hg_edge_drop=hg_edge_drop, hg_weight=hg_weight
            )
            for _ in range(self.depth)
        ])

        self.head = nn.Linear(self.emb_dim, self.num_classes)

        # init weights (保持你原逻辑)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            with torch.no_grad():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)



    def forward(self, x, num_epoch_i=100, warm_up_epoch=50, return_intermediates=False):

    # (a) Input Shape Embeddings
        x = self.shape_embed(x)                 # [B, N, C]
        # x = x + self.pos_embed                 # [B, N, C]
        x = add_pos_embed(x, self.pos_embed)
        x = self.pos_drop(x)                   # [B, N, C]
        emb_input = x                          # <-- (a)

        moe_loss = None
        end_attn_x_score = None
        emb_intra = None

        for d, shape_blk in enumerate(self.shape_blocks):
            depth_remain_ratio = 1.0 - self.sparse_ratio_d[d]
            if num_epoch_i < warm_up_epoch:
                depth_remain_ratio = 1.0

            judge_end = ((d + 1) == self.depth)

            x, _temp_mloss, end_attn_x_score = shape_blk(
                x, end_depth=judge_end, remain_ratio=depth_remain_ratio
            )

            # (b) Intra-Shape Embeddings: after first block
            if d == 0:
                emb_intra = x

            if moe_loss is None:
                moe_loss = _temp_mloss
            else:
                moe_loss = moe_loss + _temp_mloss

        # (c) Inter-Shape Embeddings: after last block
        emb_inter = x

        # (d) Output Shape Embeddings: pooled token embedding
        emb_output = x.mean(dim=1)  # [B, C]

        # classification head (保持你原来的逻辑)
        instance_logits = self.head(x)  # [B, N, num_classes]
        weighted_instance_logits = instance_logits * end_attn_x_score
        cls_logits = torch.mean(weighted_instance_logits, dim=1)  # [B, num_classes]

        if not return_intermediates:
            return cls_logits, moe_loss

        emb_dict = {
            "input": emb_input,     # [B, N, C]
            "intra": emb_intra,     # [B, N, C]
            "inter": emb_inter,     # [B, N, C]
            "output": emb_output,   # [B, C]
        }
        return cls_logits, moe_loss, emb_dict

