import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


# -----------------------------
# Basic blocks (no fastai deps)
# -----------------------------
BN1d = nn.InstanceNorm1d


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, xs):
        return torch.cat(xs, dim=self.dim)

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim})"


def conv1d_same(ni, nf, ks, stride=1, dilation=1, bias=True):
    # "same" padding for stride=1 is exact; for stride>1 it is approximate but stable
    pad = ((ks - 1) // 2) * dilation
    return nn.Conv1d(ni, nf, ks, stride=stride, padding=pad, dilation=dilation, bias=bias)


class InceptionModule(nn.Module):
    def __init__(self, ni, nf, ks=40, bottleneck=True):
        super().__init__()
        ks_list = [ks // (2 ** i) for i in range(3)]
        ks_list = [k if k % 2 != 0 else k - 1 for k in ks_list]  # ensure odd

        self.use_bottleneck = bottleneck and (ni > 1)
        self.bottleneck = nn.Conv1d(ni, nf, kernel_size=1, bias=False) if self.use_bottleneck else nn.Identity()

        in_c = nf if self.use_bottleneck else ni
        self.convs = nn.ModuleList([conv1d_same(in_c, nf, k, bias=False) for k in ks_list])

        self.maxconvpool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(ni, nf, kernel_size=1, bias=False),
        )

        self.concat = Concat(dim=1)
        self.bn = BN1d(nf * 4)
        self.out_proj = nn.Identity() if nf * 4 == ni else nn.Conv1d(nf * 4, ni, 1)
        self.act = nn.GELU()

    def forward(self, x):
        inp = x
        x1 = self.bottleneck(inp)
        xs = [c(x1) for c in self.convs] + [self.maxconvpool(inp)]
        x = self.concat(xs)
        if x.shape[-1] > 1:
            x = self.bn(x)
        return self.out_proj(self.act(x))


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # normalize over last dim
        denom = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / denom) * self.gamma * self.scale


# -----------------------------
# MoE (same logic as yours)
# -----------------------------
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
        inp_exp = inp[self._batch_index]
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = stitched.new_zeros(self._gates.size(0), expert_out[-1].size(1))
        combined = zeros.index_add(0, self._batch_index, stitched)
        return combined

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE_Block(nn.Module):
    def __init__(self, input_size, output_size, num_experts, hidden_size, k=1):
        super().__init__()
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k

        self.experts = nn.ModuleList([MLP(input_size, output_size, hidden_size) for _ in range(num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.rmsnorm = RMSNorm(dim=output_size)
        self.act = nn.GELU()
        self.softmax = nn.Softmax(dim=1)

        assert self.k <= self.num_experts

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return x.new_zeros((), dtype=torch.float32)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def top_k_gating(self, x):
        logits = self.softmax(x @ self.w_gate)  # [B, E]
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)

        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        x_flat = x.reshape(B * N, C)

        gates, load = self.top_k_gating(x_flat)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x_flat)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]

        y = x_flat + dispatcher.combine(expert_outputs)
        y = self.rmsnorm(y.view(B, N, C))
        return self.act(y), loss


# -----------------------------
# Innovation 1: Hypergraph Conv
# -----------------------------
class HypergraphConv(nn.Module):
    """
    Dynamic kNN hyperedge construction:
      Each token i defines a hyperedge e_i connecting its kNN neighbors (incl. itself).
    Then apply a standard hypergraph propagation:
      X' = Dv^{-1/2} H De^{-1} H^T Dv^{-1/2} X W
    We use soft incidence weights derived from distances for stability.
    """
    def __init__(self, dim, k=8, proj_dim=None, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.k = k
        self.proj = nn.Linear(dim, proj_dim or dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm = RMSNorm(dim)

    def _knn_indices(self, z):
        # z: (B, N, Dp)
        # pairwise distance: (B, N, N)
        # Use squared L2 via (a-b)^2 = a^2 + b^2 - 2ab
        B, N, D = z.shape
        z2 = (z * z).sum(-1, keepdim=True)  # (B, N, 1)
        dist = (z2 + z2.transpose(1, 2) - 2.0 * (z @ z.transpose(1, 2))).clamp_min(0)  # (B, N, N)
        # topk smallest distances
        k = min(self.k, N)
        idx = dist.topk(k=k, dim=-1, largest=False).indices  # (B, N, k)
        return idx, dist

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        z = self.proj(x)  # (B, N, Dp)

        idx, dist = self._knn_indices(z)  # idx: (B, N, k)

        # Build soft incidence H: (B, N, E) with E=N hyperedges (one per center token)
        # For hyperedge e=j, connect nodes in idx[:, j, :]
        H = x.new_zeros((B, N, N))  # (B, node, edge)
        # soft weights from distances: w = exp(-dist / tau)
        tau = 1.0
        # gather dist for neighbors
        dist_n = torch.gather(dist, dim=2, index=idx)  # (B, N, k)
        w = torch.exp(-dist_n / tau)  # (B, N, k)

        # scatter into H: for each edge j, set H[neighbor, j] = weight
        # Vectorized scatter:
        # edges are j along dim=1 in idx; we want to write into H[:, neighbor, edge=j]
        edge_ids = torch.arange(N, device=x.device).view(1, N, 1).expand(B, N, idx.size(-1))  # (B,N,k)
        # Assign each neighbor weight to its actual hyperedge.
        b = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, idx.size(-1))
        nbh = idx
        e = edge_ids
        H[b, nbh, e] = w

        # Degrees
        # node degree: Dv (B,N)
        Dv = H.sum(dim=2) + 1e-6
        # edge degree: De (B,E)
        De = H.sum(dim=1) + 1e-6

        Dv_inv_sqrt = torch.rsqrt(Dv).unsqueeze(-1)  # (B,N,1)
        De_inv = (1.0 / De).unsqueeze(1)            # (B,1,E)

        # Hypergraph propagation: P = Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}
        # Step1: A = Dv^{-1/2} * H  -> (B,N,E)
        A = H * Dv_inv_sqrt
        # Step2: Bm = A * De^{-1}   -> (B,N,E)
        Bm = A * De_inv
        # Step3: S = Bm @ A^T       -> (B,N,N)
        S = Bm @ A.transpose(1, 2)

        x_hat = S @ x  # (B,N,C)
        x_hat = self.out(x_hat)
        x_hat = self.drop(x_hat)
        return self.norm(x + x_hat)


# -----------------------------
# Innovation 2: Frequency-domain Spectral Gating
# -----------------------------
class SpectralGating(nn.Module):
    """
    Apply rFFT along token dimension N and learn frequency-wise gates.
    x: (B, N, C)
    """
    def __init__(self, dim, max_len=512, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.drop = nn.Dropout(dropout)
        self.norm = RMSNorm(dim)
        # gates for real FFT bins up to max_len//2+1
        self.gate = nn.Parameter(torch.ones(1, max_len // 2 + 1, 1))

    def forward(self, x):
        B, N, C = x.shape
        # rfft over N
        Xf = torch.fft.rfft(x, dim=1)  # (B, Nf, C) complex
        Nf = Xf.size(1)
        g = self.gate[:, :Nf, :]       # (1, Nf, 1)
        if Nf > self.gate.shape[1]:
            g = F.interpolate(self.gate.transpose(1, 2), size=Nf,
                              mode='linear', align_corners=False).transpose(1, 2)
        Xf = Xf * g                    # amplitude gating (complex scaling)
        y = torch.fft.irfft(Xf, n=N, dim=1)  # (B, N, C)
        y = self.drop(y)
        return self.norm(x + y)


# -----------------------------
# Patch embedding
# -----------------------------
class ShapeEmbedLayer(nn.Module):
    def __init__(self, seq_len, shape_size=8, in_chans=1, embed_dim=128, stride=4):
        super().__init__()
        if min(seq_len, shape_size, stride) < 1:
            raise ValueError('Sequence length, shape size, and stride must be positive')
        shape_size = min(shape_size, seq_len)
        num_patches = (seq_len - shape_size) // stride + 1
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=shape_size, stride=stride)

    def forward(self, x):
        # x: (B, C, L)
        x_out = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)
        return x_out


def coml_index(input_indx, dim):
    # input_indx: (B, K)
    full_idx = torch.arange(dim, device=input_indx.device)
    mask = torch.ones(input_indx.size(0), dim, dtype=torch.bool, device=input_indx.device)
    for i in range(input_indx.size(0)):
        mask[i, input_indx[i]] = False
    complement_idx = torch.stack([full_idx[mask[i]] for i in range(input_indx.size(0))])
    return complement_idx


# -----------------------------
# Main layer: keep your logic + add 2 modules
# -----------------------------
class H2SERNet_layer(nn.Module):
    def __init__(self, dim, moe_nets, atten_head, hg_k=8, max_tokens=512):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        self.attention_head = atten_head
        self.out_drop = nn.Dropout(p=0.15)

        self.moe = moe_nets
        self.inception = InceptionModule(dim, 32)

        # Innovation 1: Hypergraph
        self.hypergraph = HypergraphConv(dim=dim, k=hg_k, dropout=0.1)

        # Innovation 2: Spectral gating
        self.spectral = SpectralGating(dim=dim, max_len=max_tokens, dropout=0.1)

        self.act = nn.GELU()

    def forward(self, x, end_depth=False, remain_ratio=1.0):
        # x: (B, N, C)
        moe_loss = 0.0

        if remain_ratio < 1.0:
            x = self.norm1(x)
            attn_x_score = self.attention_head(x)  # (B,N,1)

            left_patch_tokens = math.ceil(remain_ratio * x.shape[1])
            _, left_idx = torch.topk(attn_x_score, left_patch_tokens, dim=1, largest=True, sorted=True)

            compl_left_indx = coml_index(input_indx=left_idx.squeeze(-1), dim=x.shape[1]).unsqueeze(2)

            sorted_left_idx, _ = torch.sort(left_idx, dim=1)
            left_index = sorted_left_idx.expand(-1, -1, x.shape[2])

            non_topk = torch.gather(x * attn_x_score, dim=1, index=compl_left_indx.expand(-1, -1, x.shape[2]))
            extra_token = torch.sum(non_topk, dim=1, keepdim=True)  # (B,1,C)

            left_x = torch.gather(x * attn_x_score, dim=1, index=left_index)  # (B,K,C)
            x = torch.cat([left_x, extra_token], dim=1)  # (B,K+1,C)

            # --- Innovation 1: hypergraph on selected tokens
            x = self.hypergraph(x)

            # Inception branch
            incep_x = self.inception(self.norm2(x).permute(0, 2, 1)).permute(0, 2, 1)

            # MoE branch
            temp_moe_x, moe_loss = self.moe(self.norm2(x))

            # --- Innovation 2: spectral gating on tokens
            x = self.spectral(x)

            x = x + temp_moe_x + incep_x
        else:
            x = self.norm1(x)
            attn_x_score = self.attention_head(x)
            x = x * attn_x_score

            # --- Innovation 1: hypergraph on full tokens
            x = self.hypergraph(x)

            incep_x = self.inception(self.norm2(x).permute(0, 2, 1)).permute(0, 2, 1)

            # --- Innovation 2: spectral gating on full tokens
            x = self.spectral(x)

            x = x + incep_x

        end_attn_x_score = None
        if end_depth:
            x = self.out_drop(x)
            end_attn_x_score = self.attention_head(x)

        return self.act(x), moe_loss, end_attn_x_score


# -----------------------------
# Full model
# -----------------------------
class H2SERNet(nn.Module):
    def __init__(
        self,
        seq_len,
        shape_size,
        num_channels,
        emb_dim,
        sparse_rate,
        depth,
        num_classes,
        num_experts=8,
        stride=2,
        hg_k=8,
    ):
        super().__init__()
        if min(seq_len, shape_size, num_channels, emb_dim, depth, num_classes, num_experts, stride) < 1:
            raise ValueError('Model dimensions, depth, number of experts, and stride must be positive')
        if not 0.0 <= sparse_rate <= 1.0 or hg_k < 1:
            raise ValueError('sparse_rate must be between 0 and 1; hg_k must be positive')
        self.seq_len = seq_len
        self.shape_size = shape_size
        self.num_channels = num_channels
        self.emb_dim = emb_dim
        self.sparse_rate = sparse_rate
        self.depth = depth
        self.num_classes = num_classes

        self.shape_embed = ShapeEmbedLayer(
            seq_len=self.seq_len,
            shape_size=self.shape_size,
            in_chans=self.num_channels,
            embed_dim=self.emb_dim,
            stride=stride,
        )

        self.attention_head = nn.Sequential(
            nn.Linear(self.emb_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        self.moe = MoE_Block(
            input_size=self.emb_dim,
            output_size=self.emb_dim,
            num_experts=num_experts,
            hidden_size=self.emb_dim,
            k=1,
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.shape_embed.num_patches, self.emb_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=0.15)

        self.sparse_ratio_d = [x.item() for x in torch.linspace(0, self.sparse_rate, self.depth)]

        max_tokens = self.shape_embed.num_patches + 1  # +1 because sparse branch may add extra_token
        self.shape_blocks = nn.ModuleList(
            [
                H2SERNet_layer(
                    dim=self.emb_dim,
                    moe_nets=self.moe,
                    atten_head=self.attention_head,
                    hg_k=hg_k,
                    max_tokens=max_tokens,
                )
                for _ in range(self.depth)
            ]
        )

        self.head = nn.Linear(self.emb_dim, self.num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm,)):
            with torch.no_grad():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)

    def forward(self, x, num_epoch_i=100, warm_up_epoch=50):
        # x: (B, C, L)
        x = self.shape_embed(x)  # (B, N, D)

        # positional
        x = x + self.pos_embed
        x = self.pos_drop(x)

        moe_loss = None
        end_attn_x_score = None

        for d, shape_blk in enumerate(self.shape_blocks):
            depth_remain_ratio = 1.0 - self.sparse_ratio_d[d]
            if num_epoch_i < warm_up_epoch:
                depth_remain_ratio = 1.0

            judge_end = (d + 1) == self.depth
            x, temp_mloss, end_attn_x_score = shape_blk(x, end_depth=judge_end, remain_ratio=depth_remain_ratio)

            moe_loss = temp_mloss if moe_loss is None else (moe_loss + temp_mloss)

        instance_logits = self.head(x)  # (B, N, num_classes)
        weighted_instance_logits = instance_logits * end_attn_x_score  # (B, N, num_classes)
        cls_logits = torch.mean(weighted_instance_logits, dim=1)  # (B, num_classes)

        return cls_logits, moe_loss


# -----------------------------
# Quick sanity test
# -----------------------------
if __name__ == "__main__":
    B, C, L = 2, 1, 256
    num_classes = 5

    model = H2SERNet(
        seq_len=L,
        shape_size=16,
        num_channels=C,
        emb_dim=128,
        sparse_rate=0.5,
        depth=4,
        num_classes=num_classes,
        num_experts=8,
        stride=4,
        hg_k=8,
    )

    x = torch.randn(B, C, L)
    y, mloss = model(x, num_epoch_i=10, warm_up_epoch=50)
    print("logits:", y.shape, "moe_loss:", float(mloss) if torch.is_tensor(mloss) else mloss)
