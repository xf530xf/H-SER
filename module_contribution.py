# module_contribution.py
# ------------------------------------------------------------
# 测量 H2SERNet 中每个子模块 (Module) 对最终损失 (Loss) 的贡献度。
#
# 思路：
#   每个模块的输出张量 y = f(x) 在反向传播时，会接收到 dL/dy。
#   |dL/dy| 越大，说明 Loss 对该模块输出越敏感 -> 该模块的贡献越大。
#
# 实现：
#   - 用 forward hook 包装每个目标子模块的 forward；
#   - 在 forward 输出张量上调用 tensor.register_hook(...)，
#     在反向传播时回调里记录 ||dL/dy||_2、L1 范数、均值绝对值；
#   - 如果一个模块被多次调用（例如共享的 attention_head / moe 在每个 block 都会被调），
#     用 call_idx 做唯一命名，最终在画图时按模块类型聚合。
#
# 画图：
#   一张矩形图（Bar Chart），按模块对损失的梯度 L2 范数排序。
# ------------------------------------------------------------

import os
import sys
import argparse
import json
from collections import defaultdict, OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from ts_utils import set_seed, build_loss, build_dataset, get_all_datasets
from data.preprocessing import normalize_per_series, fill_nan_value, normalize_train_val_test
from data.dataloader import UCRDataset
from data.shape_size_hyp import ucr_hyp_dict_shape_size
from models.H2SERModel import H2SERNet


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import matplotlib.font_manager as fm
_available_fonts = {f.name for f in fm.fontManager.ttflist}
_preferred_serif = next((f for f in ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"]
                         if f in _available_fonts or f == "serif"), "serif")
plt.rcParams['font.family'] = _preferred_serif
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'


# =========================================================
# Gradient tracker
# =========================================================
class ModuleGradTracker:
    """
    Wrap selected submodules' forward, register a hook on the *output tensor*
    so that the gradient flowing back into that output is captured.
    """

    def __init__(self):
        # records: list of dict per-call
        # {name, module_type, call_idx, grad_l2, grad_l1, grad_mean_abs, grad_max_abs, out_shape}
        self.records = []
        self._call_count = defaultdict(int)
        self._original_forwards = []  # [(module, original_forward)]

    # ------- public API -------
    def attach(self, module: torch.nn.Module, name: str, module_type: str = None):
        """Wrap module.forward to capture gradients on its output tensor."""
        if module_type is None:
            module_type = module.__class__.__name__

        original_forward = module.forward

        def wrapped_forward(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            self._register_hook_on_output(out, name, module_type)
            return out

        module.forward = wrapped_forward
        self._original_forwards.append((module, original_forward))

    def detach_all(self):
        for module, original_forward in self._original_forwards:
            module.forward = original_forward
        self._original_forwards.clear()

    def reset(self):
        self.records.clear()
        self._call_count.clear()

    # ------- internal -------
    def _register_hook_on_output(self, out, name, module_type):
        """out can be a Tensor or a tuple/list. We hook on the first leading Tensor."""
        tensor = self._find_main_tensor(out)
        if tensor is None or not tensor.requires_grad:
            return

        call_idx = self._call_count[name]
        self._call_count[name] += 1
        full_name = f"{name}#{call_idx}" if call_idx > 0 else name
        out_shape = tuple(tensor.shape)

        def grad_hook(grad):
            g = grad.detach()
            rec = {
                "name": full_name,
                "base_name": name,
                "module_type": module_type,
                "call_idx": call_idx,
                "grad_l2": float(g.norm(p=2).item()),
                "grad_l1": float(g.abs().sum().item()),
                "grad_mean_abs": float(g.abs().mean().item()),
                "grad_max_abs": float(g.abs().max().item()),
                "out_shape": out_shape,
            }
            self.records.append(rec)

        tensor.register_hook(grad_hook)

    @staticmethod
    def _find_main_tensor(out):
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, (list, tuple)):
            for item in out:
                if isinstance(item, torch.Tensor):
                    return item
        return None


# =========================================================
# Attach tracker to H2SERNet
# =========================================================
def attach_h2ser_modules(tracker: ModuleGradTracker, model: H2SERNet):
    """
    在 H2SERNet 的关键子模块上注册钩子。
    覆盖：
      - ShapeEmbed (multi-scale 1d-conv embedding)
      - 共享的 attention_head（按 call_idx 自动区分）
      - 共享的 MoE（按 call_idx 自动区分）
      - 每个 H2SERNet_layer 内的 Inception / HyperGraph / FusionGate / RMSNorm
      - 整个 block 的输出
      - 最终分类 head
    """
    tracker.attach(model.shape_embed, "ShapeEmbed", "MultiScaleShapeEmbed")
    tracker.attach(model.attention_head, "AttentionHead", "AttentionHead")
    tracker.attach(model.moe, "MoE", "MoE_Block")
    tracker.attach(model.head, "ClsHead", "Linear(head)")

    for d, blk in enumerate(model.shape_blocks):
        prefix = f"Block{d}"
        tracker.attach(blk.norm1, f"{prefix}.Norm1", "RMSNorm")
        tracker.attach(blk.norm2, f"{prefix}.Norm2", "RMSNorm")
        tracker.attach(blk.inception, f"{prefix}.Inception", "InceptionModule")
        tracker.attach(blk.hg, f"{prefix}.HyperGraph", "HyperGraphConv")
        tracker.attach(blk.fuse_moe_incep, f"{prefix}.FusionGate", "FusionGate")
        tracker.attach(blk, f"{prefix}.LayerOut", "H2SERNet_layer")


# =========================================================
# Aggregate / plot
# =========================================================
def aggregate_records(records, metric="grad_l2"):
    """
    将共享模块（多次调用）按 base_name 聚合（求和），返回 OrderedDict[name -> value]
    """
    agg = defaultdict(float)
    for r in records:
        agg[r["base_name"]] += r[metric]
    return OrderedDict(sorted(agg.items(), key=lambda kv: kv[1], reverse=True))


def plot_bar_contribution(agg, metric, save_path, title=None, log_scale=False):
    names = list(agg.keys())
    values = [agg[n] for n in names]

    cmap = plt.get_cmap("viridis")
    n = len(names)
    colors = [cmap(0.15 + 0.7 * (1.0 - i / max(1, n - 1))) for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * n + 4), 6), dpi=150)
    bars = ax.bar(range(n), values, color=colors, edgecolor="black", linewidth=0.6)

    vmax = max(values) if len(values) else 1.0
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + (vmax * 0.005 if not log_scale else 0),
            f"{v:.2e}",
            ha="center", va="bottom", fontsize=8, rotation=0,
        )

    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel(f"Gradient {metric}  (||dL/dy||)")
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title or f"Module Contribution to Loss  ({metric})")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] saved: {save_path}")


def plot_per_call_breakdown(records, metric, save_path, title=None):
    """
    按每次调用单独画一根柱子（用于看共享模块在不同 block 中的贡献差异）。
    """
    items = sorted(records, key=lambda r: r[metric], reverse=True)
    names = [r["name"] for r in items]
    values = [r[metric] for r in items]

    n = len(names)
    if n == 0:
        return

    cmap = plt.get_cmap("plasma")
    colors = [cmap(0.15 + 0.7 * (1.0 - i / max(1, n - 1))) for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(10, 0.45 * n + 4), 6), dpi=150)
    bars = ax.bar(range(n), values, color=colors, edgecolor="black", linewidth=0.6)

    vmax = max(values) if len(values) else 1.0
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + vmax * 0.005,
            f"{v:.2e}",
            ha="center", va="bottom", fontsize=7, rotation=0,
        )

    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(f"Gradient {metric}  (||dL/dy||)")
    ax.set_title(title or f"Per-call Module Contribution  ({metric})")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] saved: {save_path}")


# =========================================================
# Data preparation (mirrors train-net.py)
# =========================================================
def prepare_data(args, device):
    sum_dataset, sum_target, num_class = build_dataset(args)
    args.num_class = num_class
    args.moe_num_experts = num_class
    args.seq_len = sum_dataset.shape[1]

    args.shape_size = ucr_hyp_dict_shape_size.get(args.dataset, {}).get('shape_size', args.shape_size)
    args.shape_use_ratio = ucr_hyp_dict_shape_size.get(args.dataset, {}).get('shape_use_ratio', args.shape_use_ratio)
    args.shape_ratio = ucr_hyp_dict_shape_size.get(args.dataset, {}).get('shape_ratio', args.shape_ratio)

    if args.shape_use_ratio == 1:
        args.shape_size = max(1, int(args.seq_len * args.shape_ratio))
        if args.shape_size <= args.shape_stride:
            args.shape_stride = min(2, args.shape_size)
    if args.shape_stride > args.shape_size:
        args.shape_stride = args.shape_size

    train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets = get_all_datasets(
        sum_dataset, sum_target
    )

    fold = args.fold
    train_x = train_datasets[fold][:, :, np.newaxis]
    val_x = val_datasets[fold][:, :, np.newaxis]
    test_x = test_datasets[fold][:, :, np.newaxis]
    train_x, val_x, test_x = fill_nan_value(train_x, val_x, test_x)

    if args.normalize_way == 'single':
        train_x = normalize_per_series(train_x)
        test_x = normalize_per_series(test_x)
    else:
        train_x, val_x, test_x = normalize_train_val_test(train_x, val_x, test_x)

    train_set = UCRDataset(
        torch.from_numpy(train_x).type(torch.FloatTensor).to(device),
        torch.from_numpy(train_targets[fold]).type(torch.FloatTensor).to(device).to(torch.int64),
    )
    test_set = UCRDataset(
        torch.from_numpy(test_x).type(torch.FloatTensor).to(device),
        torch.from_numpy(test_targets[fold]).type(torch.FloatTensor).to(device).to(torch.int64),
    )

    bs = min(args.batch_size, len(train_set))
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, drop_last=False)
    return train_loader, test_loader


# =========================================================
# Optional: short warm-up training so gradients reflect a "real" model
# =========================================================
def warmup_train(model, loader, loss_fn, args, device):
    if args.warmup_steps <= 0:
        return
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    step = 0
    while step < args.warmup_steps:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            pred, moe_loss = model(x, num_epoch_i=step, warm_up_epoch=args.warm_up_epoch)
            loss = loss_fn(pred, y) + args.moeloss_rate * moe_loss
            loss.backward()
            optim.step()
            step += 1
            if step >= args.warmup_steps:
                break
    print(f"[WARMUP] done {step} steps")


# =========================================================
# Core: run analysis (average across multiple batches for stability)
# =========================================================
def run_contribution_analysis(model, loader, loss_fn, args, device):
    """
    在多个 batch 上平均梯度统计，得到稳定的模块贡献度。
    返回:
        agg_dict: { metric_name -> OrderedDict(base_name -> value) }
        all_records: list of per-call records (averaged across batches)
    """
    tracker = ModuleGradTracker()
    attach_h2ser_modules(tracker, model)

    # 累加每个 (name, call_idx) 的指标
    accum = defaultdict(lambda: defaultdict(float))
    accum_meta = {}
    n_batches = 0

    metrics_keys = ["grad_l2", "grad_l1", "grad_mean_abs", "grad_max_abs"]

    model.train()
    iterator = iter(loader)
    for b in range(args.num_batches):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        x, y = x.to(device), y.to(device)

        tracker.reset()
        model.zero_grad(set_to_none=True)

        pred, moe_loss = model(x, num_epoch_i=args.warm_up_epoch + 1, warm_up_epoch=args.warm_up_epoch)
        cls_loss = loss_fn(pred, y)
        total_loss = cls_loss + args.moeloss_rate * moe_loss
        total_loss.backward()

        for r in tracker.records:
            key = r["name"]
            for mk in metrics_keys:
                accum[key][mk] += r[mk]
            accum_meta[key] = {
                "base_name": r["base_name"],
                "module_type": r["module_type"],
                "call_idx": r["call_idx"],
                "out_shape": r["out_shape"],
            }
        n_batches += 1
        print(f"[BATCH {b+1}/{args.num_batches}] cls_loss={cls_loss.item():.4f} "
              f"moe_loss={float(moe_loss):.4f} hooks={len(tracker.records)}")

    tracker.detach_all()

    averaged_records = []
    for key, mdict in accum.items():
        rec = dict(accum_meta[key])
        rec["name"] = key
        for mk in metrics_keys:
            rec[mk] = mdict[mk] / max(1, n_batches)
        averaged_records.append(rec)

    agg_dict = {mk: aggregate_records(averaged_records, metric=mk) for mk in metrics_keys}
    return agg_dict, averaged_records


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='BME')
    parser.add_argument('--dataroot', type=str,
                        default=os.path.join(curPath, 'datasets', 'UCRArchive_2018'))
    parser.add_argument('--normalize_way', type=str, default='single')
    parser.add_argument('--fold', type=int, default=0)

    # model
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--sparse_rate', type=float, default=0.50)
    parser.add_argument('--shape_size', type=int, default=8)
    parser.add_argument('--shape_use_ratio', type=int, default=0)
    parser.add_argument('--shape_ratio', type=float, default=0.1)
    parser.add_argument('--shape_stride', type=int, default=4)
    parser.add_argument('--moe_num_experts', type=int, default=8)
    parser.add_argument('--warm_up_epoch', type=int, default=50)
    parser.add_argument('--moeloss_rate', type=float, default=0.001)
    parser.add_argument('--num_class', type=int, default=2)

    # training (warmup) and analysis
    parser.add_argument('--loss', type=str, choices=['cross_entropy'], default='cross_entropy')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Optional short training before analysis. 0 = analyze fresh model.')
    parser.add_argument('--num_batches', type=int, default=4,
                        help='Average gradient stats across N batches for stability.')
    parser.add_argument('--cuda', type=str, default='cuda:0')

    # output
    parser.add_argument('--save_dir', type=str,
                        default=os.path.join(curPath, 'results', 'module_contribution'))
    parser.add_argument('--metric', type=str, default='grad_l2',
                        choices=['grad_l2', 'grad_l1', 'grad_mean_abs', 'grad_max_abs'])
    parser.add_argument('--log_scale', type=int, default=0)

    args = parser.parse_args()
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    set_seed(args)

    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, test_loader = prepare_data(args, device)

    model = H2SERNet(
        seq_len=args.seq_len,
        shape_size=args.shape_size,
        num_channels=1,
        emb_dim=args.emb_dim,
        sparse_rate=args.sparse_rate,
        depth=args.depth,
        num_experts=args.moe_num_experts,
        num_classes=args.num_class,
        stride=args.shape_stride,
    ).to(device)

    loss_fn = build_loss(args).to(device)

    if args.warmup_steps > 0:
        warmup_train(model, train_loader, loss_fn, args, device)

    agg_dict, averaged_records = run_contribution_analysis(model, train_loader, loss_fn, args, device)

    tag = f"{args.dataset}_fold{args.fold}_warm{args.warmup_steps}_b{args.num_batches}"

    # main bar chart (aggregated, picked metric)
    main_path = os.path.join(args.save_dir, f"{tag}_bar_{args.metric}.png")
    plot_bar_contribution(
        agg_dict[args.metric], args.metric, main_path,
        title=f"[{args.dataset}] Module Contribution to Loss ({args.metric})",
        log_scale=bool(args.log_scale),
    )

    # also dump all 4 metrics
    for mk in ["grad_l2", "grad_l1", "grad_mean_abs", "grad_max_abs"]:
        if mk == args.metric:
            continue
        p = os.path.join(args.save_dir, f"{tag}_bar_{mk}.png")
        plot_bar_contribution(
            agg_dict[mk], mk, p,
            title=f"[{args.dataset}] Module Contribution to Loss ({mk})",
            log_scale=bool(args.log_scale),
        )

    # per-call breakdown for shared modules
    pc_path = os.path.join(args.save_dir, f"{tag}_per_call_{args.metric}.png")
    plot_per_call_breakdown(
        averaged_records, args.metric, pc_path,
        title=f"[{args.dataset}] Per-call Module Gradient ({args.metric})",
    )

    # save numeric data as json for later use
    json_path = os.path.join(args.save_dir, f"{tag}_records.json")
    payload = {
        "args": {k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v))
                 for k, v in vars(args).items()},
        "aggregated": {mk: dict(agg_dict[mk]) for mk in agg_dict},
        "per_call_records": averaged_records,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[JSON] saved: {json_path}")

    # console summary
    print("\n=== TOP modules by", args.metric, "===")
    for i, (n, v) in enumerate(agg_dict[args.metric].items()):
        print(f"  {i+1:2d}. {n:30s}  {v:.4e}")


if __name__ == "__main__":
    main()
