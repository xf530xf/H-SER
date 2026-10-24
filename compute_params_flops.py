# compute_params_flops.py
# ------------------------------------------------------------
# Compute parameter count (Params, M) and FLOPs (M / G) for the
# local H2SERNet model on a few representative UCR datasets.
#
# - Uses fvcore.nn.FlopCountAnalysis for FLOPs.
# - Uses model.parameters() for trainable param count.
# - Wraps the model so its forward only takes `x` (the underlying
#   model needs `num_epoch_i` / `warm_up_epoch` kwargs).
#
# Usage:
#   python compute_params_flops.py
#   python compute_params_flops.py --cuda cuda:0
# ------------------------------------------------------------

import os
import sys
import argparse
import warnings

import torch
import torch.nn as nn

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from models.H2SERModel import H2SERNet
from data.shape_size_hyp import ucr_hyp_dict_shape_size


# ---------------------------------------------------------------------------
# Datasets we benchmark on (small / medium / long sequence length).
# seq_len values are taken from the actual UCR archive.
# ---------------------------------------------------------------------------
BENCH_DATASETS = [
    {"name": "BME",          "seq_len": 128,  "num_class": 3},
    {"name": "CricketX",     "seq_len": 300,  "num_class": 12},
    {"name": "CinCECGTorso", "seq_len": 1639, "num_class": 4},
]


# ---------------------------------------------------------------------------
# Wrapper so fvcore can call forward(x) with no extra kwargs.
# ---------------------------------------------------------------------------
class _ForwardWrapper(nn.Module):
    def __init__(self, model: nn.Module, warm_up_epoch: int = 150):
        super().__init__()
        self.model = model
        self.warm_up_epoch = warm_up_epoch

    def forward(self, x):
        out = self.model(
            x,
            num_epoch_i=self.warm_up_epoch + 1,
            warm_up_epoch=self.warm_up_epoch,
        )
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def _resolve_shape_size(dataset_name: str, seq_len: int, default_stride: int = 4):
    """Mirror the shape_size resolution logic from train-net.py."""
    cfg = ucr_hyp_dict_shape_size[dataset_name]
    shape_size = cfg["shape_size"]
    use_ratio = cfg["shape_use_ratio"]
    ratio = cfg["shape_ratio"]
    stride = default_stride

    if use_ratio == 1:
        shape_size = int(seq_len * ratio)
        if shape_size <= stride:
            stride = min(2, shape_size)
    if stride > shape_size:
        stride = shape_size
    return shape_size, stride


def _build_model(seq_len: int, shape_size: int, stride: int, num_class: int,
                 emb_dim: int = 128, depth: int = 2, sparse_rate: float = 0.5):
    return H2SERNet(
        seq_len=seq_len,
        shape_size=shape_size,
        num_channels=1,
        emb_dim=emb_dim,
        sparse_rate=sparse_rate,
        depth=depth,
        num_experts=num_class,
        num_classes=num_class,
        stride=stride,
    )


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _count_flops(wrapper: nn.Module, x: torch.Tensor) -> int:
    """Single-sample (B=1) FLOPs as counted by fvcore."""
    from fvcore.nn import FlopCountAnalysis

    flops = FlopCountAnalysis(wrapper, x)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    flops.tracer_warnings("none")
    return int(flops.total())


def _format_params(n: int) -> str:
    return f"{n / 1e6:.3f} M ({n:,})"


def _format_flops(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.3f} G ({n / 1e6:.1f} M)"
    return f"{n / 1e6:.3f} M"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default="cpu",
                        help="Device for forward pass (e.g. cuda:0). Defaults to cpu.")
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--sparse_rate", type=float, default=0.5)
    parser.add_argument("--report", type=str,
                        default=os.path.join(curPath, "params_flops_report.md"))
    args = parser.parse_args()

    device = torch.device(args.cuda if (args.cuda.startswith("cuda")
                                        and torch.cuda.is_available()) else "cpu")
    print(f"[Info] Using device: {device}")
    print(f"[Info] emb_dim={args.emb_dim}, depth={args.depth}, "
          f"sparse_rate={args.sparse_rate}")
    print()

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ds in BENCH_DATASETS:
            name = ds["name"]
            seq_len = ds["seq_len"]
            num_class = ds["num_class"]
            shape_size, stride = _resolve_shape_size(name, seq_len)

            model = _build_model(
                seq_len=seq_len,
                shape_size=shape_size,
                stride=stride,
                num_class=num_class,
                emb_dim=args.emb_dim,
                depth=args.depth,
                sparse_rate=args.sparse_rate,
            ).to(device).eval()

            wrapper = _ForwardWrapper(model, warm_up_epoch=150).to(device).eval()

            x = torch.randn(1, 1, seq_len, device=device)

            # Forward once to make sure shapes line up.
            with torch.no_grad():
                _ = wrapper(x)

            params = _count_params(model)
            try:
                flops = _count_flops(wrapper, x)
                flops_str = _format_flops(flops)
            except Exception as e:
                flops = -1
                flops_str = f"FAILED ({type(e).__name__}: {e})"

            print(f"=== Dataset: {name} ===")
            print(f"  seq_len      : {seq_len}")
            print(f"  num_class    : {num_class}")
            print(f"  shape_size   : {shape_size} (stride={stride})")
            print(f"  num_patches  : {model.shape_embed.num_patches}")
            print(f"  Params       : {_format_params(params)}")
            print(f"  FLOPs (B=1)  : {flops_str}")
            print()

            rows.append({
                "dataset": name,
                "seq_len": seq_len,
                "num_class": num_class,
                "shape_size": shape_size,
                "stride": stride,
                "num_patches": model.shape_embed.num_patches,
                "params": params,
                "flops": flops,
            })

    # ------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------
    print("=" * 78)
    print("Summary (H2SERNet, single-sample forward)")
    print("=" * 78)
    print(f"{'Dataset':<16}{'seq_len':>9}{'#class':>8}{'shape':>7}"
          f"{'Params (M)':>14}{'FLOPs (M)':>14}")
    print("-" * 78)
    for r in rows:
        params_m = r["params"] / 1e6
        flops_m = r["flops"] / 1e6 if r["flops"] >= 0 else float("nan")
        print(f"{r['dataset']:<16}{r['seq_len']:>9}{r['num_class']:>8}"
              f"{r['shape_size']:>7}{params_m:>14.3f}{flops_m:>14.2f}")
    print("=" * 78)
    print()
    print("Caveat: fvcore counts conv / linear / matmul / einsum FLOPs, but not")
    print("the dynamic top-k / gather / scatter / softmax / index_add ops used")
    print("by SparseDispatcher in MoE_Block. These operations")
    print("are omitted; this is a partial operation count, not a full runtime estimate.")

    # ------------------------------------------------------------
    # Persist a markdown table snippet for the report.
    # ------------------------------------------------------------
    report_path = os.path.splitext(args.report)[0] + "_local.md"
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# H2SERNet — locally measured Params / FLOPs\n\n")
        f.write(f"Settings: emb_dim={args.emb_dim}, depth={args.depth}, "
                f"sparse_rate={args.sparse_rate}, batch=1, "
                f"device={device}\n\n")
        f.write("| Dataset | seq_len | #class | shape_size | "
                "Params (M) | FLOPs (M) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            params_m = r["params"] / 1e6
            flops_m = r["flops"] / 1e6 if r["flops"] >= 0 else float("nan")
            f.write(f"| {r['dataset']} | {r['seq_len']} | {r['num_class']} | "
                    f"{r['shape_size']} | {params_m:.3f} | {flops_m:.2f} |\n")
    print(f"\n[Info] Local-only report written to: "
          f"{report_path}")


if __name__ == "__main__":
    main()
