# train_h2ser_with_vis_all.py
# ------------------------------------------------------------
# Full training script (5-fold) + 2x2 embedding visualization (t-SNE)
# + Embedding featuremap visualization (token-token / token-channel heatmaps)
#
# Requires: your existing project modules (ts_utils, data/*, models/*)
# Model requirement: H2SERNet.forward(..., return_intermediates=True)
# should return a dict emb with keys: input/intra/inter/output
#   input/intra/inter: [B, N, C]
#   output: [B, C] or [B, 1, C]
# ------------------------------------------------------------

import os
import sys
import time
import argparse
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader

# Project path setup (keep your original style)
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from ts_utils import set_seed, build_loss, save_cls_result, build_dataset, get_all_datasets, evaluate_model
from data.preprocessing import normalize_per_series, fill_nan_value, normalize_train_val_test
from data.dataloader import UCRDataset
from data.shape_size_hyp import ucr_hyp_dict_shape_size
from models.H2SERModel import H2SERNet

import matplotlib.pyplot as plt

try:
    from sklearn.manifold import TSNE
    _HAS_TSNE = True
except Exception:
    _HAS_TSNE = False

# -------------------------
# Matplotlib global style
# -------------------------
plt.rcParams['font.family'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.frameon'] = True


# =========================
# t-SNE / PCA utilities
# =========================
def _to_2d(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    X: [M, D]
    return: [M, 2]
    """
    if _HAS_TSNE and X.shape[0] >= 10 and X.shape[1] >= 2:
        perplexity = min(30, max(5, (X.shape[0] - 1) // 3))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed
        )
        return tsne.fit_transform(X)

    # PCA fallback via SVD
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coordinates = U[:, :2] * S[:2]
    return np.pad(coordinates, ((0, 0), (0, 2 - coordinates.shape[1])))


@torch.no_grad()
def visualize_embeddings_2x2(model, loader, args, fold_idx=0, tag="test"):
    """
    Saves a 2x2 figure:
      (a) Input Shape Embeddings
      (b) Intra-Shape Embeddings
      (c) Inter-Shape Embeddings
      (d) Output Shape Embeddings
    """
    model.eval()

    embA_list, embB_list, embC_list, embD_list = [], [], [], []
    y_list = []

    for x, y in loader:
        out = model(
            x,
            num_epoch_i=args.warm_up_epoch,  # any value ok; we just want embeddings
            warm_up_epoch=args.warm_up_epoch,
            return_intermediates=True
        )

        emb = _extract_emb_dict(out)
        if emb is None:
            raise RuntimeError("Model did not return emb dict under return_intermediates=True")

        embA_list.append(emb["input"].detach().cpu())   # [B,N,C]
        embB_list.append(emb["intra"].detach().cpu())   # [B,N,C]
        embC_list.append(emb["inter"].detach().cpu())   # [B,N,C]
        embD_list.append(emb["output"].detach().cpu())  # [B,C] or [B,1,C]
        y_list.append(y.detach().cpu())

    A = torch.cat(embA_list, dim=0)
    B = torch.cat(embB_list, dim=0)
    C = torch.cat(embC_list, dim=0)
    D = torch.cat(embD_list, dim=0)
    Y = torch.cat(y_list, dim=0).numpy()

    rng = np.random.RandomState(args.random_seed)

    def pack_points(T: torch.Tensor, Y_np: np.ndarray):
        """
        T: [B,N,C] or [B,C] or [B,1,C]
        returns X:[M,C], y:[M]
        """
        if T.dim() == 3 and T.shape[1] == 1:
            # [B,1,C] -> [B,C]
            T = T.squeeze(1)

        if T.dim() == 2:
            X = T.numpy()
            y = Y_np
        else:
            Bn, N, Cc = T.shape
            if args.vis_per_token == 1:
                X = T.reshape(Bn * N, Cc).numpy()
                y = np.repeat(Y_np, N)
            else:
                X = T.mean(dim=1).numpy()
                y = Y_np

        if X.shape[0] > args.vis_max_points:
            idx = rng.choice(X.shape[0], size=args.vis_max_points, replace=False)
            X = X[idx]
            y = y[idx]
        return X, y

    XA, yA = pack_points(A, Y)
    XB, yB = pack_points(B, Y)
    XC, yC = pack_points(C, Y)
    XD, yD = pack_points(D, Y)

    ZA = _to_2d(XA, seed=args.random_seed)
    ZB = _to_2d(XB, seed=args.random_seed)
    ZC = _to_2d(XC, seed=args.random_seed)
    ZD = _to_2d(XD, seed=args.random_seed)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150)
    titles = [
        "(a) Input Shape Embeddings",
        "(b) +Share-Specialized Experts Routing",
        "(c) +Hypergraph&CGA",
        "(d) +All",
    ]
    Zs = [ZA, ZB, ZC, ZD]
    Ys = [yA, yB, yC, yD]

    for ax, title, Z, y in zip(axes.flatten(), titles, Zs, Ys):
        for c in np.unique(y):
            m = (y == c)
            ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.85, label=str(int(c)))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=8, frameon=True)

    plt.tight_layout()

    out_dir = os.path.join(args.save_dir, args.vis_outdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.dataset}_{tag}_fold{fold_idx}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[VIS-TSNE] saved: {out_path}")


# =========================
# Featuremap utilities
# =========================
def _extract_emb_dict(out):
    """
    Robustly extract the emb dict from model output.
    Supports:
      - (pred, moe_loss, emb_dict)
      - (pred, emb_dict)
      - emb_dict
    """
    emb_dict = None
    if isinstance(out, dict):
        emb_dict = out
    elif isinstance(out, (list, tuple)):
        for item in out[::-1]:
            if isinstance(item, dict):
                # check if it looks like emb dict
                if ("input" in item) or ("intra" in item) or ("inter" in item) or ("output" in item):
                    emb_dict = item
                    break
    return emb_dict


def _cosine_sim_matrix(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    X: [N, C] -> return S: [N, N] cosine similarity
    """
    X = X.float()
    X = X / (X.norm(dim=1, keepdim=True) + eps)
    return X @ X.t()


def _minmax_norm(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mn = M.min()
    mx = M.max()
    return (M - mn) / (mx - mn + eps)


@torch.no_grad()
def save_emb_featuremaps(
    emb: dict,
    args,
    fold_idx: int,
    epoch: int,
    tag: str = "test",
    sample_idx: int = 0,
    max_tokens: int = 256,
    save_token_channel: bool = True,
    save_token_token: bool = True,
    use_abs: bool = False,
):
    """
    emb: dict with keys: input/intra/inter/output
         input/intra/inter: [B, N, C]
         output: [B, C] or [B, 1, C]
    """
    out_dir = os.path.join(args.save_dir, args.vis_feature_outdir, args.dataset, f"fold{fold_idx}")
    os.makedirs(out_dir, exist_ok=True)

    stages = ["input", "intra", "inter", "output"]
    for st in stages:
        if st not in emb:
            continue

        T = emb[st].detach().cpu()

        # ---- normalize shape to [N, C] for heatmaps ----
        if T.dim() == 3:
            # [B, N, C] OR [B, 1, C]
            if T.shape[1] == 1:
                # [B,1,C] -> [B,C] -> [1,C]
                T = T[sample_idx].squeeze(0).unsqueeze(0) if T.shape[0] == 1 else T[sample_idx].squeeze(0).unsqueeze(0)
            else:
                T = T[sample_idx]  # [N, C]
        elif T.dim() == 2:
            # [B, C] -> [1, C]
            T = T[sample_idx].unsqueeze(0)
        elif T.dim() == 1:
            # [C] -> [1, C]
            T = T.unsqueeze(0)
        else:
            continue

        if T.shape[0] > max_tokens:
            T = T[:max_tokens, :]

        T_plot = T.abs() if use_abs else T

        # 1) Token-Channel heatmap: [N, C]
        if save_token_channel:
            A = _minmax_norm(T_plot)
            plt.figure(figsize=(6.5, 4.5), dpi=150)
            plt.imshow(A.numpy(), aspect="auto")
            plt.colorbar(fraction=0.046, pad=0.04)
            # plt.title(f"{args.dataset}")
            plt.xlabel("Channel (C)")
            plt.ylabel("Token index (N)")
            plt.tight_layout()
            fp = os.path.join(out_dir, f"ep{epoch}_{tag}_{st}_token-channel.png")
            plt.savefig(fp, bbox_inches="tight")
            plt.close()

        # 2) Token-Token similarity heatmap: [N, N]
        if save_token_token:
            if T_plot.shape[0] <= 1:
                # N=1 no meaningful token-token map
                continue
            S = _cosine_sim_matrix(T_plot)    # [-1,1]
            S01 = (S + 1.0) / 2.0             # [0,1]
            plt.figure(figsize=(5.2, 5.0), dpi=150)
            plt.imshow(S01.numpy(), aspect="equal")
            plt.colorbar(fraction=0.046, pad=0.04)
            # plt.title(f"{args.dataset}")
            plt.xlabel("Token index (N)")
            plt.ylabel("Token index (N)")
            plt.tight_layout()
            fp = os.path.join(out_dir, f"ep{epoch}_{tag}_{st}_token-token.png")
            plt.savefig(fp, bbox_inches="tight")
            plt.close()

    print(f"[VIS-FM] saved featuremaps at ep{epoch} -> {out_dir}")


# =========================
# Main training
# =========================
def main():
    parser = argparse.ArgumentParser()

    # Base setup
    parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed')

    # Dataset setup
    parser.add_argument('--dataset', type=str, default='BME', help='dataset(in ucr)')
    parser.add_argument('--dataroot', type=str, default=os.path.join(curPath, 'datasets', 'UCRArchive_2018'),
                        help='path of UCR folder')
    parser.add_argument('--num_class', type=int, default=2, help='number of class (will be overwritten by build_dataset)')
    parser.add_argument('--normalize_way', type=str, default='single', help='single or train_set')
    parser.add_argument('--input_size', type=int, default=1, help='input_size')

    # Model setup
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--sparse_rate', type=float, default=0.50, help='0.1, 0.3, or 0.7')
    parser.add_argument('--shape_size', type=int, default=8, help='(will be overwritten by hyp dict)')
    parser.add_argument('--shape_use_ratio', type=int, default=0, help='0 is False, 1 is True')
    parser.add_argument('--shape_ratio', type=float, default=0.1)
    parser.add_argument('--shape_stride', type=int, default=4, help='the patch stride')
    parser.add_argument('--moe_num_experts', type=int, default=8)
    parser.add_argument('--warm_up_epoch', type=int, default=150, help='50, 100 or 200')
    parser.add_argument('--moeloss_rate', type=float, default=0.001, help='0.01 or 0.001')

    # Training setup
    parser.add_argument('--loss', type=str, choices=['cross_entropy'], default='cross_entropy', help='loss function')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--use_large_batch', type=int, default=1, help='1 is True, 0 is False')
    parser.add_argument('--epoch', type=int, default=500, help='training epoch')
    parser.add_argument('--cuda', type=str, default='cuda:0')

    # Result setup
    parser.add_argument('--save_dir', type=str, default=os.path.join(curPath, 'results'))
    parser.add_argument('--save_csv_name', type=str, default='h2ser_five_fold_split')

    # Embedding scatter visualization setup
    parser.add_argument('--vis_embeddings', type=int, default=1, help='1: save 2x2 embedding visualization (t-SNE)')
    parser.add_argument('--vis_fold', type=int, default=4, help='which fold to visualize')
    parser.add_argument('--vis_max_points', type=int, default=1200, help='subsample points for t-SNE scatter')
    parser.add_argument('--vis_per_token', type=int, default=1, help='1: token-level; 0: pooled for a/b/c')
    parser.add_argument('--vis_outdir', type=str, default='vis_embeddings', help='subdir under save_dir')

    # Featuremap visualization setup
    parser.add_argument('--vis_featuremaps', type=int, default=1, help='1: save emb featuremap heatmaps during training')
    parser.add_argument('--vis_feature_epochs', type=str, default='20,50,80', help='comma-separated epochs to dump heatmaps')
    parser.add_argument('--vis_feature_outdir', type=str, default='vis_featuremaps', help='subdir under save_dir')
    parser.add_argument('--vis_feature_sample_idx', type=int, default=0, help='which sample in a batch to visualize')
    parser.add_argument('--vis_feature_max_tokens', type=int, default=256, help='max tokens shown in token-token maps')

    args = parser.parse_args()
    if args.batch_size < 1 or args.epoch < 1 or args.shape_stride < 1 or args.vis_max_points < 1:
        parser.error('batch_size, epoch, shape_stride, and vis_max_points must be positive')
    args.vis_feature_epochs = [int(x) for x in args.vis_feature_epochs.split(",") if x.strip() != ""]

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    set_seed(args)

    # Build dataset (your existing util)
    sum_dataset, sum_target, num_class = build_dataset(args)

    args.num_class = num_class
    args.moe_num_experts = num_class
    args.seq_len = sum_dataset.shape[1]

    # Hyp override for shape params
    args.shape_size = ucr_hyp_dict_shape_size.get(args.dataset, {}).get('shape_size', args.shape_size)
    args.shape_use_ratio = ucr_hyp_dict_shape_size.get(args.dataset, {}).get('shape_use_ratio', args.shape_use_ratio)
    args.shape_ratio = ucr_hyp_dict_shape_size.get(args.dataset, {}).get('shape_ratio', args.shape_ratio)

    print("Training dataset = ", args.dataset, ", shape_size = ", args.shape_size)

    if args.shape_use_ratio == 1:
        args.shape_size = max(1, int(args.seq_len * args.shape_ratio))
        if args.shape_size <= args.shape_stride:
            args.shape_stride = min(2, args.shape_size)

    if args.shape_stride > args.shape_size:
        args.shape_stride = args.shape_size

    # 5-fold split from your util
    train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets = get_all_datasets(
        sum_dataset, sum_target
    )

    if args.use_large_batch == 1:
        args.batch_size = 512
        if train_datasets[0].shape[0] < args.batch_size:
            args.batch_size = train_datasets[0].shape[0]

    # Loss
    loss_fn = build_loss(args).to(device)

    # Model
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

    model_init_state = deepcopy(model.state_dict())

    # Optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                     lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    optimizer_init_state = deepcopy(optimizer.state_dict())

    test_accuracies = []
    train_time = 0.0

    for i, train_dataset in enumerate(train_datasets):
        t0 = time.time()
        model.load_state_dict(model_init_state)
        optimizer.load_state_dict(deepcopy(optimizer_init_state))
        best_model_state = None

        print(f'{i} fold start training and evaluate')

        train_target = train_targets[i]
        val_dataset = val_datasets[i]
        val_target = val_targets[i]
        test_dataset = test_datasets[i]
        test_target = test_targets[i]

        # add channel dim: [B, T] -> [B, T, 1]
        train_dataset = train_dataset[:, :, np.newaxis]
        val_dataset = val_dataset[:, :, np.newaxis]
        test_dataset = test_dataset[:, :, np.newaxis]

        # fill NaNs
        train_dataset, val_dataset, test_dataset = fill_nan_value(train_dataset, val_dataset, test_dataset)

        # normalization
        if args.normalize_way == 'single':
            train_dataset = normalize_per_series(train_dataset)
            val_dataset = normalize_per_series(val_dataset)
            test_dataset = normalize_per_series(test_dataset)
        else:
            train_dataset, val_dataset, test_dataset = normalize_train_val_test(train_dataset, val_dataset, test_dataset)

        # to tensor and to device
        train_set = UCRDataset(
            torch.from_numpy(train_dataset).type(torch.FloatTensor).to(device),
            torch.from_numpy(train_target).type(torch.FloatTensor).to(device).to(torch.int64)
        )
        val_set = UCRDataset(
            torch.from_numpy(val_dataset).type(torch.FloatTensor).to(device),
            torch.from_numpy(val_target).type(torch.FloatTensor).to(device).to(torch.int64)
        )
        test_set = UCRDataset(
            torch.from_numpy(test_dataset).type(torch.FloatTensor).to(device),
            torch.from_numpy(test_target).type(torch.FloatTensor).to(device).to(torch.int64)
        )

        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0, drop_last=False)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0, drop_last=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0, drop_last=False)

        last_loss = float('inf')
        stop_count = 0
        increase_count = 0
        min_val_loss = float('inf')
        test_accuracy = 0.0

        for epoch in range(args.epoch):
            if stop_count == 80 or increase_count == 80:
                print(f'model convergent at epoch {epoch}, early stopping')
                break

            epoch_train_loss = 0.0
            num_iterations = 0

            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()

                pred, moe_loss = model(x, num_epoch_i=epoch, warm_up_epoch=args.warm_up_epoch)
                step_loss = loss_fn(pred, y)
                step_loss = step_loss + args.moeloss_rate * moe_loss
                step_loss.backward()
                optimizer.step()

                epoch_train_loss += float(step_loss.item())
                num_iterations += 1

            epoch_train_loss = epoch_train_loss / max(1, num_iterations)

            model.eval()
            val_loss, val_accu = evaluate_model(val_loader, model, loss_fn, num_epoch_i=epoch, warm_up_epoch=args.warm_up_epoch)

            if min_val_loss > val_loss:
                min_val_loss = val_loss
                best_model_state = deepcopy(model.state_dict())
                test_loss, test_accuracy = evaluate_model(test_loader, model, loss_fn, num_epoch_i=epoch, warm_up_epoch=args.warm_up_epoch)

            if (epoch > args.warm_up_epoch) and (abs(last_loss - val_loss) <= 1e-4):
                stop_count += 1
            else:
                stop_count = 0

            if (epoch > args.warm_up_epoch) and (val_loss > last_loss):
                increase_count += 1
            else:
                increase_count = 0

            last_loss = val_loss

            if epoch % 100 == 0:
                print(f"epoch : {epoch}, train loss: {epoch_train_loss:.4f}, test_accuracy: {test_accuracy:.4f}")

            # -------------------------------
            # Featuremap dump at specific epochs
            # -------------------------------
            if (args.vis_featuremaps == 1) and (i == args.vis_fold) and (epoch in args.vis_feature_epochs):
                try:
                    model.eval()
                    x_vis, y_vis = next(iter(test_loader))  # test set visualization
                    out = model(
                        x_vis,
                        num_epoch_i=epoch,
                        warm_up_epoch=args.warm_up_epoch,
                        return_intermediates=True
                    )
                    emb_dict = _extract_emb_dict(out)
                    if emb_dict is None:
                        print(f"[VIS-FM] No emb_dict returned at epoch={epoch}")
                    else:
                        save_emb_featuremaps(
                            emb=emb_dict,
                            args=args,
                            fold_idx=i,
                            epoch=epoch,
                            tag="test",
                            sample_idx=args.vis_feature_sample_idx,
                            max_tokens=args.vis_feature_max_tokens,
                            save_token_channel=True,
                            save_token_token=True,
                            use_abs=False
                        )
                except Exception as e:
                    print(f"[VIS-FM] failed at fold={i}, epoch={epoch}: {repr(e)}")

        if best_model_state is None:
            raise RuntimeError("No finite validation loss was observed; check the input data")
        model.load_state_dict(best_model_state)
        test_accuracies.append(test_accuracy)
        fold_time = time.time() - t0
        train_time += fold_time
        print(f'{i} fold finish training. time={fold_time:.2f}s best_test_acc={test_accuracy:.4f}')

        # -------------------------------
        # 2x2 embedding scatter for selected fold (after training)
        # -------------------------------
        if args.vis_embeddings == 1 and i == args.vis_fold:
            try:
                visualize_embeddings_2x2(model, test_loader, args, fold_idx=i, tag="test")
            except Exception as e:
                print(f"[VIS-TSNE] failed on fold {i}: {repr(e)}")

    test_accuracies = torch.Tensor(test_accuracies).cpu().numpy()
    print("Training end. mean_test_acc: {:.4f}, training time {:.4f}".format(np.mean(test_accuracies), train_time))

    save_cls_result(args, np.mean(test_accuracies), train_time)
    print('Done!')


if __name__ == '__main__':
    main()
