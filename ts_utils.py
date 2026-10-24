import random

import numpy as np
import torch

import os
import pandas as pd

from data.preprocessing import load_data, transfer_labels, k_fold
from data.data_provideruea import TSC_multivariate_data_loader, fill_out_with_Nan
from models.loss import cross_entropy, reconstruction_loss
from sklearn.metrics import accuracy_score


def set_seed(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)


def build_dataset(args):
    sum_dataset, sum_target, num_classes = load_data(args.dataroot, args.dataset)
    sum_target = transfer_labels(sum_target)
    return sum_dataset, sum_target, num_classes


def build_uea_dataset(args):
    train_x, train_y, test_x, test_y = TSC_multivariate_data_loader(args.dataroot, args.dataset)
    max_length = max(train_x.shape[-1], test_x.shape[-1])
    data = np.concatenate([
        fill_out_with_Nan(train_x, max_length),
        fill_out_with_Nan(test_x, max_length),
    ]).transpose(0, 2, 1).astype(np.float32)
    labels = transfer_labels(np.concatenate([train_y, test_y]))
    return data, labels, len(np.unique(labels))


def build_loss(args):
    if args.loss == 'cross_entropy':
        return cross_entropy()
    elif args.loss == 'reconstruction':
        return reconstruction_loss()
    raise ValueError(f'Unknown loss: {args.loss}')


def get_all_datasets(data, target):
    return k_fold(data, target)


def evaluate_model(val_loader, model, loss, **forward_kwargs):
    val_loss = 0
    val_pred_labels = []
    real_labels = []

    sum_len = 0
    for data, target in val_loader:
        with torch.no_grad():
            val_pred, _ = model(data, **forward_kwargs)
            batch_loss = loss(val_pred, target)
            if getattr(loss, 'reduction', 'mean') == 'sum':
                val_loss += batch_loss.item()
            else:
                val_loss += batch_loss.mean().item() * len(target)
            sum_len = sum_len + len(target)
            val_pred_labels.append(torch.argmax(val_pred.data, axis=1).cpu().numpy())
            real_labels.append(target.cpu().numpy())

    if sum_len == 0:
        raise ValueError('Cannot evaluate an empty data loader')
    val_pred_labels = np.concatenate(val_pred_labels)
    real_labels = np.concatenate(real_labels)

    return val_loss / sum_len, accuracy_score(real_labels, val_pred_labels)


def save_cls_result(args, mean_accu, train_time):
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, '', args.save_csv_name + '_cls.csv')
    if os.path.exists(save_path):
        result_form = pd.read_csv(save_path, index_col=0)
    else:
        result_form = pd.DataFrame(columns=['dataset_name', 'mean_accu', 'train_time'])

    result_form = pd.concat([result_form, pd.DataFrame([{'dataset_name': args.dataset, 'mean_accu': '%.4f' % mean_accu, 'train_time': '%.4f' % train_time}])], ignore_index=True)

    result_form.to_csv(save_path, index=True, index_label="id")

