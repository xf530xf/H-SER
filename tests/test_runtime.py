"""Runtime regression checks using synthetic data; no benchmark accuracy claims."""
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.preprocessing import fill_nan_value, normalize_train_val_test
from models.H2SERModel import H2SERNet, coml_index
from models.H2SERModel_copy import H2SERNet as AlternateNet, HypergraphConv
from ts_utils import build_uea_dataset, evaluate_model, save_cls_result

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


@pytest.mark.parametrize('net', [H2SERNet, AlternateNet])
@pytest.mark.parametrize('length,width,channels,rate', [
    (128, 128, 1, .5), (24, 64, 3, .9), (1, 1, 1, 1.), (32, 32, 1, 0.),
])
def test_forward_backward(net, length, width, channels, rate):
    model = net(length, 8, channels, width, rate, 2, 3, num_experts=3, stride=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for epoch in (0, 2):
        optimizer.zero_grad()
        logits, aux = model(torch.randn(2, channels, length), num_epoch_i=epoch, warm_up_epoch=1)
        assert logits.shape == (2, 3)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])) + .001 * aux
        assert torch.isfinite(loss)
        loss.backward()
        assert model.head.weight.grad is not None
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    if net is H2SERNet:
        model.eval()
        with torch.no_grad():
            logits, _, embeddings = model(torch.randn(1, channels, length), return_intermediates=True)
        assert logits.shape == (1, 3)
        assert set(embeddings) == {'input', 'intra', 'inter', 'output'}
        assert embeddings['input'].shape[1] == model.shape_embed.num_patches


def test_hypergraph_projection_receives_gradients():
    module = HypergraphConv(4, k=3, dropout=0.)
    module(torch.randn(2, 5, 4)).square().sum().backward()
    assert module.proj.weight.grad is not None
    assert torch.isfinite(module.proj.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware unavailable')
def test_cuda_complement_indices():
    idx = torch.tensor([[1, 3]], device='cuda')
    result = coml_index(idx, 5)
    assert result.device == idx.device
    assert result.tolist() == [[0, 2, 4]]


def test_multichannel_nan_fill_and_constant_normalization():
    train = np.array([[[1., 10.], [3., 30.]], [[5., np.nan], [np.nan, 70.]]])
    val = np.full((1, 2, 2), np.nan)
    train, val, test = fill_nan_value(train, val, val.copy())
    np.testing.assert_allclose(train[1], [[5., 10.], [3., 70.]])
    np.testing.assert_allclose(val[0], [[3., 10.], [3., 50.]])
    assert np.isfinite(test).all()
    for split in normalize_train_val_test(np.ones((2, 2)), np.ones((1, 2)), np.ones((1, 2))):
        np.testing.assert_array_equal(split, np.zeros_like(split))


def test_evaluation_loss_is_independent_of_batch_size():
    class FixedModel(torch.nn.Module):
        def forward(self, x, num_epoch_i):
            assert num_epoch_i == 7
            return x, 0.
    logits = torch.tensor([[2., -1.], [0., 3.], [-1., 1.], [3., 0.], [2., 1.]])
    labels = torch.tensor([0, 1, 0, 1, 1])
    criterion = torch.nn.CrossEntropyLoss()
    expected = criterion(logits, labels).item()
    for batch_size in [1, 2, 5]:
        loader = DataLoader(TensorDataset(logits, labels), batch_size=batch_size)
        actual, acc = evaluate_model(loader, FixedModel(), criterion, num_epoch_i=7)
        assert actual == pytest.approx(expected)
        assert acc == pytest.approx(.4)


def test_results_create_parent_directory(tmp_path):
    args = SimpleNamespace(save_dir=tmp_path / 'nested' / 'results', save_csv_name='h2ser', dataset='Toy')
    save_cls_result(args, .75, 2.)
    save_cls_result(args, .5, 3.)
    frame = pd.read_csv(args.save_dir / 'h2ser_cls.csv')
    assert frame.mean_accu.tolist() == [.75, .5]


@pytest.fixture
def synthetic_data(tmp_path):
    rng = np.random.default_rng(42)
    ucr = tmp_path / 'ucr' / 'ToyUCR'
    uea = tmp_path / 'uea' / 'ToyUEA'
    ucr.mkdir(parents=True)
    uea.mkdir(parents=True)
    for split, count, length in [('TRAIN', 30, 24), ('TEST', 20, 20)]:
        labels = np.arange(count) % 2
        x = rng.normal(size=(count, 24))
        np.savetxt(ucr / f'ToyUCR_{split}.tsv', np.column_stack([labels, x]), delimiter='\t')
        rows = ['@problemName ToyUEA', '@timestamps false', '@classLabel true a b', '@data', '# comment', '']
        for i, label in enumerate(labels):
            channels = []
            for c in range(3):
                values = [str(v) for v in rng.normal(size=length)]
                if i == 0 and c == 1:
                    values[3] = '?'
                channels.append(','.join(values))
            rows.append(':'.join(channels + ['ab'[label]]))
        (uea / f'ToyUEA_{split}.ts').write_text('\n'.join(rows) + '\n')
    return tmp_path


def test_uea_loading_preserves_channels_and_missing_values(synthetic_data):
    args = SimpleNamespace(dataroot=synthetic_data / 'uea', dataset='ToyUEA')
    data, labels, count = build_uea_dataset(args)
    assert data.shape == (50, 24, 3)
    assert np.isnan(data[0, 3, 1])
    assert np.isnan(data[30:, 20:, :]).all()
    assert labels.shape == (50,) and count == 2


@pytest.mark.parametrize('entry', ['main_h2ser.py', 'train-net.py', 'train-uea.py'])
def test_training_entry_points_and_fold_reset(entry, synthetic_data, monkeypatch, capsys):
    is_uea = entry == 'train-uea.py'
    output = synthetic_data / entry
    argv = [entry, '--dataset', 'ToyUEA' if is_uea else 'ToyUCR',
            '--dataroot', str(synthetic_data / ('uea' if is_uea else 'ucr')),
            '--save_dir', str(output), '--epoch', '2', '--warm_up_epoch', '1',
            '--cuda', 'cpu', '--emb_dim', '16', '--use_large_batch', '0', '--batch_size', '8',
            '--vis_embeddings', '1', '--vis_fold', '0', '--vis_max_points', '16']
    if entry == 'train-net.py':
        argv += ['--vis_featuremaps', '1', '--vis_feature_epochs', '0', '--vis_feature_max_tokens', '8']
    monkeypatch.setattr(sys, 'argv', argv)
    original = torch.optim.Adam.load_state_dict
    fold_parameters = []
    def track_reset(optimizer, state):
        assert state['state'] == {}, 'Each fold must start with fresh optimizer moments'
        fold_parameters.append([p.detach().clone() for group in optimizer.param_groups for p in group['params']])
        return original(optimizer, state)
    monkeypatch.setattr(torch.optim.Adam, 'load_state_dict', track_reset)
    runpy.run_path(str(ROOT / entry), run_name='__main__')
    assert len(fold_parameters) == 5
    for snapshot in fold_parameters[1:]:
        assert all(torch.equal(a, b) for a, b in zip(snapshot, fold_parameters[0]))
    captured = capsys.readouterr().out
    assert 'failed' not in captured.lower(), captured
    assert 'Done!' in captured
    frame = pd.read_csv(output / 'h2ser_five_fold_split_cls.csv')
    assert len(frame) == 1 and 0 <= frame.mean_accu.iloc[0] <= 1
    assert list((output / 'vis_embeddings').glob('*.png'))
    if entry == 'train-net.py':
        assert list((output / 'vis_featuremaps').rglob('*.png'))


@pytest.mark.parametrize('entry', [
    'main_h2ser.py', 'train-net.py', 'train-uea.py', 'module_contribution.py',
    'compute_params_flops.py', 'test.py', 'scripts/generate.py',
])
def test_cli_help(entry):
    result = subprocess.run([sys.executable, str(ROOT / entry), '--help'],
                            text=True, capture_output=True, timeout=60, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout
