import numpy as np
from pathlib import Path


def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a


def fill_out_with_Nan(data, max_length):
    pad_length = max_length - data.shape[-1]
    if pad_length < 0:
        raise ValueError('max_length cannot be shorter than the input')
    if pad_length == 0:
        return data
    else:
        pad_shape = list(data.shape[:-1])
        pad_shape.append(pad_length)
        Nan_pad = np.full(pad_shape, np.nan)
        return np.concatenate((data, Nan_pad), axis=-1)


def get_label_dict(file_path):
    with open(file_path, encoding="utf-8-sig") as file:
        for line in file:
            parts = line.strip().split()
            if parts and parts[0].lower() == '@classlabel':
                if len(parts) < 3 or parts[1].lower() != 'true':
                    raise ValueError(f'{file_path}: expected @classLabel true followed by labels')
                return {label: i for i, label in enumerate(parts[2:])}
    raise ValueError(f'{file_path}: missing @classLabel header')


def get_data_and_label_from_ts_file(file_path, label_dict):
    samples, labels = [], []
    reading_data = False
    num_channels = None
    with open(file_path, encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if not reading_data:
                if line.lower().startswith('@timestamps') and line.split()[-1].lower() == 'true':
                    raise ValueError(f'{file_path}: timestamped .ts data is not supported')
                if line.lower() == '@data':
                    reading_data = True
                continue
            parts = [part.strip() for part in line.split(':')]
            if len(parts) < 2 or parts[-1] not in label_dict:
                raise ValueError(f'{file_path}:{line_number}: invalid data row or unknown label')
            channels = []
            for channel in parts[:-1]:
                values = [float(value) if value.strip() != '?' else np.nan
                          for value in channel.split(',')] if channel else [np.nan]
                channels.append(np.asarray(values, dtype=np.float32))
            if num_channels is None:
                num_channels = len(channels)
            if len(channels) != num_channels:
                raise ValueError(f'{file_path}:{line_number}: inconsistent number of channels')
            length = max(len(channel) for channel in channels)
            samples.append(np.stack([fill_out_with_Nan(channel, length) for channel in channels]))
            labels.append(label_dict[parts[-1]])
    if not samples:
        raise ValueError(f'{file_path}: no samples found after @data')
    max_length = max(sample.shape[-1] for sample in samples)
    return (np.stack([fill_out_with_Nan(sample, max_length) for sample in samples]).astype(np.float32),
            np.asarray(labels, dtype=np.int64))


def TSC_multivariate_data_loader(dataset_path, dataset_name):
    directory = Path(dataset_path) / dataset_name
    Train_dataset_path = directory / f'{dataset_name}_TRAIN.ts'
    Test_dataset_path = directory / f'{dataset_name}_TEST.ts'
    label_dict = get_label_dict(Train_dataset_path)
    X_train, y_train = get_data_and_label_from_ts_file(Train_dataset_path, label_dict)
    X_test, y_test = get_data_and_label_from_ts_file(Test_dataset_path, label_dict)

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    dataset_path = './dataset/General/'
    dataset_name = 'JapaneseVowels'
    X_train, y_train, X_test, y_test = TSC_multivariate_data_loader(dataset_path, dataset_name)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print(y_test.dtype)
