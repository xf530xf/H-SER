"""Summarize a H²SER classification-results CSV."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', nargs='?', type=Path,
                        default=Path(__file__).parent / 'results' / 'main_h2ser_results_cls.csv')
    args = parser.parse_args()
    if not args.csv_path.is_file():
        parser.error(f'Results CSV does not exist: {args.csv_path}')
    df = pd.read_csv(args.csv_path)
    if 'mean_accu' not in df.columns:
        parser.error('CSV 文件中未找到 mean_accu 列')
    values = pd.to_numeric(df['mean_accu'], errors='coerce')
    if not values.notna().any():
        parser.error('CSV 文件中没有有效的 mean_accu 数值')
    print(f'mean_accu 的平均值为: {values.mean():.6f}')


if __name__ == '__main__':
    main()
