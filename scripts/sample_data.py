"""从原始 CSV 中抽取小样本。"""

import argparse
from pathlib import Path

import pandas as pd


def sample_first_n_rows(input_file, output_file, n_rows=10000):
    """从输入 CSV 中抽取前 n_rows 条记录并保存。"""
    print(f"正在从 {input_file} 抽取前 {n_rows} 条数据...")

    sampled_df = pd.read_csv(input_file, nrows=n_rows)
    print(f"读取完成，shape: {sampled_df.shape}")

    sampled_df.to_csv(output_file, index=False)
    print(f"已保存到: {output_file}")

    print("\n前5行数据:")
    print(sampled_df.head())
    return sampled_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--rows", type=int, default=10000)
    args = parser.parse_args()
    sample_first_n_rows(
        input_file=args.input_file,
        output_file=args.output_file,
        n_rows=args.rows,
    )
