"""
从原始数据中抽取前 N 条数据
用于小样本测试
"""

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
    input_file = "data\\ctr_data_1M.csv"
    output_file = "data\\ctr_data_500k.csv"

    sampled_df = sample_first_n_rows(
        input_file=input_file,
        output_file=output_file,
        n_rows=500000,
    )
    print("\n完成!")
