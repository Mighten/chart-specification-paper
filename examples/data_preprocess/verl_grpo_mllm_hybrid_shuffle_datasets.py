import pandas as pd
import os
from datetime import datetime


local_dir = f'/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/hybrid_mllm-ablation_chartqa_no_cot_{datetime.now():%Y%m%d_%H%M%S}'
os.makedirs(local_dir, exist_ok=True)


parquet_filepath_list = [
    # 带 CoT ChartQA
    # '/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/chartqa_mllm-20250627_194023/chartqa_human_train.parquet',
    
    # 消融实验1：不带 CoT ChartQA + 带 CoT chart2code
    '/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/chartqa_mllm_no_cot_20250901_185703/chartqa_human_train.parquet',
    '/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/char2code_mllm-20250730_103514/chart2code_combo_3k_train.parquet',
]


if __name__ == '__main__':
    ans = pd.DataFrame()

    # type of each item: <class 'pandas.core.frame.DataFrame'>
    df_list = [pd.read_parquet(parquet_filepath) for parquet_filepath in parquet_filepath_list]    
    ans = pd.concat(df_list)
    # 打乱所有行（默认保留原索引）

    random_seed = 42
    shuffled_ans = ans.sample(frac=1, random_state=random_seed)
    print(shuffled_ans)

    train_parquet_filepath = os.path.join(local_dir, 'hybrid_train.parquet')
    shuffled_ans.to_parquet(train_parquet_filepath)
    shuffled_ans.to_csv(f'{train_parquet_filepath}.csv', index=False)
    print(f'Output Hybrid Training Apache Parquet(ChartQA without CoT, chart2code) dataset filepath:', train_parquet_filepath)
