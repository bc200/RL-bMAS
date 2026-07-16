
"""
Preprocess the QA dataset to parquet format
"""

import os
#import datasets
import json
from glob import glob
import pandas as pd

from verl.utils.hdfs_io import copy, makedirs
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/verl-agent/data/')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--dataset_name', default="musique")
    parser.add_argument('--train_data_size', default=128, type=int)
    parser.add_argument('--val_data_size', default=256, type=int)

    args = parser.parse_args()

    #data_source = args.dataset_name
    args.local_dir = os.path.join(args.local_dir, args.dataset_name)
    #print('data_path:',os.path.join(args.local_dir, 'musique_500.jsonl'))
    df = pd.read_json(os.path.join(args.local_dir, 'hotpotqa_500.jsonl'), lines=True)
    question_num = len(df)
    print(args.dataset_name+" question number:", question_num)

    df = df.rename(columns={"question": "task"})
    #在df中添加一列 decision_prompt，内容为"\nOutput your answers in the format:\n\box{}\nFor instance, if there were 7 options and the answers were G, the output would be:\n\box{G}\n"
    df['decision_prompt'] = "\nReturn your answers in the format:\n\\{{the final answer is :<your answer here>}} at the end of your response.\n"
    df['answer_type'] = 'single'
    df['data_source'] = args.dataset_name
    df['category'] = 'multi-hop'
    df['ability'] = 'agent'
    df['prompt'] = [{
        "role": "user",
        "content": row['task']
    } for _, row in df.iterrows()]
    df = df[["task", "answer", "decision_prompt", "category", "prompt", "data_source", "ability", "answer_type"]]

    train_dataset = df[:100]
    test_dataset = df[100:500]


    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
