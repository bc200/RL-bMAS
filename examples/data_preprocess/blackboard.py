# Copyright 2026 RL-bMAS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
    parser.add_argument('--dataset_name', default="MMLU-Pro-subset")
    parser.add_argument('--train_data_size', default=128, type=int)
    parser.add_argument('--val_data_size', default=256, type=int)

    args = parser.parse_args()

    #data_source = args.dataset_name
    args.local_dir = os.path.join(args.local_dir, args.dataset_name)

    #dataset = datasets.load_dataset(data_source)
    #train_data_size = 128
    question_datas = []
    #读取csv
    #df = pd.read_csv(os.path.join(args.local_dir, "hle-choice.csv"))
    #读取jsonl
    df = pd.read_json(os.path.join(args.local_dir, "test.jsonl"), lines=True)
    question_num = len(df)
    print("MMLU-Pro question number:", question_num)
    #提取出dataframe数据中question和groundtruth的数据，将question改名成task，groundtruth改名成answer
    #df = df.rename(columns={"question": "task", "groundtruth": "answer", "task": "category"})
    df = df.rename(columns={"question": "task"})
    #在df中添加一列 decision_prompt，内容为"\nOutput your answers in the format:\n\box{}\nFor instance, if there were 7 options and the answers were G, the output would be:\n\box{G}\n"
    #For instance, if there were 7 options and the answers were G, the output would be:\n\\{{the final answer is boxed[C]}}   This is a Single choice question question. 
    df['decision_prompt'] = "There is one and only one correct answer. \nOutput your answers in the format:\n\\{{the final answer is boxed[answer]}} at the end of your response.\n"
    df['answer_type'] = 'single'
    df['category'] = 'rbench'
    df['data_source'] = args.dataset_name
    df['ability'] = 'agent'
    df['prompt'] = [{
        "role": "user",
        "content": row['task']
    } for _, row in df.iterrows()]
    df = df[["task", "answer", "decision_prompt", "category", "prompt", "data_source", "ability", 'answer_type']]

    #用0-200和600-800作为train
    #train_dataset = pd.concat([df[0:200], df[600:800]], ignore_index=True)
    train_dataset = df[0:200]
    #用200-600作为test
    test_dataset = df[200:800]


    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
