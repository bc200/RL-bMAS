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

import pandas as pd

def convert_with_pandas(input_file, output_file):
    try:
        # 读取 tsv 文件
        df = pd.read_csv(input_file, sep='\t')
        
        # 重命名列：把 prompt 换成 question
        df = df.rename(columns={'Prompt': 'question'})
        df = df.rename(columns={'Answer': 'answer'})
        
        # 只保留 question 和 answer 列
        # 如果还有其他列，这一步会将其过滤掉
        df_filtered = df[['question', 'answer']]
        
        # 导出为 jsonl 格式
        # orient='records' 表示每行是一个对象，lines=True 表示每行一个 json
        df_filtered.to_json(output_file, orient='records', lines=True, force_ascii=False)
        
        print(f"转换成功：{output_file}")
        
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    convert_with_pandas('test.tsv', 'test.jsonl')