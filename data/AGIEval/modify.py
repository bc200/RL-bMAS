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

import json
import csv
import os

# 文件路径
jsonl_path = os.path.join(os.path.dirname(__file__), 'logiqa-en.jsonl')
csv_path = os.path.join(os.path.dirname(__file__), 'logiqa-en.csv')

with open(jsonl_path, 'r', encoding='utf-8') as fin, open(csv_path, 'w', encoding='utf-8', newline='') as fout:
    writer = csv.writer(fout)
    writer.writerow(['question', 'answer', 'category'])  # 写入表头

    for line in fin:
        data = json.loads(line)
        # 合并 passage, question, options
        passage = data.get('passage', '')
        question = data.get('question', '')
        options = data.get('options', [])
        options_str = '\n'.join([f"{opt}" for opt in options])
        merged_question = f"{passage}\n{question}\nOptions:\n{options_str}".strip()
        answer = data.get('label', '')
        category = 'logical reasoning'
        writer.writerow([merged_question, answer, category])