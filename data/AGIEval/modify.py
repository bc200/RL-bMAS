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