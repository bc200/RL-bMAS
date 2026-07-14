import json
import pandas as pd

# 文件名
jsonl_file = 'test.jsonl'

# 决策提示模板，可以根据需要修改文本
def generate_decision_prompt(answer_type):
    if answer_type == 'exactMatch':
        return "Output your answers in the format:\n\\{{the final answer is boxed[answer]}}"
    elif answer_type == 'multipleChoice':
        return "Output your answers in the format:\n\\{{the final answer is boxed[answer]}}\nFor instance, if there were 7 options and the answers were G, the output would be:\n\\{{the final answer is boxed[G]}}"


# 读取 jsonl 文件
data = []
with open(jsonl_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            if 'question' in obj:
                obj['task'] = obj.get('question', '')
            obj['decision_prompt'] = "Output your answers in the format:\n\\{{the final answer is boxed[answer]}}"
            obj['answer_type'] = 'single'
            data.append(obj)
            # # 添加 decision_prompt
            # obj['decision_prompt'] = generate_decision_prompt(atype)
            # if atype == 'exactMatch':
            #     obj['answer_type'] = 'single'
            # elif atype == 'multipleChoice':
            #     obj['answer_type'] = 'multiple'
            # data.append(obj)

# 转成 DataFrame
df = pd.DataFrame(data)

# 切片
train_df = df.iloc[:200]
test_df = df.iloc[200:800]

# 保存成 Parquet
train_df.to_parquet('train.parquet', index=False)
test_df.to_parquet('test.parquet', index=False)

print(f"已生成 train.parquet({len(train_df)}条) 和 test.parquet({len(test_df)}条)")