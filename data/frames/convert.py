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